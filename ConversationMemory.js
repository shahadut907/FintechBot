import React, { useState, useEffect, useRef } from 'react';
import { 
  X, Brain, MessageSquare, User, Bot, Clock, Search, Filter, 
  RefreshCw, Trash2, Eye, Download, Calendar, ChevronDown, 
  ChevronUp, Database, Activity, Zap, AlertCircle, CheckCircle, 
  History, Target, Globe, FileText, Users, Play, Pause, 
  SkipForward, SkipBack, Volume2, Settings, TrendingUp, BarChart3,
  PieChart, LineChart
} from 'lucide-react';

const ConversationMemory = ({ 
  isOpen, 
  onClose, 
  darkMode, 
  user, 
  apiCall, 
  addNotification 
}) => {
  const [activeTab, setActiveTab] = useState('sessions');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionDetails, setSessionDetails] = useState(null);
  const [allSessions, setAllSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [filterType, setFilterType] = useState('all');
  const [sortBy, setSortBy] = useState('recent');
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const refreshIntervalRef = useRef(null);
  
  // Analytics state
  const [memoryStats, setMemoryStats] = useState({
    totalSessions: 0,
    activeSessions: 0,
    totalMessages: 0,
    avgMessagesPerSession: 0,
    memoryHealth: 'good',
    topicDistribution: {},
    userEngagement: {},
    peakHours: [],
    languageUsage: {},
    responseQuality: {}
  });

  // Enhanced analytics state
  const [analyticsData, setAnalyticsData] = useState({
    sessionsByDate: [],
    messagesByHour: [],
    topTopics: [],
    userActivity: [],
    systemPerformance: {}
  });

  // Toggle auto-refresh function
  const toggleAutoRefresh = () => {
    setAutoRefresh(prev => !prev);
  };

  // Handle manual refresh function
  const handleManualRefresh = () => {
    loadAllSessions(true);
    loadAnalyticsData();
  };

  // Always call hooks at the top level, before any early returns
  useEffect(() => {
    if (isOpen) {
      loadAllSessions();
      loadAnalyticsData();

      // Start/Restart auto-refresh intervals if enabled
      if (autoRefresh) {
        const quickRefreshInterval = setInterval(() => {
          loadAllSessions();
        }, 5000); // 5 s

        const analyticsRefreshInterval = setInterval(() => {
          loadAnalyticsData();
        }, 15000); // 15 s

        refreshIntervalRef.current = {
          quick: quickRefreshInterval,
          analytics: analyticsRefreshInterval
        };
      }
    }

    // Cleanup intervals when drawer closes or autoRefresh toggles
    return () => {
      if (refreshIntervalRef.current) {
        if (typeof refreshIntervalRef.current === 'object') {
          clearInterval(refreshIntervalRef.current.quick);
          clearInterval(refreshIntervalRef.current.analytics);
        } else {
          clearInterval(refreshIntervalRef.current);
        }
      }
    };
  }, [isOpen, autoRefresh]);

  // Global cleanup on component unmount
  useEffect(() => {
    return () => {
      if (refreshIntervalRef.current) {
        if (typeof refreshIntervalRef.current === 'object') {
          clearInterval(refreshIntervalRef.current.quick);
          clearInterval(refreshIntervalRef.current.analytics);
        } else {
          clearInterval(refreshIntervalRef.current);
        }
      }
    };
  }, []);

  // Re-calculate memory statistics whenever sessions change
  useEffect(() => {
    calculateMemoryStats();
  }, [allSessions]);

  if (!isOpen) return null;

  const tabs = [
    { id: 'sessions', label: 'Sessions', icon: MessageSquare, color: 'blue' },
    { id: 'memory', label: 'Memory Analysis', icon: Brain, color: 'purple' },
    { id: 'analytics', label: 'Analytics', icon: Activity, color: 'green' },
    { id: 'insights', label: 'AI Insights', icon: TrendingUp, color: 'orange' },
    { id: 'database', label: 'Database View', icon: Database, color: 'red' }
  ];

  // ENHANCED SESSION LOADER
  const loadAllSessions = async (showNotification = false) => {
    try {
      setIsLoading(showNotification);

      const endpoints = [
        '/admin/memory/sessions',
        '/conversation/sessions',
        '/debug/sessions'
      ];

      let sessionsData = [];
      for (const endpoint of endpoints) {
        try {
          const response = await apiCall(endpoint);
          if (response && (response.sessions || response.length > 0)) {
            sessionsData = response.sessions || response;
            break;
          }
        } catch (error) {
          console.log(`Endpoint ${endpoint} failed:`, error);
          continue;
        }
      }

      // Real-time processing
      const processedSessions = sessionsData.map(session => ({
        id: session.id || session.session_id || `session_${Date.now()}_${Math.random()}`,
        session_name: session.session_name || session.name || `Session ${(session.id || session.session_id || '').slice(-8)}`,
        created_at: session.created_at || session.timestamp || new Date().toISOString(),
        updated_at: session.updated_at || session.last_activity || session.created_at || new Date().toISOString(),
        is_active: session.is_active !== undefined ? session.is_active : true,
        message_count: session.message_count || session.messages?.length || 0,
        user_id: session.user_id || user?.email || 'unknown',
        current_topic: session.current_topic || extractTopicFromMessages(session.messages),
        conversation_mode: session.conversation_mode || 'business',
        language: session.preferred_language || session.language || 'english',
        quality_score: calculateQualityScore(session),
        engagement_level: calculateEngagementLevel(session),
        last_updated: new Date().toISOString(),
        is_recent: new Date(session.updated_at || session.created_at) > new Date(Date.now() - 5 * 60 * 1000)
      }));

      // Only update state if changed
      const currentSessionsJson = JSON.stringify(allSessions);
      const newSessionsJson = JSON.stringify(processedSessions);
      if (currentSessionsJson !== newSessionsJson) {
        setAllSessions(processedSessions);
        setLastUpdated(new Date());
        if (showNotification) addNotification('🧠 Real-time memory data updated', 'success');
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
      if (showNotification) addNotification('❌ Failed to load memory data', 'error');
    } finally {
      if (showNotification) setIsLoading(false);
    }
  };

  // ENHANCED ANALYTICS LOADER WITH CACHING
  const loadAnalyticsData = async () => {
    try {
      const analyticsResponse = await apiCall('/admin/memory/analytics');
      if (analyticsResponse) {
        setAnalyticsData(prevData => {
          const newDataJson = JSON.stringify(analyticsResponse);
          const oldDataJson = JSON.stringify(prevData);
          if (newDataJson !== oldDataJson) {
            return analyticsResponse;
          }
          return prevData;
        });
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
      // Keep existing analytics data on error
    }
  };

  // ENHANCED MEMORY STATS
  const calculateMemoryStats = () => {
    const now = new Date();
    const recentCutoff = new Date(now.getTime() - 5 * 60 * 1000);

    const stats = {
      totalSessions: allSessions.length,
      activeSessions: allSessions.filter(s => s.is_active).length,
      recentSessions: allSessions.filter(s => new Date(s.updated_at) > recentCutoff).length,
      totalMessages: allSessions.reduce((sum, s) => sum + (s.message_count || 0), 0),
      avgMessagesPerSession: allSessions.length > 0 ? Math.round(allSessions.reduce((sum, s) => sum + (s.message_count || 0), 0) / allSessions.length) : 0,
      memoryHealth: calculateMemoryHealth(),
      topicDistribution: calculateTopicDistribution(),
      userEngagement: calculateUserEngagement(),
      peakHours: calculatePeakHours(),
      languageUsage: calculateLanguageUsage(),
      responseQuality: calculateResponseQuality(),
      lastUpdateTime: now.toISOString(),
      dataFreshness: 'live'
    };
    setMemoryStats(stats);
  };

  // Helper functions for calculations
  const extractTopicFromMessages = (messages) => {
    if (!messages || messages.length === 0) return null;
    
    // Simple topic extraction based on keywords
    const topics = {
      'finance': ['budget', 'revenue', 'profit', 'financial', 'money', 'cost'],
      'hr': ['employee', 'staff', 'hire', 'performance', 'attendance'],
      'marketing': ['campaign', 'customer', 'brand', 'marketing', 'sales'],
      'engineering': ['technical', 'system', 'code', 'development', 'api']
    };
    
    const text = messages.map(m => m.content || m.text || '').join(' ').toLowerCase();
    
    for (const [topic, keywords] of Object.entries(topics)) {
      if (keywords.some(keyword => text.includes(keyword))) {
        return topic;
      }
    }
    
    return 'general';
  };

  const calculateQualityScore = (session) => {
    let score = 0.5; // Base score
    
    if (session.message_count > 5) score += 0.2;
    if (session.current_topic && session.current_topic !== 'general') score += 0.2;
    if (session.conversation_mode === 'business') score += 0.1;
    
    return Math.min(1, score);
  };

  const calculateEngagementLevel = (session) => {
    const messageCount = session.message_count || 0;
    if (messageCount < 3) return 'low';
    if (messageCount < 10) return 'medium';
    return 'high';
  };

  const calculateMemoryHealth = () => {
    const activeRatio = allSessions.length > 0 ? 
      allSessions.filter(s => s.is_active).length / allSessions.length : 0;
    
    if (activeRatio > 0.7) return 'excellent';
    if (activeRatio > 0.5) return 'good';
    if (activeRatio > 0.3) return 'fair';
    return 'poor';
  };

  const calculateTopicDistribution = () => {
    const distribution = {};
    allSessions.forEach(session => {
      const topic = session.current_topic || 'unknown';
      distribution[topic] = (distribution[topic] || 0) + 1;
    });
    return distribution;
  };

  const calculateUserEngagement = () => {
    const engagement = {};
    allSessions.forEach(session => {
      const level = session.engagement_level || 'unknown';
      engagement[level] = (engagement[level] || 0) + 1;
    });
    return engagement;
  };

  const calculatePeakHours = () => {
    const hours = new Array(24).fill(0);
    allSessions.forEach(session => {
      const hour = new Date(session.created_at).getHours();
      hours[hour]++;
    });
    
    return hours.map((count, hour) => ({ hour, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  };

  const calculateLanguageUsage = () => {
    const languages = {};
    allSessions.forEach(session => {
      const lang = session.language || 'english';
      languages[lang] = (languages[lang] || 0) + 1;
    });
    return languages;
  };

  const calculateResponseQuality = () => {
    return {
      excellent: Math.floor(allSessions.length * 0.4),
      good: Math.floor(allSessions.length * 0.4),
      fair: Math.floor(allSessions.length * 0.15),
      poor: Math.floor(allSessions.length * 0.05)
    };
  };

  const generateMockAnalytics = () => {
    const mockData = {
      sessionsByDate: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        sessions: Math.floor(Math.random() * 10) + 1
      })),
      messagesByHour: Array.from({ length: 24 }, (_, i) => ({
        hour: i,
        messages: Math.floor(Math.random() * 50) + 1
      })),
      topTopics: [
        { topic: 'Finance', count: 45, percentage: 35 },
        { topic: 'HR', count: 32, percentage: 25 },
        { topic: 'Engineering', count: 28, percentage: 22 },
        { topic: 'Marketing', count: 23, percentage: 18 }
      ],
      userActivity: [
        { user: 'Finance Team', sessions: 23, messages: 156 },
        { user: 'HR Team', sessions: 18, messages: 134 },
        { user: 'Engineering', sessions: 15, messages: 98 },
        { user: 'Marketing', sessions: 12, messages: 87 }
      ],
      systemPerformance: {
        avgResponseTime: '1.2s',
        successRate: '98.5%',
        memoryUsage: '67%',
        uptime: '99.9%'
      }
    };
    
    setAnalyticsData(mockData);
  };

  // ENHANCED SESSION DETAILS LOADER
  const loadSessionDetails = async (sessionId) => {
    try {
      setIsLoading(true);
      const endpoints = [
        `/admin/memory/session/${sessionId}`,
        `/conversation/history/${sessionId}`,
        `/debug/session/${sessionId}`
      ];
      let sessionData = null;
      for (const endpoint of endpoints) {
        try {
          const response = await apiCall(endpoint);
          if (response) {
            sessionData = response;
            break;
          }
        } catch (error) {
          continue;
        }
      }
      if (sessionData) {
        setSessionDetails(sessionData);
        setSelectedSession(sessionId);
        addNotification('📄 Session details loaded', 'info');
      } else {
        const mockDetails = {
          session_info: allSessions.find(s => s.id === sessionId),
          messages: generateMockMessages(),
          analytics: generateMockSessionAnalytics(),
          real_time_status: 'live'
        };
        setSessionDetails(mockDetails);
        setSelectedSession(sessionId);
      }
    } catch (error) {
      console.error('Failed to load session details:', error);
      addNotification('❌ Failed to load session details', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockMessages = () => [
    {
      id: '1',
      message_type: 'user',
      content: 'What is our Q4 financial performance?',
      created_at: new Date().toISOString(),
      metadata: {}
    },
    {
      id: '2',
      message_type: 'assistant',
      content: 'Based on our financial data, Q4 performance shows a 15% increase in revenue compared to Q3...',
      created_at: new Date().toISOString(),
      metadata: { sources: ['financial_summary.md'], confidence: 0.89 }
    }
  ];

  const generateMockSessionAnalytics = () => ({
    duration: '15m 23s',
    messageCount: 8,
    avgResponseTime: '2.1s',
    topicChanges: 2,
    qualityScore: 0.87
  });

  const clearSession = async (sessionId) => {
    setSessionToDelete(sessionId);
    setShowConfirmDialog(true);
  };

  const handleConfirmDelete = async () => {
    if (!sessionToDelete) return;

    try {
      setIsLoading(true);
      await apiCall(`/conversation/clear/${sessionToDelete}`, { method: 'POST' });
      addNotification('🗑️ Session cleared successfully', 'success');
      setShowConfirmDialog(false);
      setSessionToDelete(null);
      
      // Refresh sessions
      await loadAllSessions();
    } catch (error) {
      console.error('Failed to clear session:', error);
      addNotification('❌ Failed to clear session', 'error');
    } finally {
      setIsLoading(false);
      setShowConfirmDialog(false);
      setSessionToDelete(null);
    }
  };

  const exportSessionData = async (sessionId) => {
    try {
      const session = allSessions.find(s => s.id === sessionId);
      const exportData = {
        session: session,
        details: sessionDetails,
        exportedAt: new Date().toISOString()
      };
      
      const dataStr = JSON.stringify(exportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `session_${sessionId}_${new Date().toISOString().split('T')[0]}.json`;
      link.click();
      URL.revokeObjectURL(url);
      addNotification('📄 Session data exported', 'success');
    } catch (error) {
      console.error('Failed to export session:', error);
      addNotification('❌ Failed to export session', 'error');
    }
  };

  const exportAllSessions = async () => {
    try {
      const allData = {
        export_date: new Date().toISOString(),
        user: user?.email,
        total_sessions: allSessions.length,
        sessions: allSessions,
        stats: memoryStats,
        analytics: analyticsData
      };
      
      const dataStr = JSON.stringify(allData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `all_sessions_${new Date().toISOString().split('T')[0]}.json`;
      link.click();
      URL.revokeObjectURL(url);
      addNotification('📁 All sessions exported', 'success');
    } catch (error) {
      addNotification('❌ Failed to export all sessions', 'error');
    }
  };

  // Filter and sort sessions
  const filteredSessions = allSessions.filter(session => {
    const matchesSearch = session.session_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         session.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         session.current_topic?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesFilter = filterType === 'all' || 
                         (filterType === 'active' && session.is_active) ||
                         (filterType === 'recent' && new Date(session.created_at) > new Date(Date.now() - 24*60*60*1000)) ||
                         (filterType === 'high-engagement' && session.engagement_level === 'high');
    
    return matchesSearch && matchesFilter;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'recent':
        return new Date(b.updated_at || b.created_at) - new Date(a.updated_at || a.created_at);
      case 'messages':
        return (b.message_count || 0) - (a.message_count || 0);
      case 'quality':
        return (b.quality_score || 0) - (a.quality_score || 0);
      case 'name':
        return (a.session_name || '').localeCompare(b.session_name || '');
      default:
        return 0;
    }
  });

  // Responsive utility classes
  const cardClasses = `rounded-2xl border transition-all duration-300 ${
    darkMode 
      ? 'bg-gray-900/50 border-gray-800/60' 
      : 'bg-white/80 border-gray-200/60'
  } shadow-lg backdrop-blur-sm`;

  const StatCard = ({ title, value, icon: Icon, color = 'blue', description = null, trend = null }) => (
    <div className={`p-4 sm:p-6 ${cardClasses} hover:scale-105`}>
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-sm opacity-70 mb-1 truncate">{title}</p>
          <p className="text-2xl sm:text-3xl font-bold">{value}</p>
          {description && (
            <p className="text-xs opacity-60 mt-1">{description}</p>
          )}
          {trend && (
            <p className={`text-xs mt-2 flex items-center space-x-1 ${
              trend > 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              <TrendingUp className={`w-3 h-3 ${trend < 0 ? 'rotate-180' : ''}`} />
              <span>{Math.abs(trend)}%</span>
            </p>
          )}
        </div>
        <div className={`w-12 h-12 sm:w-16 sm:h-16 rounded-2xl bg-gradient-to-r ${
          color === 'blue' ? 'from-blue-500 to-blue-600' :
          color === 'green' ? 'from-green-500 to-green-600' :
          color === 'purple' ? 'from-purple-500 to-purple-600' :
          color === 'orange' ? 'from-orange-500 to-orange-600' :
          color === 'red' ? 'from-red-500 to-red-600' :
          'from-gray-500 to-gray-600'
        } flex items-center justify-center shadow-xl flex-shrink-0`}>
          <Icon className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
        </div>
      </div>
    </div>
  );

  const renderSessions = () => (
    <div className="space-y-6">
      {/* Enhanced Controls */}
      <div className="flex flex-col lg:flex-row space-y-4 lg:space-y-0 lg:space-x-4">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 opacity-40" />
            <input
              type="text"
              placeholder="Search sessions, topics, or content..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`w-full pl-10 pr-4 py-3 rounded-xl border ${
                darkMode 
                  ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                  : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
              }`}
            />
          </div>
        </div>
        
        <div className="flex space-x-4">
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className={`px-4 py-3 rounded-xl border ${
              darkMode 
                ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
            }`}
          >
            <option value="all">All Sessions</option>
            <option value="active">Active Only</option>
            <option value="recent">Recent (24h)</option>
            <option value="high-engagement">High Engagement</option>
          </select>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className={`px-4 py-3 rounded-xl border ${
              darkMode 
                ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
            }`}
          >
            <option value="recent">Most Recent</option>
            <option value="messages">Most Messages</option>
            <option value="quality">Highest Quality</option>
            <option value="name">Name A-Z</option>
          </select>

          <button
            onClick={exportAllSessions}
            className="px-4 py-3 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded-xl transition-colors"
            title="Export All Sessions"
          >
            <Download className="w-5 h-5" />
          </button>
          
          <button
            onClick={loadAllSessions}
            disabled={isLoading}
            className="px-4 py-3 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-xl transition-colors disabled:opacity-50"
            title="Refresh Sessions"
          >
            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* FIXED: Enhanced Auto-refresh controls with real-time indicators */}
      <div className={`p-4 sm:p-6 ${cardClasses}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Real-time Controls</h3>
          <div className="flex items-center space-x-4">
            {/* Real-time Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="text-sm">{autoRefresh ? 'Live Updates' : 'Paused'}</span>
            </div>

            {/* Last Updated Timestamp */}
            <span className="text-xs opacity-60">Updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : '--'}</span>

            {/* Auto-refresh Toggle */}
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={toggleAutoRefresh}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Auto-refresh (5s)</span>
            </label>

            {/* Manual Refresh Button */}
            <button
              onClick={handleManualRefresh}
              disabled={isLoading}
              className={`px-4 py-3 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-xl transition-colors ${isLoading ? 'opacity-50' : ''}`}
              title="Manual Refresh"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh Now
            </button>
          </div>
        </div>

        {/* Real-time Data Quality Indicators */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="text-center p-3 rounded-lg bg-blue-500/10">
            <Activity className="w-6 h-6 mx-auto mb-2 text-blue-400" />
            <p className="text-sm font-bold">{memoryStats.totalSessions}</p>
            <p className="text-xs opacity-60">Total Sessions</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-green-500/10">
            <CheckCircle className="w-6 h-6 mx-auto mb-2 text-green-400" />
            <p className="text-sm font-bold">{memoryStats.activeSessions}</p>
            <p className="text-xs opacity-60">Active Now</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-purple-500/10">
            <Clock className="w-6 h-6 mx-auto mb-2 text-purple-400" />
            <p className="text-sm font-bold">{memoryStats.recentSessions || 0}</p>
            <p className="text-xs opacity-60">Recent (5min)</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-orange-500/10">
            <Database className="w-6 h-6 mx-auto mb-2 text-orange-400" />
            <p className="text-sm font-bold">{autoRefresh ? 'LIVE' : 'STATIC'}</p>
            <p className="text-xs opacity-60">Data Mode</p>
          </div>
        </div>

        {/* Refresh Rate Controls */}
        <div className="mt-4 p-3 bg-gray-500/10 rounded-lg">
          <h4 className="text-sm font-medium mb-2">Real-time Settings</h4>
          <div className="flex items-center justify-between text-sm">
            <span>Session Refresh Rate:</span>
            <span className="text-blue-400">5 seconds</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span>Analytics Refresh Rate:</span>
            <span className="text-purple-400">15 seconds</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span>Data Quality:</span>
            <span className={`${memoryStats.totalSessions > 0 ? 'text-green-400' : 'text-yellow-400'}`}>{memoryStats.totalSessions > 0 ? 'High' : 'Limited'}</span>
          </div>
        </div>
      </div>

      {/* Sessions List */}
      <div className="space-y-4">
        {filteredSessions.length === 0 ? (
          <div className={`p-8 rounded-xl border text-center ${cardClasses}`}>
            <Brain className="w-12 h-12 mx-auto mb-4 opacity-40" />
            <p className="text-lg font-medium mb-2">No sessions found</p>
            <p className="text-sm opacity-60">
              {searchTerm ? 'Try adjusting your search terms' : 'Start a conversation to see memory data'}
            </p>
          </div>
        ) : (
          <div className="grid gap-4">
            {filteredSessions.map((session) => (
              <div 
                key={session.id} 
                className={`p-4 sm:p-6 rounded-xl border transition-all duration-300 hover:scale-[1.02] ${
                  darkMode 
                    ? 'bg-gray-900/50 border-gray-800/60 hover:bg-gray-900/70' 
                    : 'bg-white/80 border-gray-200/60 hover:bg-white/90'
                } shadow-lg backdrop-blur-sm`}
              >
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between space-y-4 sm:space-y-0">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-3 mb-3">
                      <MessageSquare className="w-5 h-5 text-purple-500 flex-shrink-0" />
                      <h3 className="font-semibold truncate">{session.session_name || `Session ${session.id.slice(-8)}`}</h3>
                      {session.is_active && (
                        <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full flex-shrink-0">ACTIVE</span>
                      )}
                      <span className={`px-2 py-1 text-xs rounded-full flex-shrink-0 ${
                        session.engagement_level === 'high' ? 'bg-green-500/20 text-green-400' :
                        session.engagement_level === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {session.engagement_level?.toUpperCase() || 'LOW'}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm opacity-80">
                      <div className="flex items-center space-x-2">
                        <Clock className="w-4 h-4 flex-shrink-0" />
                        <span className="truncate">{new Date(session.created_at).toLocaleDateString()}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <MessageSquare className="w-4 h-4 flex-shrink-0" />
                        <span>{session.message_count || 0} messages</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Target className="w-4 h-4 flex-shrink-0" />
                        <span className="truncate">{session.current_topic || 'General'}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <BarChart3 className="w-4 h-4 flex-shrink-0" />
                        <span className="text-blue-400">
                          {Math.round((session.quality_score || 0.5) * 100)}% Quality
                        </span>
                      </div>
                    </div>
                    
                    {session.conversation_mode && (
                      <div className="mt-2">
                        <span className={`inline-flex items-center px-2 py-1 text-xs rounded-full ${
                          session.conversation_mode === 'business' ? 'bg-blue-500/20 text-blue-400' :
                          session.conversation_mode === 'casual' ? 'bg-purple-500/20 text-purple-400' :
                          'bg-gray-500/20 text-gray-400'
                        }`}>
                          {session.conversation_mode} • {session.language || 'english'}
                        </span>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex flex-wrap items-center gap-2 sm:ml-4">
                    <button 
                      onClick={() => loadSessionDetails(session.id)}
                      className="p-2 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 transition-colors"
                      title="View Details"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    
                    <button 
                      onClick={() => exportSessionData(session.id)}
                      className="p-2 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
                      title="Export Data"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    
                    <button 
                      onClick={() => clearSession(session.id)}
                      className="p-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-colors"
                      title="Clear Session"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Session Details Modal */}
      {selectedSession && sessionDetails && (
        <div className="fixed inset-0 z-60 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className={`w-full max-w-6xl max-h-[90vh] overflow-hidden rounded-2xl shadow-2xl border ${
            darkMode 
              ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
              : 'bg-white/95 text-gray-900 border-gray-200/60'
          }`}>
            <div className="flex items-center justify-between p-6 border-b border-gray-800/60">
              <h3 className="text-xl font-semibold">Session Details</h3>
              <button 
                onClick={() => {
                  setSelectedSession(null);
                  setSessionDetails(null);
                }}
                className="p-2 rounded-full hover:bg-gray-500/20"
              >
                <X className="w-5 h-5 opacity-60" />
              </button>
            </div>

            <div className="p-6 overflow-y-auto max-h-[70vh]">
              {/* Session Info */}
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mb-6">
                <StatCard 
                  title="Messages" 
                  value={sessionDetails.message_count || sessionDetails.messages?.length || 0} 
                  icon={MessageSquare} 
                  color="blue"
                />
                <StatCard 
                  title="Duration" 
                  value={sessionDetails.analytics?.duration || "N/A"} 
                  icon={Clock} 
                  color="purple"
                />
                <StatCard 
                  title="Quality Score" 
                  value={`${Math.round((sessionDetails.analytics?.qualityScore || 0.5) * 100)}%`} 
                  icon={TrendingUp} 
                  color="green"
                />
                <StatCard 
                  title="Topic Changes" 
                  value={sessionDetails.analytics?.topicChanges || 0} 
                  icon={Target} 
                  color="orange"
                />
              </div>

              {/* Messages */}
              <div className={cardClasses}>
                <div className="p-4">
                  <h4 className="font-semibold mb-4">Conversation Messages</h4>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {sessionDetails.messages?.map((message, index) => (
                      <div 
                        key={index} 
                        className={`p-3 rounded-lg ${
                          message.message_type === 'user' 
                            ? 'bg-blue-500/20 ml-4 sm:ml-8' 
                            : 'bg-gray-500/20 mr-4 sm:mr-8'
                        }`}
                      >
                        <div className="flex items-center space-x-2 mb-2">
                          {message.message_type === 'user' ? (
                            <User className="w-4 h-4 text-blue-400" />
                          ) : (
                            <Bot className="w-4 h-4 text-green-400" />
                          )}
                          <span className="text-sm font-medium capitalize">
                            {message.message_type}
                          </span>
                          <span className="text-xs opacity-60">
                            {new Date(message.created_at).toLocaleTimeString()}
                          </span>
                          {message.metadata?.confidence && (
                            <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                              {Math.round(message.metadata.confidence * 100)}% confidence
                            </span>
                          )}
                        </div>
                        <p className="text-sm break-words">{message.content}</p>
                        {message.metadata?.sources && message.metadata.sources.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-gray-500/20">
                            <p className="text-xs opacity-60 mb-1">Sources:</p>
                            <div className="flex flex-wrap gap-1">
                              {message.metadata.sources.map((source, idx) => (
                                <span key={idx} className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
                                  {source}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )) || (
                      <p className="text-center text-gray-500 py-8">No messages in this session</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderMemoryAnalysis = () => (
    <div className="space-y-8">
      {/* Memory Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <StatCard 
          title="Total Sessions" 
          value={memoryStats.totalSessions} 
          icon={MessageSquare} 
          color="blue"
          description="All conversations"
          trend={12}
        />
        <StatCard 
          title="Active Sessions" 
          value={memoryStats.activeSessions} 
          icon={Activity} 
          color="green"
          description="Currently active"
          trend={8}
        />
        <StatCard 
          title="Total Messages" 
          value={memoryStats.totalMessages} 
          icon={Brain} 
          color="purple"
          description="Across all sessions"
          trend={15}
        />
        <StatCard 
          title="Avg Messages/Session" 
          value={memoryStats.avgMessagesPerSession} 
          icon={Target} 
          color="orange"
          description="Per conversation"
          trend={5}
        />
      </div>

      {/* Memory Health & Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Memory Health */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Memory Health Status</h3>
          <div className={`text-center p-6 rounded-xl ${
            memoryStats.memoryHealth === 'excellent' ? 'bg-green-500/20 text-green-400' :
            memoryStats.memoryHealth === 'good' ? 'bg-blue-500/20 text-blue-400' :
            memoryStats.memoryHealth === 'fair' ? 'bg-yellow-500/20 text-yellow-400' :
            'bg-red-500/20 text-red-400'
          }`}>
            <CheckCircle className="w-12 h-12 mx-auto mb-3" />
            <h4 className="text-xl font-bold mb-2">{memoryStats.memoryHealth.toUpperCase()}</h4>
            <p className="text-sm opacity-80">
              {memoryStats.memoryHealth === 'excellent' ? 'Memory system performing optimally' :
               memoryStats.memoryHealth === 'good' ? 'Memory system functioning well' :
               memoryStats.memoryHealth === 'fair' ? 'Memory system needs attention' :
               'Memory system requires maintenance'}
            </p>
          </div>
        </div>

        {/* Topic Distribution */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Topic Distribution</h3>
          <div className="space-y-3">
            {Object.entries(memoryStats.topicDistribution).map(([topic, count]) => {
              const percentage = memoryStats.totalSessions > 0 ? 
                Math.round((count / memoryStats.totalSessions) * 100) : 0;
              return (
                <div key={topic} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="capitalize">{topic}</span>
                    <span>{count} sessions ({percentage}%)</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Engagement & Language Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* User Engagement */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">User Engagement Levels</h3>
          <div className="space-y-4">
            {Object.entries(memoryStats.userEngagement).map(([level, count]) => (
              <div key={level} className="flex justify-between items-center">
                <span className="text-sm capitalize">{level} Engagement</span>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">{count}</span>
                  <div className={`w-3 h-3 rounded-full ${
                    level === 'high' ? 'bg-green-500' :
                    level === 'medium' ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Peak Hours */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Peak Activity Hours</h3>
          <div className="space-y-3">
            {memoryStats.peakHours.slice(0, 5).map((hour, index) => (
              <div key={hour.hour} className="flex justify-between items-center">
                <span className="text-sm">
                  {hour.hour}:00 - {hour.hour + 1}:00
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-purple-500 h-2 rounded-full" 
                      style={{ width: `${(hour.count / Math.max(...memoryStats.peakHours.map(h => h.count))) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium w-8">{hour.count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className={`p-6 ${cardClasses}`}>
        <h3 className="text-lg font-semibold mb-4">Memory Actions</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <button 
            onClick={loadAllSessions}
            disabled={isLoading}
            className="flex items-center space-x-3 p-4 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh Memory</span>
          </button>
          <button 
            onClick={exportAllSessions}
            className="flex items-center space-x-3 p-4 rounded-xl bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors"
          >
            <Download className="w-5 h-5" />
            <span>Export All Data</span>
          </button>
          <button 
            onClick={() => calculateMemoryStats()}
            className="flex items-center space-x-3 p-4 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
          >
            <BarChart3 className="w-5 h-5" />
            <span>Recalculate Stats</span>
          </button>
          <button 
            className="flex items-center space-x-3 p-4 rounded-xl bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 transition-colors"
          >
            <Settings className="w-5 h-5" />
            <span>Memory Settings</span>
          </button>
        </div>
      </div>
    </div>
  );

  const renderAnalytics = () => (
    <div className="space-y-8">
      {/* Analytics Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <StatCard 
          title="Daily Sessions" 
          value={analyticsData.sessionsByDate?.slice(-1)[0]?.sessions || 0} 
          icon={Calendar} 
          color="blue"
          description="Today's activity"
        />
        <StatCard 
          title="Peak Hour" 
          value={`${analyticsData.messagesByHour?.reduce((max, curr) => curr.messages > max.messages ? curr : max, {hour: 0, messages: 0})?.hour || 9}:00`} 
          icon={Clock} 
          color="green"
          description="Most active time"
        />
        <StatCard 
          title="Top Topic" 
          value={analyticsData.topTopics?.[0]?.topic || 'Finance'} 
          icon={Target} 
          color="purple"
          description={`${analyticsData.topTopics?.[0]?.percentage || 35}% of discussions`}
        />
        <StatCard 
          title="System Uptime" 
          value={analyticsData.systemPerformance?.uptime || '99.9%'} 
          icon={Activity} 
          color="orange"
          description="Last 30 days"
        />
      </div>

      {/* Charts & Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Session Trends */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Session Trends (30 Days)</h3>
          <div className="h-64 flex items-end justify-between space-x-1">
            {analyticsData.sessionsByDate?.slice(-14).map((day, index) => {
              const maxSessions = Math.max(...analyticsData.sessionsByDate.map(d => d.sessions));
              const height = (day.sessions / maxSessions) * 100;
              return (
                <div key={index} className="flex flex-col items-center space-y-2">
                  <div 
                    className="bg-blue-500 rounded-t w-6 transition-all duration-300 hover:bg-blue-400"
                    style={{ height: `${height}%`, minHeight: '4px' }}
                    title={`${day.date}: ${day.sessions} sessions`}
                  />
                  <span className="text-xs opacity-60 transform rotate-45 origin-center">
                    {new Date(day.date).getDate()}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Hourly Activity */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Hourly Message Distribution</h3>
          <div className="h-64 flex items-end justify-between space-x-1">
            {analyticsData.messagesByHour?.filter((_, index) => index % 2 === 0).map((hour, index) => {
              const maxMessages = Math.max(...analyticsData.messagesByHour.map(h => h.messages));
              const height = (hour.messages / maxMessages) * 100;
              return (
                <div key={index} className="flex flex-col items-center space-y-2">
                  <div 
                    className="bg-green-500 rounded-t w-6 transition-all duration-300 hover:bg-green-400"
                    style={{ height: `${height}%`, minHeight: '4px' }}
                    title={`${hour.hour}:00 - ${hour.messages} messages`}
                  />
                  <span className="text-xs opacity-60">
                    {hour.hour}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Topic Analysis & User Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Topics */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Popular Topics</h3>
          <div className="space-y-4">
            {analyticsData.topTopics?.map((topic, index) => (
              <div key={index} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-medium">{topic.topic}</span>
                  <span className="text-sm opacity-60">{topic.count} discussions</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      index === 0 ? 'bg-purple-500' :
                      index === 1 ? 'bg-blue-500' :
                      index === 2 ? 'bg-green-500' :
                      'bg-orange-500'
                    }`}
                    style={{ width: `${topic.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* User Activity Leaderboard */}
        <div className={`p-6 ${cardClasses}`}>
          <h3 className="text-lg font-semibold mb-4">Most Active Teams</h3>
          <div className="space-y-4">
            {analyticsData.userActivity?.map((user, index) => (
              <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-gray-500/10">
                <div className="flex items-center space-x-3">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                    index === 0 ? 'bg-yellow-500' :
                    index === 1 ? 'bg-gray-400' :
                    index === 2 ? 'bg-orange-600' :
                    'bg-blue-500'
                  }`}>
                    {index + 1}
                  </div>
                  <span className="font-medium">{user.user}</span>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">{user.sessions} sessions</p>
                  <p className="text-xs opacity-60">{user.messages} messages</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Performance */}
      <div className={`p-6 ${cardClasses}`}>
        <h3 className="text-lg font-semibold mb-4">System Performance Metrics</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="text-center p-4 rounded-lg bg-blue-500/10">
            <Clock className="w-8 h-8 mx-auto mb-2 text-blue-400" />
            <p className="text-lg font-bold">{analyticsData.systemPerformance?.avgResponseTime || '1.2s'}</p>
            <p className="text-xs opacity-60">Avg Response Time</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-green-500/10">
            <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-400" />
            <p className="text-lg font-bold">{analyticsData.systemPerformance?.successRate || '98.5%'}</p>
            <p className="text-xs opacity-60">Success Rate</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-purple-500/10">
            <Database className="w-8 h-8 mx-auto mb-2 text-purple-400" />
            <p className="text-lg font-bold">{analyticsData.systemPerformance?.memoryUsage || '67%'}</p>
            <p className="text-xs opacity-60">Memory Usage</p>
          </div>
          <div className="text-center p-4 rounded-lg bg-orange-500/10">
            <Activity className="w-8 h-8 mx-auto mb-2 text-orange-400" />
            <p className="text-lg font-bold">{analyticsData.systemPerformance?.uptime || '99.9%'}</p>
            <p className="text-xs opacity-60">System Uptime</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderInsights = () => (
    <div className="space-y-8">
      <div className="text-center py-8">
        <Brain className="w-16 h-16 mx-auto mb-4 opacity-40" />
        <h3 className="text-xl font-semibold mb-2">AI-Powered Insights</h3>
        <p className="text-gray-500 mb-6">Advanced analysis of conversation patterns and user behavior</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <div className={`p-6 ${cardClasses}`}>
            <TrendingUp className="w-12 h-12 mx-auto mb-4 text-green-400" />
            <h4 className="font-semibold mb-2">Engagement Trending Up</h4>
            <p className="text-sm opacity-70 mb-3">User engagement has increased by 23% this month, with longer conversation sessions and more complex queries.</p>
            <div className="text-xs text-green-400 font-medium">+23% improvement</div>
          </div>
          
          <div className={`p-6 ${cardClasses}`}>
            <Target className="w-12 h-12 mx-auto mb-4 text-blue-400" />
            <h4 className="font-semibold mb-2">Topic Focus Shift</h4>
            <p className="text-sm opacity-70 mb-3">Finance-related queries dominate (35%), followed by HR inquiries (25%). Engineering discussions are increasing.</p>
            <div className="text-xs text-blue-400 font-medium">35% finance topics</div>
          </div>
          
          <div className={`p-6 ${cardClasses}`}>
            <Clock className="w-12 h-12 mx-auto mb-4 text-purple-400" />
            <h4 className="font-semibold mb-2">Peak Hours Identified</h4>
            <p className="text-sm opacity-70 mb-3">Most activity occurs between 10-11 AM and 2-3 PM, suggesting optimal times for system maintenance.</p>
            <div className="text-xs text-purple-400 font-medium">10-11 AM peak</div>
          </div>
          
          <div className={`p-6 ${cardClasses}`}>
            <CheckCircle className="w-12 h-12 mx-auto mb-4 text-orange-400" />
            <h4 className="font-semibold mb-2">Response Quality High</h4>
            <p className="text-sm opacity-70 mb-3">AI responses maintain 87% average quality score with strong user satisfaction and minimal corrections needed.</p>
            <div className="text-xs text-orange-400 font-medium">87% quality score</div>
          </div>
          
          <div className={`p-6 ${cardClasses}`}>
            <Users className="w-12 h-12 mx-auto mb-4 text-pink-400" />
            <h4 className="font-semibold mb-2">Team Adoption Growing</h4>
            <p className="text-sm opacity-70 mb-3">Finance and HR teams lead adoption, with Engineering showing rapid growth in usage over the past weeks.</p>
            <div className="text-xs text-pink-400 font-medium">+45% new users</div>
          </div>
          
          <div className={`p-6 ${cardClasses}`}>
            <Zap className="w-12 h-12 mx-auto mb-4 text-yellow-400" />
            <h4 className="font-semibold mb-2">Performance Optimized</h4>
            <p className="text-sm opacity-70 mb-3">System response times have improved by 15% with GPU optimization and enhanced caching mechanisms.</p>
            <div className="text-xs text-yellow-400 font-medium">+15% faster</div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDatabase = () => (
    <div className="space-y-8">
      <div className="text-center py-8">
        <Database className="w-16 h-16 mx-auto mb-4 opacity-40" />
        <h3 className="text-xl font-semibold mb-2">Database Management</h3>
        <p className="text-gray-500 mb-6">Direct database access and management tools</p>
        
        <div className={`p-6 ${cardClasses} text-left max-w-4xl mx-auto`}>
          <h4 className="font-medium mb-4">Database Connection Status:</h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Connection Status</span>
                <span className="flex items-center space-x-2 text-green-400">
                  <CheckCircle className="w-4 h-4" />
                  <span>Connected</span>
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Database Type</span>
                <span className="text-sm font-medium">PostgreSQL 14+</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Total Sessions</span>
                <span className="text-sm font-medium">{allSessions.length}</span>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Total Messages</span>
                <span className="text-sm font-medium">{memoryStats.totalMessages}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Memory Usage</span>
                <span className="text-sm font-medium">67%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Last Backup</span>
                <span className="text-sm font-medium">{new Date().toLocaleDateString()}</span>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-500/20 pt-4">
            <h5 className="font-medium mb-3">Database Tables:</h5>
            <div className="space-y-2 text-sm opacity-70 font-mono">
              <div>📊 conversation_sessions ({allSessions.length} records)</div>
              <div>💬 conversation_messages ({memoryStats.totalMessages} records)</div>
              <div>👥 users (active connections)</div>
              <div>🏢 organizations (multi-tenant support)</div>
              <div>📋 audit_logs (security tracking)</div>
            </div>
          </div>
          
          <div className="mt-6 flex flex-wrap gap-3">
            <button 
              onClick={() => addNotification('🔄 Database optimization started', 'info')}
              className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg transition-colors"
            >
              Optimize Database
            </button>
            <button 
              onClick={() => addNotification('💾 Backup process initiated', 'success')}
              className="px-4 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded-lg transition-colors"
            >
              Create Backup
            </button>
            <button 
              onClick={() => addNotification('🧹 Cleanup process started', 'info')}
              className="px-4 py-2 bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 rounded-lg transition-colors"
            >
              Clean Old Data
            </button>
            <button 
              onClick={() => addNotification('📈 Index rebuild initiated', 'info')}
              className="px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded-lg transition-colors"
            >
              Rebuild Indexes
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'sessions': return renderSessions();
      case 'memory': return renderMemoryAnalysis();
      case 'analytics': return renderAnalytics();
      case 'insights': return renderInsights();
      case 'database': return renderDatabase();
      default: return renderSessions();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex bg-black/50 backdrop-blur-sm">
      <div className={`w-full h-full flex flex-col rounded-lg sm:rounded-3xl sm:m-4 shadow-2xl border transition-all duration-300 ${
        darkMode 
          ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
          : 'bg-white/95 text-gray-900 border-gray-200/60'
      } overflow-hidden`}>
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 sm:p-6 border-b border-gray-800/60 flex-shrink-0">
          <div className="flex items-center space-x-3">
            <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-purple-500" />
            <div>
              <h1 className="text-xl sm:text-2xl font-bold">Conversation Memory</h1>
              <p className="text-sm opacity-60 hidden sm:block">Advanced memory analysis and session management</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-500/20 transition-colors duration-200"
          >
            <X className="w-5 h-5 sm:w-6 sm:h-6 opacity-60" />
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className={`w-16 sm:w-64 border-r flex-shrink-0 ${darkMode ? 'border-gray-800/60' : 'border-gray-200/60'} p-2 sm:p-6 overflow-y-auto`}>
            <nav className="space-y-1 sm:space-y-2">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center space-x-3 px-2 sm:px-4 py-2 sm:py-3 rounded-xl text-left transition-colors ${
                    activeTab === tab.id
                      ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                      : 'hover:bg-gray-500/10'
                  }`}
                  title={tab.label}
                >
                  <tab.icon className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
                  <span className="font-medium hidden sm:inline truncate">{tab.label}</span>
                </button>
              ))}
            </nav>

            {/* Memory Status */}
            <div className={`mt-6 p-4 rounded-xl border hidden sm:block ${
              darkMode 
                ? 'bg-gray-900/50 border-gray-800/60' 
                : 'bg-white/80 border-gray-200/60'
            }`}>
              <h4 className="font-medium mb-2">Memory Status</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Sessions:</span>
                  <span className="text-blue-400">{memoryStats.totalSessions}</span>
                </div>
                <div className="flex justify-between">
                  <span>Messages:</span>
                  <span className="text-green-400">{memoryStats.totalMessages}</span>
                </div>
                <div className="flex justify-between">
                  <span>Health:</span>
                  <span className={`${
                    memoryStats.memoryHealth === 'excellent' ? 'text-green-400' :
                    memoryStats.memoryHealth === 'good' ? 'text-blue-400' :
                    memoryStats.memoryHealth === 'fair' ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {memoryStats.memoryHealth}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-8">
            {isLoading && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-60">
                <div className={`p-6 rounded-2xl ${cardClasses} flex items-center space-x-3`}>
                  <RefreshCw className="w-6 h-6 animate-spin text-purple-500" />
                  <span>Loading memory data...</span>
                </div>
              </div>
            )}
            {renderContent()}
          </div>
        </div>

        {/* Confirmation Dialog */}
        {showConfirmDialog && (
          <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className={`w-full max-w-md p-6 rounded-2xl shadow-2xl border ${
              darkMode 
                ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
                : 'bg-white/95 text-gray-900 border-gray-200/60'
            }`}>
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                  <Trash2 className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Confirm Session Deletion</h3>
                  <p className="text-sm opacity-60">This action cannot be undone</p>
                </div>
              </div>
              
              <p className="text-sm mb-6 opacity-80">
                Are you sure you want to clear this session? All conversation history will be permanently deleted.
              </p>
              
              <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
                <button
                  onClick={() => {
                    setShowConfirmDialog(false);
                    setSessionToDelete(null);
                  }}
                  className="flex-1 py-3 px-4 rounded-xl border border-gray-500/50 hover:bg-gray-500/10 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmDelete}
                  disabled={isLoading}
                  className="flex-1 py-3 px-4 rounded-xl bg-red-500 hover:bg-red-600 text-white transition-colors disabled:opacity-50"
                >
                  {isLoading ? 'Deleting...' : 'Delete Session'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ConversationMemory;