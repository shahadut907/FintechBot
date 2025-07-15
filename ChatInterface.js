import React, { useState, useEffect, useRef, useContext } from 'react';
import { 
  Send, User, Bot, Upload, Image, Mic, MicOff, 
  ThumbsUp, ThumbsDown, Copy, FileText, X, 
  Loader, Brain, MessageSquare, Clock, Database,
  Target, AlertTriangle, CheckCircle, TrendingUp,
  Settings, Search, Filter, Download, Archive,
  ChevronLeft, ChevronRight, MoreVertical, 
  Plus, Trash2, Edit3, BookOpen, Palette,
  Type, Volume2, VolumeX, Maximize2, Minimize2,
  History, Star, Tag, SortAsc, SortDesc, FileDown,
  Sun, Moon, LogOut,
  SlidersHorizontal, Info, HelpCircle, ChevronDown, RefreshCw
} from 'lucide-react';
import './ChatInterface.css';

const EnhancedChatInterface = ({ user, darkMode, apiCall, addNotification, authToken, onLogout, onToggleTheme }) => {
  // Chat State
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  
  // File Upload State
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  
  // Voice State
  const [isRecording, setIsRecording] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  
  // UI State
  const [showQuickPrompts, setShowQuickPrompts] = useState(true);
  const [feedbackShown, setFeedbackShown] = useState(new Set());
  
  // New Enhanced Features State
  const [showSettings, setShowSettings] = useState(false);
  const [showConversationList, setShowConversationList] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredMessages, setFilteredMessages] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [starredMessages, setStarredMessages] = useState(new Set());
  const [fontSize, setFontSize] = useState('medium');
  const [chatTheme, setChatTheme] = useState('default');
  const [autoSave, setAutoSave] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [compactMode, setCompactMode] = useState(false);
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [exportFormat, setExportFormat] = useState('json');
  
  // Typewriter Effect State
  const [typewriterMessages, setTypewriterMessages] = useState(new Map());
  const [isTyping, setIsTyping] = useState(false);
  
  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const speechRecognitionRef = useRef(null);
  const typewriterTimeouts = useRef(new Map());

  // Department configurations
  const departmentInfo = {
    'employee': { 
      name: 'General Employee', 
      icon: User, 
      gradient: 'from-slate-500 via-blue-500 to-cyan-500',
      color: 'blue'
    },
    'engineering': { 
      name: 'Engineering', 
      icon: Brain, 
      gradient: 'from-emerald-500 via-green-500 to-teal-500',
      color: 'green'
    },
    'finance': { 
      name: 'Finance', 
      icon: TrendingUp, 
      gradient: 'from-violet-500 via-purple-500 to-indigo-500',
      color: 'purple'
    },
    'marketing': { 
      name: 'Marketing', 
      icon: Target, 
      gradient: 'from-orange-500 via-red-500 to-pink-500',
      color: 'orange'
    },
    'hr': { 
      name: 'Human Resources', 
      icon: User, 
      gradient: 'from-pink-500 via-rose-500 to-red-500',
      color: 'pink'
    },
    'c_level': { 
      name: 'C-Level Executive', 
      icon: CheckCircle, 
      gradient: 'from-amber-500 via-yellow-500 to-orange-500',
      color: 'amber'
    }
  };

  // Quick prompts based on role
  const quickPrompts = {
    'finance': [
      { text: 'Q4 Financial Report', prompt: '📊 Show me the latest quarterly financial report with key metrics' },
      { text: 'Budget Analysis', prompt: '💰 Analyze our current budget allocation and spending trends' },
      { text: 'Revenue Trends', prompt: '📈 What are our revenue trends for the past 6 months?' }
    ],
    'marketing': [
      { text: 'Campaign Performance', prompt: '🎯 Show me our latest marketing campaign performance metrics' },
      { text: 'Customer Insights', prompt: '👥 What are the key customer insights from recent campaigns?' },
      { text: 'Brand Sentiment', prompt: '❤️ How is our brand sentiment tracking across channels?' }
    ],
    'hr': [
      { text: 'Employee Metrics', prompt: '👥 Show me current employee satisfaction and retention metrics' },
      { text: 'Attendance Report', prompt: '📅 Generate attendance report for this month' },
      { text: 'Performance Reviews', prompt: '🏆 Summary of recent performance review cycles' }
    ],
    'engineering': [
      { text: 'System Architecture', prompt: '🏗️ Explain our current system architecture and tech stack' },
      { text: 'Security Audit', prompt: '🔐 Latest security audit results and recommendations' },
      { text: 'DevOps Metrics', prompt: '⚡ Display DevOps pipeline metrics and deployment stats' }
    ],
    'c_level': [
      { text: 'Executive Dashboard', prompt: '📊 Show me the complete executive dashboard with all KPIs' },
      { text: 'Strategic Goals', prompt: '🎯 Progress on strategic goals and initiatives' },
      { text: 'Growth Metrics', prompt: '📈 Key growth metrics and market position analysis' }
    ],
    'employee': [
      { text: 'Company Events', prompt: '📅 What are the upcoming company events and activities?' },
      { text: 'HR Policies', prompt: '❓ Explain the company HR policies and procedures' },
      { text: 'General Info', prompt: '☕ General company information and FAQs' }
    ]
  };

  // Font size configurations
  const fontSizes = {
    small: { text: 'text-sm', input: 'text-sm', header: 'text-base' },
    medium: { text: 'text-base', input: 'text-base', header: 'text-lg' },
    large: { text: 'text-lg', input: 'text-lg', header: 'text-xl' },
    extra_large: { text: 'text-xl', input: 'text-xl', header: 'text-2xl' }
  };

  // Theme configurations
  const themes = {
    default: { primary: 'blue', secondary: 'gray' },
    warm: { primary: 'orange', secondary: 'amber' },
    cool: { primary: 'indigo', secondary: 'slate' },
    forest: { primary: 'green', secondary: 'emerald' },
    sunset: { primary: 'pink', secondary: 'rose' }
  };

  // ===== TYPEWRITER EFFECT =====
  const typewriterEffect = (messageId, text, callback) => {
    setIsTyping(true);
    let index = 0;
    const speed = 30; // Adjust typing speed (ms per character)
    
    const typeChar = () => {
      if (index < text.length) {
        setTypewriterMessages(prev => {
          const newMap = new Map(prev);
          newMap.set(messageId, text.substring(0, index + 1));
          return newMap;
        });
        index++;
        
        const timeout = setTimeout(typeChar, speed);
        typewriterTimeouts.current.set(messageId, timeout);
      } else {
        setIsTyping(false);
        if (callback) callback();
        
        // Clean up after typing is complete
        setTimeout(() => {
          setTypewriterMessages(prev => {
            const newMap = new Map(prev);
            newMap.delete(messageId);
            return newMap;
          });
        }, 1000);
      }
    };
    
    typeChar();
  };

  const stopTypewriter = (messageId) => {
    const timeout = typewriterTimeouts.current.get(messageId);
    if (timeout) {
      clearTimeout(timeout);
      typewriterTimeouts.current.delete(messageId);
    }
    setIsTyping(false);
  };

  // Initialize speech recognition
  useEffect(() => {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      speechRecognitionRef.current = new SpeechRecognition();
      speechRecognitionRef.current.continuous = false;
      speechRecognitionRef.current.interimResults = true;
      speechRecognitionRef.current.lang = 'en-US';

      speechRecognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');
        setInputValue(transcript);
      };

      speechRecognitionRef.current.onerror = () => {
        setIsRecognizing(false);
        addNotification('❌ Speech recognition error', 'error');
      };

      speechRecognitionRef.current.onend = () => {
        setIsRecognizing(false);
      };
    }
  }, [addNotification]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Search messages
  useEffect(() => {
    if (searchTerm) {
      const filtered = messages.filter(message => 
        message.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
        message.sources?.some(source => source.toLowerCase().includes(searchTerm.toLowerCase()))
      );
      setFilteredMessages(filtered);
    } else {
      setFilteredMessages([]);
    }
  }, [searchTerm, messages]);

  // Load conversation history
  useEffect(() => {
    loadConversationHistory();
  }, [sessionId]);

  // Initialize with welcome message
  useEffect(() => {
    if (user && messages.length === 0) {
      const dept = departmentInfo[user.role] || departmentInfo['employee'];
      const welcomeMessage = {
        id: Date.now(),
        text: `✨ Welcome to FinSolve AI, ${user.name.split(' ')[0]}! 

I'm your intelligent assistant for ${dept.name.toLowerCase()} operations. I can help you with:

• Data analysis and reporting
• Document processing and insights  
• Department-specific queries
• File uploads (PDF, CSV, Excel)

How can I assist you today?`,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isWelcome: true
      };
      setMessages([welcomeMessage]);
    }
  }, [user, messages.length]);

  // Utility Functions
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      addNotification('✅ Copied to clipboard!', 'success');
    } catch (error) {
      addNotification('❌ Copy failed', 'error');
    }
  };

  const playNotificationSound = () => {
    if (soundEnabled) {
      // Create a simple notification sound
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      oscillator.frequency.value = 800;
      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.5);
    }
  };

  const loadConversationHistory = async () => {
    try {
      const response = await apiCall('/conversation/sessions');
      if (response && response.sessions) {
        setConversationHistory(response.sessions);
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error);
    }
  };

  const exportConversation = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      user: user.name,
      session_id: sessionId,
      messages: messages.map(msg => ({
        sender: msg.sender,
        text: msg.text,
        timestamp: msg.timestamp,
        sources: msg.sources || []
      }))
    };

    let dataStr, mimeType, fileName;
    
    switch (exportFormat) {
      case 'json':
        dataStr = JSON.stringify(exportData, null, 2);
        mimeType = 'application/json';
        fileName = `chat_export_${new Date().toISOString().split('T')[0]}.json`;
        break;
      case 'csv':
        const csvHeaders = 'Sender,Message,Timestamp,Sources\n';
        const csvRows = messages.map(msg => 
          `"${msg.sender}","${msg.text.replace(/"/g, '""')}","${msg.timestamp}","${(msg.sources || []).join('; ')}"`
        ).join('\n');
        dataStr = csvHeaders + csvRows;
        mimeType = 'text/csv';
        fileName = `chat_export_${new Date().toISOString().split('T')[0]}.csv`;
        break;
      case 'txt':
        dataStr = messages.map(msg => 
          `[${new Date(msg.timestamp).toLocaleString()}] ${msg.sender.toUpperCase()}: ${msg.text}`
        ).join('\n\n');
        mimeType = 'text/plain';
        fileName = `chat_export_${new Date().toISOString().split('T')[0]}.txt`;
        break;
      case 'pdf':
        exportToPDF();
        return;
      default:
        return;
    }

    const dataBlob = new Blob([dataStr], { type: mimeType });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
    addNotification(`📄 Conversation exported as ${exportFormat.toUpperCase()}`, 'success');
  };

  // ===== PDF EXPORT FUNCTIONALITY =====
  const exportToPDF = () => {
    const printWindow = window.open('', '_blank');
    const dept = departmentInfo[user.role] || departmentInfo['employee'];
    
    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>FinSolve AI Chat History - ${user.name}</title>
          <style>
            body { 
              font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
              margin: 20px; 
              color: #333;
              line-height: 1.6;
            }
            .header { 
              text-align: center; 
              margin-bottom: 30px; 
              padding-bottom: 20px;
              border-bottom: 2px solid #007bff;
            }
            .header h1 { 
              color: #007bff; 
              margin: 0;
              font-size: 28px;
            }
            .header p { 
              margin: 10px 0; 
              color: #666;
              font-size: 14px;
            }
            .message { 
              margin: 20px 0; 
              padding: 15px; 
              border-radius: 10px; 
              break-inside: avoid;
            }
            .user-message { 
              background-color: #e3f2fd; 
              margin-left: 20%;
              border-left: 4px solid #007bff;
            }
            .bot-message { 
              background-color: #f5f5f5; 
              margin-right: 20%;
              border-left: 4px solid #28a745;
            }
            .sender { 
              font-weight: bold; 
              margin-bottom: 8px;
              text-transform: uppercase;
              font-size: 12px;
              letter-spacing: 1px;
            }
            .timestamp { 
              font-size: 11px; 
              color: #888; 
              margin-top: 8px;
            }
            .sources { 
              margin-top: 10px; 
              padding-top: 10px; 
              border-top: 1px solid #ddd;
              font-size: 12px;
            }
            .sources-title { 
              font-weight: bold; 
              margin-bottom: 5px;
              color: #007bff;
            }
            .footer { 
              margin-top: 30px; 
              padding-top: 20px; 
              border-top: 1px solid #ddd; 
              text-align: center; 
              color: #888;
              font-size: 12px;
            }
            @media print {
              body { margin: 0; }
              .message { break-inside: avoid; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>FinSolve AI Chat History</h1>
            <p><strong>User:</strong> ${user.name} (${dept.name})</p>
            <p><strong>Session:</strong> ${sessionId || 'New Session'}</p>
            <p><strong>Exported:</strong> ${new Date().toLocaleString()}</p>
            <p><strong>Total Messages:</strong> ${messages.length}</p>
          </div>
          
          ${messages.map(msg => `
            <div class="message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}">
              <div class="sender">${msg.sender === 'user' ? '👤 User' : '🤖 FinSolve AI'}</div>
              <div class="content">${msg.text.replace(/\n/g, '<br>')}</div>
              <div class="timestamp">${new Date(msg.timestamp).toLocaleString()}</div>
              ${msg.sources && msg.sources.length > 0 ? `
                <div class="sources">
                  <div class="sources-title">📚 Sources:</div>
                  ${msg.sources.map(source => `<div>• ${source}</div>`).join('')}
                </div>
              ` : ''}
            </div>
          `).join('')}
          
          <div class="footer">
            <p>Generated by FinSolve AI - Intelligent Business Assistant</p>
            <p>This document contains ${messages.length} messages from your conversation session.</p>
          </div>
        </body>
      </html>
    `;

    printWindow.document.write(htmlContent);
    printWindow.document.close();
    
    printWindow.onload = () => {
      printWindow.print();
      addNotification('📄 PDF export initiated - check your browser\'s print dialog', 'success');
    };
  };

  const starMessage = (messageId) => {
    setStarredMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
        addNotification('⭐ Message unstarred', 'info');
      } else {
        newSet.add(messageId);
        addNotification('⭐ Message starred', 'success');
      }
      return newSet;
    });
  };

  const calculateConfidenceScore = (message) => {
    if (message.access_denied) return { score: 0, level: 'denied' };
    
    let score = 50;
    if (message.sources?.length > 0) score += 30;
    if (message.documents_found > 3) score += 20;
    
    const level = score >= 80 ? 'high' : score >= 60 ? 'medium' : 'low';
    return { score, level };
  };

  // File handling
  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    const validFiles = files.filter(file => {
      const validTypes = ['application/pdf', 'text/csv', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/plain'];
      return validTypes.includes(file.type) && file.size < 10 * 1024 * 1024;
    });

    setUploadedFiles(prev => [...prev, ...validFiles.map(file => ({
      name: file.name,
      size: file.size,
      type: file.type,
      file: file
    }))]);

    if (validFiles.length !== files.length) {
      addNotification('⚠️ Some files rejected. Use PDF, CSV, Excel, or text files under 10MB.', 'warning');
    } else {
      addNotification(`📁 ${validFiles.length} file(s) uploaded successfully`, 'success');
    }
  };

  const handleImageUpload = (files) => {
    const imageFiles = Array.from(files).filter(file => 
      file.type.startsWith('image/') && file.size < 10 * 1024 * 1024
    );
    
    if (imageFiles.length > 0) {
      const newImages = imageFiles.map(file => ({
        id: Date.now() + Math.random(),
        file,
        name: file.name,
        size: file.size,
        preview: URL.createObjectURL(file),
        type: 'image'
      }));
      
      setUploadedImages(prev => [...prev, ...newImages]);
      addNotification(`📸 ${imageFiles.length} image(s) uploaded successfully`, 'success');
    }
  };

  // Drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    handleImageUpload(files);
  };

  // Voice recording
  const startVoiceRecording = () => {
    if (speechRecognitionRef.current && !isRecognizing) {
      setIsRecognizing(true);
      speechRecognitionRef.current.start();
      addNotification('🎤 Listening...', 'info');
    }
  };

  const stopVoiceRecording = () => {
    if (speechRecognitionRef.current && isRecognizing) {
      speechRecognitionRef.current.stop();
      setIsRecognizing(false);
    }
  };

  // Enhanced send message with better error handling
  const sendMessage = async (customMessage = null) => {
    const messageText = customMessage || inputValue;
    if (!messageText.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: messageText,
      sender: 'user',
      timestamp: new Date().toISOString(),
      files: uploadedFiles.length > 0 ? [...uploadedFiles] : null,
      images: uploadedImages.length > 0 ? [...uploadedImages] : null
    };

    setMessages(prev => [...prev, userMessage]);
    if (!customMessage) setInputValue('');
    setIsLoading(true);
    setUploadedFiles([]);
    setUploadedImages([]);

    try {
      const chatResponse = await apiCall('/chat', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: messageText,
          role: user.role,
          user_name: user.name,
          session_id: sessionId
        })
      });

      if (chatResponse.session_id) {
        setSessionId(chatResponse.session_id);
      }

      // Enhanced response handling
      let responseText = chatResponse.response;
      
      // Improve response quality - handle unknown queries
      if (!responseText || responseText.trim().length < 10) {
        responseText = "I'm not sure I understand that completely. Could you please provide more details or rephrase your question? I'm here to help with any information you need.";
      }
      
      // Check for potential hallucinations or uncertain responses
      if (responseText.includes("I don't have") || responseText.includes("I cannot find") || responseText.includes("not available")) {
        responseText += "\n\nIs there something more specific I can help you with, or would you like me to search for related information?";
      }

      const botMessage = {
        id: Date.now() + 1,
        text: responseText,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        sources: chatResponse.sources || [],
        success: chatResponse.success,
        documents_found: chatResponse.documents_found,
        processing_time: chatResponse.processing_time,
        access_denied: chatResponse.access_denied || false,
        isFollowUp: chatResponse.is_follow_up,
        conversationLength: chatResponse.conversation_length
      };

      setMessages(prev => [...prev, botMessage]);

      // Start typewriter effect for the bot response
      typewriterEffect(botMessage.id, responseText, () => {
        playNotificationSound();
      });

      if (chatResponse.access_denied) {
        addNotification('🚫 Access denied - insufficient permissions', 'warning');
      } else {
        const contextInfo = chatResponse.is_follow_up ? ' (follow-up)' : '';
        addNotification(`✅ Response generated successfully${contextInfo}`, 'success');
      }

      // Auto-save if enabled
      if (autoSave) {
        const conversationData = {
          sessionId,
          messages: [...messages, userMessage, botMessage],
          timestamp: new Date().toISOString()
        };
        localStorage.setItem(`finsolve_chat_${sessionId}`, JSON.stringify(conversationData));
      }

    } catch (error) {
      console.error('Chat error:', error);
      addNotification('❌ Failed to send message', 'error');
      
      const errorMessage = {
        id: Date.now() + 1,
        text: 'I apologize, but I encountered a technical issue while processing your request. Please try again in a moment, or contact support if the problem persists.',
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Get current department info
  const currentDept = departmentInfo[user?.role] || departmentInfo['employee'];
  const currentQuickPrompts = quickPrompts[user?.role] || quickPrompts['employee'];
  const currentFontSize = fontSizes[fontSize];
  const currentTheme = themes[chatTheme];

  // Function to determine confidence level display
  function getConfidenceDisplay(confidence) {
    if (!confidence) return { level: 'warning', text: 'Unknown', color: 'orange' };
    
    if (confidence >= 0.85) {
      return { level: 'success', text: 'High', color: 'green' };
    } else if (confidence >= 0.7) {
      return { level: 'normal', text: 'Medium', color: 'blue' };
    } else {
      return { level: 'warning', text: 'Low', color: 'orange' };
    }
  }

  // Settings Panel Component
  const SettingsPanel = () => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className={`w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl border ${
        darkMode 
          ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
          : 'bg-white/95 text-gray-900 border-gray-200/60'
      }`}>
        <div className="flex items-center justify-between p-6 border-b border-gray-800/60">
          <h3 className="text-xl font-semibold">Chat Settings</h3>
          <button 
            onClick={() => setShowSettings(false)}
            className="p-2 rounded-full hover:bg-gray-500/20"
          >
            <X className="w-5 h-5 opacity-60" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Appearance */}
          <div>
            <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Palette className="w-5 h-5" />
              <span>Appearance</span>
            </h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Font Size</label>
                <select
                  value={fontSize}
                  onChange={(e) => setFontSize(e.target.value)}
                  className={`w-full p-3 rounded-xl border ${
                    darkMode 
                      ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                      : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                  }`}
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                  <option value="extra_large">Extra Large</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Theme</label>
                <select
                  value={chatTheme}
                  onChange={(e) => setChatTheme(e.target.value)}
                  className={`w-full p-3 rounded-xl border ${
                    darkMode 
                      ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                      : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                  }`}
                >
                  <option value="default">Default Blue</option>
                  <option value="warm">Warm Orange</option>
                  <option value="cool">Cool Indigo</option>
                  <option value="forest">Forest Green</option>
                  <option value="sunset">Sunset Pink</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Compact Mode</span>
                <button
                  onClick={() => setCompactMode(!compactMode)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    compactMode ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                    compactMode ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Show Timestamps</span>
                <button
                  onClick={() => setShowTimestamps(!showTimestamps)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    showTimestamps ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                    showTimestamps ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          </div>

          {/* Behavior */}
          <div>
            <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Settings className="w-5 h-5" />
              <span>Behavior</span>
            </h4>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Auto-save Conversations</span>
                <button
                  onClick={() => setAutoSave(!autoSave)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    autoSave ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                    autoSave ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Sound Notifications</span>
                <button
                  onClick={() => setSoundEnabled(!soundEnabled)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    soundEnabled ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
                    soundEnabled ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>
          </div>

          {/* Export */}
          <div>
            <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2">
              <Download className="w-5 h-5" />
              <span>Export Options</span>
            </h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Export Format</label>
                <select
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value)}
                  className={`w-full p-3 rounded-xl border ${
                    darkMode 
                      ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                      : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                  }`}
                >
                  <option value="pdf">PDF (Formatted)</option>
                  <option value="json">JSON</option>
                  <option value="csv">CSV</option>
                  <option value="txt">Text</option>
                </select>
              </div>

              <button
                onClick={exportConversation}
                className="w-full py-3 px-4 bg-blue-500 hover:bg-blue-600 text-white rounded-xl transition-colors flex items-center justify-center space-x-2"
              >
                <FileDown className="w-5 h-5" />
                <span>Export Chat History</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Conversation List Panel Component
  const ConversationListPanel = () => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className={`w-full max-w-4xl max-h-[90vh] overflow-hidden rounded-2xl shadow-2xl border ${
        darkMode 
          ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
          : 'bg-white/95 text-gray-900 border-gray-200/60'
      }`}>
        <div className="flex items-center justify-between p-6 border-b border-gray-800/60">
          <h3 className="text-xl font-semibold">Conversation History</h3>
          <button 
            onClick={() => setShowConversationList(false)}
            className="p-2 rounded-full hover:bg-gray-500/20"
          >
            <X className="w-5 h-5 opacity-60" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[70vh]">
          {conversationHistory.length === 0 ? (
            <div className="text-center py-8">
              <History className="w-16 h-16 mx-auto mb-4 opacity-40" />
              <p className="text-gray-500">No conversation history found</p>
            </div>
          ) : (
            <div className="space-y-4">
              {conversationHistory.map((conv, index) => (
                <div 
                  key={index}
                  className={`p-4 rounded-xl border transition-all duration-300 hover:scale-[1.02] cursor-pointer ${
                    darkMode 
                      ? 'bg-gray-900/50 border-gray-800/60 hover:bg-gray-900/70' 
                      : 'bg-white/80 border-gray-200/60 hover:bg-white/90'
                  }`}
                  onClick={() => {
                    setSessionId(conv.session_id);
                    setShowConversationList(false);
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-semibold">{conv.current_topic || `Session ${conv.session_id.slice(-8)}`}</h4>
                      <p className="text-sm opacity-60">{new Date(conv.last_activity).toLocaleString()}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{conv.message_count} messages</p>
                      <div className="flex items-center space-x-2 mt-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // Load this conversation
                          }}
                          className="p-1 rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
                        >
                          <BookOpen className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            // Delete conversation
                          }}
                          className="p-1 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-screen max-h-screen overflow-hidden">
      {/* Enhanced Header */}
      <div className={`flex-shrink-0 p-4 border-b ${
        darkMode ? 'bg-gray-950/95 border-gray-800/60' : 'bg-white/90 border-gray-200/60'
      } backdrop-blur-sm`}>
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-10 h-10 rounded-2xl bg-gradient-to-r ${currentDept.gradient} flex items-center justify-center shadow-lg`}>
              <currentDept.icon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className={`font-semibold ${currentFontSize.header}`}>FinSolve AI</h1>
              <p className={`opacity-60 ${currentFontSize.text}`}>{currentDept.name} Assistant</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Search Button */}
            <button
              onClick={() => setShowSearch(!showSearch)}
              className={`p-2 rounded-lg transition-colors ${
                showSearch ? 'bg-blue-500/20 text-blue-400' : 'hover:bg-gray-500/20'
              }`}
              title="Search Messages"
            >
              <Search className="w-5 h-5" />
            </button>

            {/* Conversation History Button */}
            <button
              onClick={() => setShowConversationList(true)}
              className="p-2 rounded-lg hover:bg-gray-500/20 transition-colors"
              title="Conversation History"
            >
              <History className="w-5 h-5" />
            </button>

            {/* Export Button */}
            <button
              onClick={exportConversation}
              className="p-2 rounded-lg hover:bg-gray-500/20 transition-colors"
              title="Export Chat History"
            >
              <FileDown className="w-5 h-5" />
            </button>

            {/* Settings Button */}
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 rounded-lg hover:bg-gray-500/20 transition-colors"
              title="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>

            {/* Theme Toggle */}
            {onToggleTheme && (
              <button
                onClick={onToggleTheme}
                className="p-2 rounded-lg hover:bg-gray-500/20 transition-colors"
                title={darkMode ? 'Light Mode' : 'Dark Mode'}
              >
                {darkMode ? <Sun className="w-5 h-5 text-yellow-400" /> : <Moon className="w-5 h-5 text-gray-600" />}
              </button>
            )}

            {/* Logout */}
            {onLogout && (
              <button
                onClick={onLogout}
                className="p-2 rounded-lg hover:bg-red-500/20 text-red-400 transition-colors"
                title="Logout"
              >
                <LogOut className="w-5 h-5" />
              </button>
            )}

            {/* Sound Toggle */}
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-lg transition-colors ${
                soundEnabled ? 'text-blue-400' : 'text-gray-500'
              }`}
              title="Toggle Sound"
            >
              {soundEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
            </button>
            
            {sessionId && (
              <div className={`px-3 py-1 rounded-full text-xs ${
                darkMode ? 'bg-gray-800/60 text-gray-300' : 'bg-gray-100/60 text-gray-600'
              }`}>
                Session: {sessionId.slice(-6)}
              </div>
            )}
          </div>
        </div>

        {/* Search Bar */}
        {showSearch && (
          <div className="max-w-4xl mx-auto mt-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 opacity-40" />
              <input
                type="text"
                placeholder="Search messages..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className={`w-full pl-10 pr-4 py-2 rounded-xl border ${
                  darkMode 
                    ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                    : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                }`}
              />
              {searchTerm && (
                <button
                  onClick={() => setSearchTerm('')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2"
                >
                  <X className="w-4 h-4 opacity-60" />
                </button>
              )}
            </div>
            {filteredMessages.length > 0 && (
              <div className="mt-2 text-sm opacity-60">
                Found {filteredMessages.length} message(s)
              </div>
            )}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        <div className="max-w-4xl mx-auto">
          
          {/* Welcome Quick Prompts */}
          {messages.length === 1 && showQuickPrompts && (
            <div className="mb-8">
              <h3 className={`font-semibold mb-4 text-center ${currentFontSize.header}`}>Quick Start Suggestions</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {currentQuickPrompts.slice(0, 6).map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => sendMessage(prompt.prompt)}
                    className={`p-4 rounded-xl border transition-all duration-200 hover:scale-105 text-left ${
                      darkMode 
                        ? 'bg-gray-900/50 border-gray-800/60 hover:bg-gray-800/60 text-white' 
                        : 'bg-white/50 border-gray-200/40 hover:bg-gray-50 text-gray-900'
                    }`}
                  >
                    <div className={`font-medium mb-1 ${currentFontSize.text}`}>{prompt.text}</div>
                    <p className={`line-clamp-2 ${currentFontSize.text} ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      {prompt.prompt}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chat Messages */}
          {(searchTerm ? filteredMessages : messages).map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-6 ${
                compactMode ? 'mb-3' : ''
              }`}
            >
              <div className={`max-w-full sm:max-w-3xl w-full ${message.sender === 'user' ? 'flex justify-end' : ''}`}>
                <div className={`flex space-x-3 ${
                  message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}>
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-2xl flex items-center justify-center ${
                    message.sender === 'user'
                      ? `bg-gradient-to-r ${currentDept.gradient} shadow-lg`
                      : darkMode
                      ? 'bg-gray-800 border border-gray-700'
                      : 'bg-white border border-gray-200 shadow-md'
                  }`}>
                    {message.sender === 'user' ? (
                      <currentDept.icon className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                    ) : (
                      <Bot className="w-4 h-4 sm:w-5 sm:h-5 text-blue-500" />
                    )}
                  </div>

                  {/* Message Content */}
                  <div className={`flex-1 min-w-0 ${message.sender === 'user' ? 'text-right' : ''}`}>
                    {/* Message Bubble */}
                    <div className={`inline-block max-w-full p-3 sm:p-4 rounded-2xl ${currentFontSize.text} ${
                      message.sender === 'user'
                        ? `bg-gradient-to-r ${currentDept.gradient} text-white shadow-lg`
                        : darkMode
                        ? 'bg-gray-900/80 border border-gray-800/60 text-gray-100 shadow-lg'
                        : 'bg-white border border-gray-200/60 text-gray-900 shadow-md'
                    } ${message.sender !== 'user' ? 'w-full' : ''}`}>
                      
                      {/* Access Denied Warning */}
                      {message.access_denied && (
                        <div className="mb-3 p-2 sm:p-3 bg-red-500/20 border border-red-400/40 rounded-xl flex items-center space-x-2">
                          <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />
                          <span className="text-red-200 text-xs sm:text-sm font-medium">Access Denied</span>
                        </div>
                      )}

                      {/* Message Text with Typewriter Effect */}
                      <div className="whitespace-pre-wrap break-words">
                        {message.sender === 'bot' && typewriterMessages.has(message.id) 
                          ? typewriterMessages.get(message.id)
                          : message.text
                        }
                        {message.sender === 'bot' && isTyping && typewriterMessages.has(message.id) && (
                          <span className="animate-pulse">|</span>
                        )}
                      </div>

                      {/* File Attachments */}
                      {message.files && message.files.length > 0 && (
                        <div className="mt-3 space-y-2">
                          {message.files.map((file, fileIndex) => (
                            <a
                              key={fileIndex}
                              href={URL.createObjectURL(file.file)}
                              download={file.name}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center space-x-2 p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors cursor-pointer"
                              title={`Click to download ${file.name}`}
                            >
                              <FileText className="w-4 h-4 opacity-60 flex-shrink-0" />
                              <span className="text-xs sm:text-sm opacity-80 truncate">{file.name}</span>
                              <span className="text-xs opacity-60 flex-shrink-0">({(file.size / 1024).toFixed(1)} KB)</span>
                            </a>
                          ))}
                        </div>
                      )}

                      {/* Image Attachments */}
                      {message.images && message.images.length > 0 && (
                        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2">
                          {message.images.map((image, imageIndex) => (
                            <div key={imageIndex} className="relative">
                              <img 
                                src={image.preview} 
                                alt={image.name}
                                className="w-full h-24 sm:h-32 object-cover rounded-lg"
                              />
                              <div className="absolute bottom-1 left-1 bg-black/60 text-white text-xs px-2 py-1 rounded">
                                {image.name}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Bot Message Metadata */}
                      {message.sender === 'bot' && !message.isWelcome && (
                        <div className="mt-4 space-y-3">
                          {/* Sources */}
                          {message.sources && message.sources.length > 0 && (
                            <div className={`border-t pt-3 ${darkMode ? 'border-gray-500/20' : 'border-gray-300/20'}`}>
                              <div className="flex items-center space-x-2 mb-2">
                                <Database className="w-4 h-4 opacity-60" />
                                <span className="text-xs sm:text-sm font-medium opacity-80">Sources Used</span>
                              </div>
                              <div className="space-y-1">
                                {message.sources.map((source, sourceIndex) => (
                                  <div key={sourceIndex} className="text-xs opacity-70 flex items-center space-x-2">
                                    <div className="w-1 h-1 bg-blue-400 rounded-full flex-shrink-0"></div>
                                    <span className="truncate">{source}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Confidence Score */}
                          {!message.access_denied && (
                            <div className={`border-t pt-3 ${darkMode ? 'border-gray-500/20' : 'border-gray-300/20'}`}>
                              {(() => {
                                const confidence = calculateConfidenceScore(message);
                                const confidenceInfo = getConfidenceDisplay(confidence.score);
                                return (
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-2">
                                      <span className="text-xs sm:text-sm opacity-80">Confidence:</span>
                                      <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                                        confidenceInfo.level === 'high' ? 'bg-green-500/20 text-green-400' :
                                        confidenceInfo.level === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                        'bg-red-500/20 text-red-400'
                                      }`}>
                                        {confidenceInfo.text}
                                      </div>
                                    </div>
                                    <div className="text-xs opacity-60">
                                      {message.documents_found || 0} docs • {message.processing_time || '0.00s'}
                                    </div>
                                  </div>
                                );
                              })()}
                            </div>
                          )}

                          {/* Action Buttons */}
                          {!feedbackShown.has(message.id) && (
                            <div className={`border-t pt-3 flex items-center justify-between ${darkMode ? 'border-gray-500/20' : 'border-gray-300/20'}`}>
                              <div className="flex items-center space-x-2">
                                <button
                                  onClick={() => {
                                    setFeedbackShown(prev => new Set([...prev, message.id]));
                                    addNotification('👍 Positive feedback recorded', 'success');
                                  }}
                                  className="p-1 rounded-full hover:bg-green-500/20 transition-colors duration-200"
                                  title="This was helpful"
                                >
                                  <ThumbsUp className="w-4 h-4 opacity-60 hover:opacity-100" />
                                </button>
                                <button
                                  onClick={() => {
                                    setFeedbackShown(prev => new Set([...prev, message.id]));
                                    addNotification('👎 Negative feedback recorded', 'info');
                                  }}
                                  className="p-1 rounded-full hover:bg-red-500/20 transition-colors duration-200"
                                  title="This needs improvement"
                                >
                                  <ThumbsDown className="w-4 h-4 opacity-60 hover:opacity-100" />
                                </button>
                                <button
                                  onClick={() => starMessage(message.id)}
                                  className={`p-1 rounded-full transition-colors duration-200 ${
                                    starredMessages.has(message.id) 
                                      ? 'text-yellow-400 hover:bg-yellow-500/20' 
                                      : 'hover:bg-yellow-500/20'
                                  }`}
                                  title="Star message"
                                >
                                  <Star className={`w-4 h-4 ${starredMessages.has(message.id) ? 'fill-current' : ''} opacity-60 hover:opacity-100`} />
                                </button>
                              </div>
                              <button
                                onClick={() => copyToClipboard(message.text)}
                                className="p-1 rounded-full hover:bg-blue-500/20 transition-colors duration-200"
                                title="Copy message"
                              >
                                <Copy className="w-4 h-4 opacity-60 hover:opacity-100" />
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Timestamp */}
                    {showTimestamps && (
                      <div className={`text-xs mt-1 ${
                        message.sender === 'user' ? 'text-right' : 'text-left'
                      } ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-4xl w-full">
                <div className="flex space-x-3">
                  <div className={`flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-2xl flex items-center justify-center ${
                    darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200 shadow-md'
                  }`}>
                    <Bot className="w-4 h-4 sm:w-5 sm:h-5 text-blue-500" />
                  </div>
                  <div className={`flex-1 p-3 sm:p-4 rounded-2xl shadow-lg ${
                    darkMode ? 'bg-gray-900/80 border border-gray-800/60' : 'bg-white border border-gray-200/60'
                  }`}>
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className={`${currentFontSize.text} ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>AI is thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div 
        className={`flex-shrink-0 p-4 border-t transition-all duration-300 ${
          darkMode ? 'bg-gray-950/95 border-gray-800/60' : 'bg-white/90 border-gray-200/60'
        } ${isDragging ? 'border-blue-500 bg-blue-500/10' : ''} backdrop-blur-sm`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="max-w-4xl mx-auto">
          {/* File Upload Preview */}
          {(uploadedFiles.length > 0 || uploadedImages.length > 0) && (
            <div className="mb-4 space-y-3">
              {/* Files */}
              {uploadedFiles.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Uploaded Files</span>
                    <button
                      onClick={() => setUploadedFiles([])}
                      className="text-xs text-red-400 hover:text-red-300 transition-colors"
                    >
                      Clear All
                    </button>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                    {uploadedFiles.map((file, index) => (
                      <a
                        key={index}
                        href={URL.createObjectURL(file.file)}
                        download={file.name}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={`flex items-center space-x-2 p-2 rounded-lg border ${darkMode ? 'bg-gray-800/60 border-gray-700/50' : 'bg-gray-100/60 border-gray-200/40'} hover:bg-opacity-80 transition-all duration-200 cursor-pointer`}
                        title={`Click to view/download ${file.name}`}
                      >
                        <FileText className={`w-4 h-4 flex-shrink-0 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`} />
                        <div className="flex-1 min-w-0">
                          <div className={`text-sm truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>{file.name}</div>
                          <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{(file.size / 1024).toFixed(1)} KB</div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.preventDefault(); // Prevent the link from being followed
                            setUploadedFiles(prev => prev.filter((_, i) => i !== index));
                          }}
                          className="p-1 rounded-full hover:bg-red-500/20 transition-colors flex-shrink-0"
                        >
                          <X className="w-3 h-3 text-red-400" />
                        </button>
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Images */}
              {uploadedImages.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Uploaded Images</span>
                    <button
                      onClick={() => setUploadedImages([])}
                      className="text-xs text-red-400 hover:text-red-300 transition-colors"
                    >
                      Clear All
                    </button>
                  </div>
                  <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-2">
                    {uploadedImages.map((image, index) => (
                      <div key={index} className="relative group">
                        <img 
                          src={image.preview} 
                          alt={image.name}
                          className="w-full h-16 object-cover rounded-lg"
                        />
                        <button
                          onClick={() => setUploadedImages(prev => prev.filter((_, i) => i !== index))}
                          className="absolute -top-1 -right-1 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Main Input Area */}
          <div className="flex items-end space-x-2 sm:space-x-4">
            {/* File Upload Button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className={`p-2 sm:p-3 rounded-xl transition-all duration-200 hover:scale-105 flex-shrink-0 ${
                darkMode 
                  ? 'bg-gray-800/60 hover:bg-gray-700/60 border border-gray-700/50' 
                  : 'bg-gray-100/60 hover:bg-gray-200/60 border border-gray-200/40'
              }`}
              title="Upload files"
            >
              <Upload className={`w-4 h-4 sm:w-5 sm:h-5 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`} />
            </button>

            {/* Image Upload Button */}
            <button
              onClick={() => imageInputRef.current?.click()}
              className={`p-2 sm:p-3 rounded-xl transition-all duration-200 hover:scale-105 flex-shrink-0 ${
                darkMode 
                  ? 'bg-gray-800/60 hover:bg-gray-700/60 border border-gray-700/50' 
                  : 'bg-gray-100/60 hover:bg-gray-200/60 border border-gray-200/40'
              }`}
              title="Upload images"
            >
              <Image className={`w-4 h-4 sm:w-5 sm:h-5 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`} />
            </button>

            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`Ask me anything about ${currentDept.name.toLowerCase()} data...`}
                disabled={isLoading}
                className={`w-full p-3 sm:p-4 pr-10 sm:pr-12 rounded-xl border resize-none transition-all duration-200 ${currentFontSize.input} ${
                  darkMode 
                    ? 'bg-gray-900/80 border-gray-800/60 text-gray-100 placeholder-gray-400 focus:bg-gray-900 focus:border-gray-700' 
                    : 'bg-white border-gray-200/60 text-gray-900 placeholder-gray-500 focus:bg-white focus:border-gray-300'
                } focus:outline-none focus:ring-2 focus:ring-blue-500/50 ${
                  isLoading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                rows={inputValue.includes('\n') ? Math.min(inputValue.split('\n').length, 4) : 1}
                style={{ minHeight: '48px' }}
              />
              
              {/* Character Counter */}
              <div className={`absolute bottom-2 right-10 sm:right-12 text-xs ${
                inputValue.length > 1000 ? 'text-red-400' : darkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                {inputValue.length}/2000
              </div>
            </div>

            {/* Voice Input Button */}
            {speechRecognitionRef.current && (
              <button
                onClick={isRecognizing ? stopVoiceRecording : startVoiceRecording}
                className={`p-2 sm:p-3 rounded-xl transition-all duration-200 hover:scale-105 flex-shrink-0 ${
                  isRecognizing 
                    ? 'bg-red-500 text-white animate-pulse' 
                    : darkMode 
                      ? 'bg-gray-800/60 hover:bg-gray-700/60 border border-gray-700/50' 
                      : 'bg-gray-100/60 hover:bg-gray-200/60 border border-gray-200/40'
                }`}
                title={isRecognizing ? 'Stop recording' : 'Start voice input'}
              >
                {isRecognizing ? (
                  <MicOff className="w-4 h-4 sm:w-5 sm:h-5" />
                ) : (
                  <Mic className={`w-4 h-4 sm:w-5 sm:h-5 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`} />
                )}
              </button>
            )}

            {/* Send Button */}
            <button
              onClick={() => sendMessage()}
              disabled={!inputValue.trim() || isLoading}
              className={`p-2 sm:p-3 rounded-xl transition-all duration-200 hover:scale-105 flex-shrink-0 ${
                !inputValue.trim() || isLoading
                  ? 'bg-gray-500/30 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600 text-white shadow-lg'
              }`}
              title="Send message"
            >
              {isLoading ? (
                <Loader className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
              ) : (
                <Send className="w-4 h-4 sm:w-5 sm:h-5" />
              )}
            </button>
          </div>

          {/* Input Hints */}
          <div className={`mt-3 flex flex-col sm:flex-row sm:items-center sm:justify-between text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'} space-y-1 sm:space-y-0`}>
            <span>Press Enter to send, Shift+Enter for new line</span>
            <div className="flex items-center space-x-2">
              <span>FinSolve AI</span>
              {sessionId && (
                <span className="text-blue-400">
                  Session: {sessionId.slice(-6)}
                </span>
              )}
            </div>
          </div>

          {/* Hidden File Inputs */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.csv,.xlsx,.txt"
            onChange={handleFileUpload}
            className="hidden"
          />
          <input
            ref={imageInputRef}
            type="file"
            multiple
            accept="image/*"
            onChange={(e) => handleImageUpload(e.target.files)}
            className="hidden"
          />
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && <SettingsPanel />}

      {/* Conversation List Panel */}
      {showConversationList && <ConversationListPanel />}
    </div>
  );
};

export default EnhancedChatInterface;