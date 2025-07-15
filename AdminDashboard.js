import React, { useState, useEffect, useRef } from 'react';
import { 
  X, Users, Building, FileText, BarChart3, Shield, Search, Download, 
  Plus, Edit3, Trash2, Eye, RefreshCw, Database, Activity, TrendingUp, 
  AlertCircle, CheckCircle, XCircle, Clock, User, DollarSign, Calendar, 
  Filter, ChevronDown, ChevronUp, ExternalLink, Settings, Lock, Unlock, 
  UserCheck, ShieldCheck, AlertTriangle, Zap, Globe, Monitor, Brain,
  MessageSquare, Upload, Server, Wifi, WifiOff, Save, Folder, FileCheck,
  LogOut
} from 'lucide-react';
import DocumentViewerModal from './DocumentViewerModal';
import ConversationMemory from './ConversationMemory';

const AdminDashboard = ({ isOpen, onClose, darkMode, user, apiCall, addNotification, onLogout }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // FIXED: Real-time data state
  const [realTimeData, setRealTimeData] = useState({
    users: [],
    organizations: [],
    documents: [],
    analytics: {},
    auditLogs: [],
    sessions: [],
    systemMetrics: {},
    recentActivity: []
  });
  
  // Auto-refresh state
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const refreshIntervalRef = useRef(null);
  
  // Form states & refs
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [showEditUser, setShowEditUser] = useState(false);
  const [showUserDetails, setShowUserDetails] = useState(false);
  const [showDocumentViewer, setShowDocumentViewer] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [confirmAction, setConfirmAction] = useState(null);

  // UI States
  const [expandedSections, setExpandedSections] = useState({
    users: true,
    documents: true,
    analytics: true,
    logs: false
  });

  // Upload states
  const [showDepartmentSelector, setShowDepartmentSelector] = useState(false);
  const [filesToUpload, setFilesToUpload] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedDepartment, setSelectedDepartment] = useState('');
  const [selectedTargetRole, setSelectedTargetRole] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const fileUploadRef = useRef(null);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3, color: 'blue' },
    { id: 'users', label: 'Users', icon: Users, color: 'green' },
    { id: 'organizations', label: 'Organizations', icon: Building, color: 'purple' },
    { id: 'documents', label: 'Documents', icon: FileText, color: 'orange' },
    { id: 'analytics', label: 'Analytics', icon: TrendingUp, color: 'pink' },
    { id: 'memory', label: 'Memory', icon: Brain, color: 'purple' },
    { id: 'audit', label: 'Audit Logs', icon: Shield, color: 'red' },
    { id: 'system', label: 'System', icon: Settings, color: 'gray' }
  ];

  // FIXED: Department and role options
  const departmentOptions = [
    { value: 'finance', label: 'Finance Department', roles: ['finance', 'c_level'] },
    { value: 'hr', label: 'Human Resources', roles: ['hr', 'c_level'] },
    { value: 'marketing', label: 'Marketing Department', roles: ['marketing', 'c_level'] },
    { value: 'engineering', label: 'Engineering Department', roles: ['engineering', 'c_level'] },
    { value: 'general', label: 'General/Company-wide', roles: ['employee', 'c_level', 'finance', 'hr', 'marketing', 'engineering'] }
  ];

  const roleOptions = [
    { value: 'employee', label: 'General Employee' },
    { value: 'finance', label: 'Finance Team' },
    { value: 'hr', label: 'HR Team' },
    { value: 'marketing', label: 'Marketing Team' },
    { value: 'engineering', label: 'Engineering Team' },
    { value: 'c_level', label: 'C-Level Executives' }
  ];

  // FIXED: Load real-time admin data
  const loadAdminData = async (showNotification = false) => {
    try {
      if (showNotification) {
        setIsLoading(true);
      }

      // Fetch concurrently with error handling
      const [usersRes, orgsRes, docsRes, analyticsRes, auditRes] = await Promise.all([
        apiCall('/admin/users').catch(e => { console.error('Users fetch failed:', e); return []; }),
        apiCall('/admin/organizations').catch(e => { console.error('Orgs fetch failed:', e); return []; }),
        apiCall('/admin/documents').catch(e => { console.error('Docs fetch failed:', e); return []; }),
        apiCall('/admin/analytics').catch(e => { console.error('Analytics fetch failed:', e); return {}; }),
        apiCall('/admin/audit-logs').catch(e => { console.error('Audit fetch failed:', e); return { logs: [] }; })
      ]);

      const newData = {
        users: Array.isArray(usersRes) ? usersRes : [],
        organizations: Array.isArray(orgsRes) ? orgsRes : [],
        documents: Array.isArray(docsRes) ? docsRes : [],
        analytics: analyticsRes || {},
        auditLogs: auditRes?.logs || [],
        recentActivity: analyticsRes?.recent_activity || []
      };

      const currentJson = JSON.stringify(realTimeData);
      const newJson = JSON.stringify(newData);

      if (currentJson !== newJson) {
        setRealTimeData(prev => ({
          ...prev,
          ...newData,
          lastUpdated: new Date().toISOString(),
          dataFreshness: 'live'
        }));

        setLastUpdated(new Date());

        if (showNotification) {
          addNotification('📊 Real-time admin data refreshed', 'success');
        }
      }
    } catch (error) {
      console.error('Failed to load admin data:', error);
      if (showNotification) {
        addNotification('❌ Failed to load some admin data', 'error');
      }
    } finally {
      if (showNotification) {
        setIsLoading(false);
      }
    }
  };

  // FIXED: Set up enhanced real-time data loading
  useEffect(() => {
    // Helper to start the polling interval
    const startPolling = () => {
      if (refreshIntervalRef.current) clearInterval(refreshIntervalRef.current);
      refreshIntervalRef.current = setInterval(() => {
        loadAdminData();
      }, 3000); // 3 seconds
    };

    // Helper to stop the polling interval
    const stopPolling = () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
        refreshIntervalRef.current = null;
      }
    };

    // Determine if we should pause real-time updates (when any blocking modal is open)
    const isBlockingModalOpen = showCreateUser || showEditUser || showDepartmentSelector || showUserDetails;

    if (isOpen) {
      loadAdminData();

      if (autoRefresh && !isBlockingModalOpen) {
        startPolling();
      } else {
        stopPolling();
      }
    }

    return () => {
      stopPolling();
    };
  }, [isOpen, autoRefresh, showCreateUser, showEditUser, showDepartmentSelector, showUserDetails]);

  // Manual refresh handler
  const handleManualRefresh = async () => {
    await loadAdminData(true);
  };

  // Auto-refresh toggle
  const toggleAutoRefresh = () => {
    const newState = !autoRefresh;
    setAutoRefresh(newState);

    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
    }

    if (newState) {
      refreshIntervalRef.current = setInterval(() => {
        loadAdminData();
      }, 3000);
      addNotification('🔄 Real-time admin updates enabled (3s refresh)', 'success');
    } else {
      addNotification('⏸️ Real-time admin updates paused', 'info');
    }
  };

  // ===== USER CRUD OPERATIONS =====
  const handleEditUser = async () => {
    if (!selectedUser || !selectedUser.email || !selectedUser.name) {
      addNotification('❌ Please fill in all required fields', 'error');
      return;
    }

    try {
      setIsLoading(true);
      const response = await apiCall(`/admin/users/${selectedUser.id}`, {
        method: 'PUT',
        body: JSON.stringify({
          name: selectedUser.name,
          role: selectedUser.role,
          department: selectedUser.department,
          is_active: selectedUser.is_active
        })
      });

      if (response.success) {
        addNotification('✅ User updated successfully', 'success');
        setShowEditUser(false);
        setSelectedUser(null);
        await loadAdminData();
      }
    } catch (error) {
      console.error('Failed to update user:', error);
      addNotification('❌ Failed to update user', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      setIsLoading(true);
      const response = await apiCall(`/admin/users/${userId}`, {
        method: 'DELETE'
      });

      if (response.success) {
        addNotification('✅ User deleted successfully', 'success');
        await loadAdminData();
      }
    } catch (error) {
      console.error('Failed to delete user:', error);
      addNotification('❌ Failed to delete user', 'error');
    } finally {
      setIsLoading(false);
      setShowConfirmDialog(false);
      setConfirmAction(null);
    }
  };

  const toggleUserStatus = async (userId, currentStatus) => {
    try {
      setIsLoading(true);
      const response = await apiCall(`/admin/users/${userId}/toggle-status`, {
        method: 'PATCH',
        body: JSON.stringify({ is_active: !currentStatus })
      });

      if (response.success) {
        addNotification(`✅ User ${!currentStatus ? 'activated' : 'deactivated'} successfully`, 'success');
        await loadAdminData();
      }
    } catch (error) {
      console.error('Failed to toggle user status:', error);
      addNotification('❌ Failed to update user status', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // ===== DOCUMENT OPERATIONS =====
  const handleViewDocument = async (document) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/admin/documents/${document.id}/content`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('finsolve_admin_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to load document: ${response.status}`);
      }

      const result = await response.json();
      setSelectedDocument({ ...document, content: result.content });
      setShowDocumentViewer(true);
    } catch (error) {
      console.error('Failed to load document:', error);
      addNotification('❌ Failed to load document content', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadDocument = async (document) => {
    try {
      const response = await apiCall(`/admin/documents/${document.id}/download`, {
        method: 'GET'
      });
      
      // If backend returns base64 data
      let blob;
      if (response.data) {
        const binary = atob(response.data);
        const len = binary.length;
        const buffer = new Uint8Array(len);
        for (let i = 0; i < len; i++) buffer[i] = binary.charCodeAt(i);
        blob = new Blob([buffer], { type: document.file_type || 'application/octet-stream' });
      } else {
        // Fallback for plain text response
        blob = new Blob([JSON.stringify(response, null, 2)], { type: 'application/json' });
      }
      
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = document.filename;
      link.click();
      URL.revokeObjectURL(url);
      
      addNotification('📄 Document downloaded successfully', 'success');
    } catch (error) {
      console.error('Failed to download document:', error);
      addNotification('❌ Failed to download document', 'error');
    }
  };

  const handleDeleteDocument = async (documentId) => {
    try {
      setIsLoading(true);
      const response = await apiCall(`/admin/documents/${documentId}`, {
        method: 'DELETE'
      });

      if (response.success) {
        addNotification('✅ Document deleted successfully', 'success');
        await loadAdminData();
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
      addNotification('❌ Failed to delete document', 'error');
    } finally {
      setIsLoading(false);
      setShowConfirmDialog(false);
      setConfirmAction(null);
    }
  };

  // ===== DOCUMENT UPLOAD OPERATIONS =====
  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    console.log('Files selected:', files);

    if (!files.length) {
      console.log('No files selected');
      return;
    }

    const validFiles = files.filter(file => {
      const validTypes = [
        'application/pdf', 
        'text/csv', 
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/plain',
        'text/markdown'
      ];
      const validExtensions = ['.pdf', '.csv', '.xlsx', '.xls', '.txt', '.md'];
      const hasValidType = validTypes.includes(file.type);
      const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      const isValidSize = file.size < 50 * 1024 * 1024; // 50MB
      return (hasValidType || hasValidExtension) && isValidSize;
    });

    if (validFiles.length === 0) {
      addNotification('❌ No valid files selected. Please use PDF, CSV, Excel, or text files under 50MB.', 'error');
      return;
    }

    if (validFiles.length !== files.length) {
      addNotification(`⚠️ ${files.length - validFiles.length} files rejected. Using only valid files.`, 'warning');
    }

    console.log('Valid files:', validFiles);
    setFilesToUpload(validFiles);
    setShowDepartmentSelector(true);
    addNotification(`📁 ${validFiles.length} file(s) selected for upload`, 'info');
  };

  const uploadToDepartment = async () => {
    if (!filesToUpload.length) {
      addNotification('❌ No files to upload', 'error');
      return;
    }
  
    if (!selectedDepartment) {
      addNotification('❌ Please select a department', 'error');
      return;
    }
  
    console.log('Starting upload...', {
      files: filesToUpload.length,
      department: selectedDepartment,
      targetRole: selectedTargetRole
    });
  
    setIsUploading(true);
    setUploadProgress(0);
  
    try {
      // Create FormData
      const formData = new FormData();
      
      // Add files
      filesToUpload.forEach((file, index) => {
        console.log(`Adding file ${index}:`, file.name, file.type, file.size);
        formData.append('files', file);
      });
      
      // Add form fields
      formData.append('department', selectedDepartment);
      if (selectedTargetRole) {
        formData.append('target_role', selectedTargetRole);
      }
  
      console.log('FormData created, sending request...');
  
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 300);
  
      // Get admin token
      const adminToken = localStorage.getItem('finsolve_admin_token');
      if (!adminToken) {
        clearInterval(progressInterval);
        throw new Error('No admin token found. Please login as admin again.');
      }
  
      console.log('Admin token found, sending request...');
  
      // FIXED: Enhanced fetch with better error handling
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/admin/upload-department-data`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${adminToken}`,
          // Don't set Content-Type - let browser set it for FormData
        },
        body: formData
      });
  
      clearInterval(progressInterval);
      setUploadProgress(100);
  
      console.log('Response received:', response.status, response.statusText);
      console.log('Response headers:', [...response.headers.entries()]);
  
      // FIXED: Better response handling
      if (!response) {
        throw new Error('No response received from server');
      }
  
      if (!response.ok) {
        console.error('Response not OK:', response.status, response.statusText);
        
        let errorMessage;
        try {
          const errorText = await response.text();
          console.error('Error response body:', errorText);
          
          // Try to parse as JSON first
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorText;
          } catch {
            errorMessage = errorText;
          }
        } catch (e) {
          console.error('Could not read error response:', e);
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(`Upload failed: ${errorMessage}`);
      }
  
      // FIXED: Safe JSON parsing
      let result;
      let responseText = ''; // Declare responseText in a higher scope
      try {
        responseText = await response.text();
        console.log('Response text:', responseText);
        console.log('Response text type:', typeof responseText);
        console.log('Response text length:', responseText.length);
        
        if (!responseText) {
          throw new Error('Empty response from server');
        }
        
        result = JSON.parse(responseText);
        console.log('Parsed result:', result);
      } catch (parseError) {
        console.error('JSON parse error:', parseError);
        throw new Error(`Invalid response format from server: ${responseText}`);
      }
  
      // FIXED: Check result exists and has expected structure
      if (!result) {
        throw new Error('No data returned from server');
      }
  
      if (result.success) {
        addNotification(`✅ Upload successful: ${result.message}`, 'success');
        
        if (result.total_chunks > 0) {
          addNotification(`🔍 Created ${result.total_chunks} searchable chunks`, 'info');
          addNotification(`📊 Files immediately available for ${selectedDepartment} department`, 'info');
        }
  
        // Show detailed results
        if (result.results && result.results.length > 0) {
          const successful = result.results.filter(r => r.status && r.status.includes('success'));
          const failed = result.results.filter(r => r.status && !r.status.includes('success'));
          
          if (successful.length > 0) {
            addNotification(`📁 Successfully processed: ${successful.map(r => r.filename).join(', ')}`, 'success');
          }
          
          if (failed.length > 0) {
            console.warn('Failed files:', failed);
            failed.forEach(f => {
              addNotification(`⚠️ Failed: ${f.filename} - ${f.status}`, 'warning');
            });
          }
        }
  
        // Refresh the admin data
        await loadAdminData();
        
        // Close the modal
        setShowDepartmentSelector(false);
        setFilesToUpload([]);
        setSelectedDepartment('');
        setSelectedTargetRole('');
        
      } else {
        // Handle unsuccessful upload
        const errorMessage = result.message || result.error || 'Upload failed for unknown reason';
        throw new Error(errorMessage);
      }
  
    } catch (error) {
      console.error('Upload error details:', error);
      console.error('Error stack:', error.stack);
      
      // Provide more specific error messages
      let userMessage;
      if (error.message.includes('Failed to fetch')) {
        userMessage = '❌ Connection failed. Check if the backend server is running.';
      } else if (error.message.includes('401') || error.message.includes('Unauthorized')) {
        userMessage = '❌ Authentication failed. Please login as admin again.';
      } else if (error.message.includes('403') || error.message.includes('Forbidden')) {
        userMessage = '❌ Access denied. Admin privileges required.';
      } else if (error.message.includes('413') || error.message.includes('too large')) {
        userMessage = '❌ File too large. Please use files under 50MB.';
      } else {
        userMessage = `❌ Upload failed: ${error.message}`;
      }
      
      addNotification(userMessage, 'error');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
      setFilesToUpload([]); // Clear files on both success and failure
      
      // Clear the file input
      if (fileUploadRef.current) {
        fileUploadRef.current.value = '';
      }
    }
  };
  
  // ALSO ADD: Enhanced test function for debugging
  const testUploadEndpoint = async () => {
    try {
      console.log('=== UPLOAD DEBUG TEST ===');
      
      // Check admin token
      const adminToken = localStorage.getItem('finsolve_admin_token');
      console.log('1. Admin token check:', adminToken ? 'EXISTS' : 'MISSING');
      
      if (!adminToken) {
        addNotification('❌ No admin token found. Please login as admin.', 'error');
        return;
      }
      
      // Test API connectivity
      console.log('2. Testing API connectivity...');
      const healthResponse = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/health`);
      console.log('Health check:', healthResponse.status, healthResponse.statusText);
      
      // Test admin authentication
      console.log('3. Testing admin authentication...');
      const authResponse = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/admin/validate-session`, {
        headers: {
          'Authorization': `Bearer ${adminToken}`,
        }
      });
      console.log('Auth check:', authResponse.status, authResponse.statusText);
      
      if (!authResponse.ok) {
        const authError = await authResponse.text();
        console.error('Auth error:', authError);
        addNotification('❌ Admin authentication failed. Please login again.', 'error');
        return;
      }
      
      // Create test file
      console.log('4. Creating test file...');
      const testContent = `# Test Upload File
      
  This is a test file created at ${new Date().toISOString()}.
  
  ## Content
  - Department: General
  - Purpose: Testing upload functionality
  - File size: Small text file
  
  The upload system should process this file and create searchable chunks.`;
      
      const testFile = new Blob([testContent], { type: 'text/markdown' });
      const file = new File([testFile], 'test-upload.md', { type: 'text/markdown' });
      
      console.log('Test file created:', file.name, file.size, file.type);
      
      // Create FormData
      const formData = new FormData();
      formData.append('files', file);
      formData.append('department', 'general');
      
      console.log('5. Sending test upload request...');
      
      const uploadResponse = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/admin/test-upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${adminToken}`,
        },
        body: formData
      });
      
      console.log('Upload response:', uploadResponse.status, uploadResponse.statusText);
      console.log('Upload response headers:', [...uploadResponse.headers.entries()]);
      
      const responseText = await uploadResponse.text();
      console.log('Upload response text:', responseText);
      
      if (uploadResponse.ok) {
        const result = JSON.parse(responseText);
        console.log('Upload result:', result);
        addNotification('✅ Test upload successful!', 'success');
      } else {
        console.error('Upload failed:', responseText);
        addNotification(`❌ Test upload failed: ${responseText}`, 'error');
      }
      
    } catch (error) {
      console.error('Test upload error:', error);
      addNotification(`❌ Test error: ${error.message}`, 'error');
    }
  };

  // ===== SYSTEM OPERATIONS =====
  const reloadDocuments = async () => {
    try {
      setIsLoading(true);
      const response = await apiCall('/admin/reload-documents', {
        method: 'POST'
      });
      addNotification('🔄 Documents reloaded successfully', 'success');
      await loadAdminData(); // Refresh data
    } catch (error) {
      console.error('Failed to reload documents:', error);
      addNotification('❌ Failed to reload documents', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSystemBackup = async () => {
    try {
      setIsLoading(true);
      const response = await apiCall('/admin/system/backup', {
        method: 'POST'
      });
      
      if (response.success) {
        addNotification('💾 System backup created successfully', 'success');
        // Trigger download of backup file
        const blob = new Blob([JSON.stringify(response.backup_data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `system_backup_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to create backup:', error);
      addNotification('❌ Failed to create system backup', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSystemHealthCheck = async () => {
    try {
      setIsLoading(true);
      const response = await apiCall('/admin/system/health-check');
      
      if (response.status === 'healthy') {
        addNotification('✅ System health check passed', 'success');
      } else {
        addNotification('⚠️ System health issues detected', 'warning');
      }
      
      // Show detailed health info
      console.log('System Health:', response);
    } catch (error) {
      console.error('Failed to perform health check:', error);
      addNotification('❌ Health check failed', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // ===== CONFIRMATION DIALOG =====
  const showConfirmationDialog = (action, item) => {
    setConfirmAction(() => action);
    setShowConfirmDialog(true);
  };

  // NEW: helper to run the stored confirmAction and close the dialog
  const handleConfirmAction = async () => {
    if (confirmAction) {
      try {
        await confirmAction();
      } catch (e) {
        console.error('Confirmation action failed:', e);
      }
    }
    setShowConfirmDialog(false);
    setConfirmAction(null);
  };

  // Responsive utility classes
  const cardClasses = `rounded-2xl border transition-all duration-300 ${
    darkMode 
      ? 'bg-gray-900/50 border-gray-800/60' 
      : 'bg-white/80 border-gray-200/60'
  } shadow-lg backdrop-blur-sm`;

  const buttonClasses = (color = 'blue', variant = 'primary') => {
    const colors = {
      blue: variant === 'primary' ? 'bg-blue-500 hover:bg-blue-600 text-white' : 'bg-blue-500/20 hover:bg-blue-500/30 text-blue-400',
      green: variant === 'primary' ? 'bg-green-500 hover:bg-green-600 text-white' : 'bg-green-500/20 hover:bg-green-500/30 text-green-400',
      red: variant === 'primary' ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-red-500/20 hover:bg-red-500/30 text-red-400',
      purple: variant === 'primary' ? 'bg-purple-500 hover:bg-purple-600 text-white' : 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400',
      orange: variant === 'primary' ? 'bg-orange-500 hover:bg-orange-600 text-white' : 'bg-orange-500/20 hover:bg-orange-500/30 text-orange-400',
      gray: variant === 'primary' ? 'bg-gray-500 hover:bg-gray-600 text-white' : 'bg-gray-500/20 hover:bg-gray-500/30 text-gray-400'
    };
    return `px-3 py-2 rounded-lg transition-colors ${colors[color]}`;
  };

  // FIXED: StatCard with real-time data
  const StatCard = ({ title, value, icon: Icon, color = 'blue', change = null, description = null, className = '', trend = null, isLive = false }) => (
    <div className={`p-4 sm:p-6 ${cardClasses} hover:scale-105 ${className} ${isLive ? 'ring-1 ring-green-500/30' : ''}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-sm opacity-70 mb-2 truncate">{title}</p>
          <p className="text-2xl sm:text-3xl font-bold truncate">{value}</p>
          {change && (
            <p className={`text-sm mt-2 flex items-center space-x-1 ${
              change > 0 ? 'text-green-500' : 'text-red-500'
            }`}>
              <TrendingUp className={`w-4 h-4 ${change < 0 ? 'rotate-180' : ''}`} />
              <span>{Math.abs(change)}%</span>
            </p>
          )}
          {description && (
            <p className="text-xs opacity-60 mt-1">{description}</p>
          )}
          {trend && (
            <p className="text-xs mt-1 text-blue-400">Real-time</p>
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

  const DataTable = ({ title, data, columns, actions = null }) => (
    <div className={cardClasses}>
      <div className="p-4 sm:p-6 border-b border-gray-800/60">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-2 sm:space-y-0">
          <h3 className="text-lg font-semibold">{title}</h3>
          {actions && <div className="flex space-x-2">{actions}</div>}
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className={`${darkMode ? 'bg-gray-800/50' : 'bg-gray-100/50'}`}>
            <tr>
              {columns.map((column, index) => (
                <th key={index} className="px-4 sm:px-6 py-3 sm:py-4 text-left text-sm font-medium opacity-80 whitespace-nowrap">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr 
                key={index} 
                className={`border-t ${darkMode ? 'border-gray-800/60' : 'border-gray-200/60'} hover:bg-gray-500/10 transition-colors`}
              >
                {columns.map((column, colIndex) => (
                  <td key={colIndex} className="px-4 sm:px-6 py-3 sm:py-4 text-sm">
                    <div className={column.render ? '' : 'truncate max-w-xs'}>
                      {column.render ? column.render(row) : row[column.key]}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
            {data.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-6 py-8 text-center text-gray-500">
                  No data available
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );

  // FIXED: Overview with real-time data
  const renderOverview = () => (
    <div className="space-y-6">
      {/* FIXED: Real-time Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <StatCard 
          title="Total Users" 
          value={realTimeData.users?.length || 0} 
          icon={Users} 
          color="blue"
          description="Registered users"
          trend="real-time"
          isLive={true}
        />
        <StatCard 
          title="Active Sessions" 
          value={realTimeData.analytics?.active_sessions || 0} 
          icon={Activity} 
          color="green"
          description="Current active sessions"
          trend="real-time"
          isLive={true}
        />
        <StatCard 
          title="Documents" 
          value={realTimeData.documents?.length || 0} 
          icon={FileText} 
          color="purple"
          description="Total documents processed"
          trend="real-time"
          isLive={true}
        />
        <StatCard 
          title="Messages (30min)" 
          value={realTimeData.analytics?.messages_last_30min || 0} 
          icon={MessageSquare} 
          color="orange"
          description="Recent chat activity"
          trend="real-time"
          isLive={true}
        />
      </div>

      {/* FIXED: Real-time System Status */}
      <div className={`p-4 sm:p-6 ${cardClasses}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">System Status</h3>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400">Live</span>
            <span className="text-xs opacity-60">Updated: {lastUpdated.toLocaleTimeString()}</span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="flex items-center space-x-3 p-3 rounded-xl bg-green-500/20">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <div>
              <p className="text-sm font-medium">System Uptime</p>
              <p className="text-xs opacity-60">{realTimeData.analytics?.system_uptime || 'Unknown'}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3 p-3 rounded-xl bg-blue-500/20">
            <Activity className="w-5 h-5 text-blue-400" />
            <div>
              <p className="text-sm font-medium">Success Rate</p>
              <p className="text-xs opacity-60">{realTimeData.analytics?.success_rate || 'N/A'}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3 p-3 rounded-xl bg-purple-500/20">
            <Clock className="w-5 h-5 text-purple-400" />
            <div>
              <p className="text-sm font-medium">Avg Response</p>
              <p className="text-xs opacity-60">{realTimeData.analytics?.avg_response_time || 'N/A'}</p>
            </div>
          </div>
          <div className="flex items-center space-x-3 p-3 rounded-xl bg-orange-500/20">
            <Database className="w-5 h-5 text-orange-400" />
            <div>
              <p className="text-sm font-medium">Total Requests</p>
              <p className="text-xs opacity-60">{realTimeData.analytics?.total_requests || 0}</p>
            </div>
          </div>
        </div>
      </div>

      {/* FIXED: Auto-refresh controls */}
      <div className={`p-4 sm:p-6 ${cardClasses}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Real-time Admin Controls</h3>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
              <span className="text-sm">{autoRefresh ? 'Live Data' : 'Static'}</span>
            </div>
            <span className="text-xs opacity-60">Updated: {lastUpdated.toLocaleTimeString()}</span>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input type="checkbox" checked={autoRefresh} onChange={toggleAutoRefresh} className="rounded border-gray-300" />
              <span className="text-sm">Auto-refresh (3s)</span>
            </label>
            <button onClick={handleManualRefresh} disabled={isLoading} className={`${buttonClasses('blue', 'primary')} ${isLoading ? 'opacity-50' : ''}`}>
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh Now
            </button>
          </div>
        </div>

        {/* Real-time Data Quality */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="text-center p-3 rounded-lg bg-blue-500/10">
            <Users className="w-6 h-6 mx-auto mb-2 text-blue-400" />
            <p className="text-sm font-bold">{realTimeData.users?.length || 0}</p>
            <p className="text-xs opacity-60">Users</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-green-500/10">
            <Activity className="w-6 h-6 mx-auto mb-2 text-green-400" />
            <p className="text-sm font-bold">{realTimeData.analytics?.active_sessions || 0}</p>
            <p className="text-xs opacity-60">Active Sessions</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-purple-500/10">
            <FileText className="w-6 h-6 mx-auto mb-2 text-purple-400" />
            <p className="text-sm font-bold">{realTimeData.documents?.length || 0}</p>
            <p className="text-xs opacity-60">Documents</p>
          </div>
          <div className="text-center p-3 rounded-lg bg-orange-500/10">
            <Database className="w-6 h-6 mx-auto mb-2 text-orange-400" />
            <p className="text-sm font-bold">{autoRefresh ? 'LIVE' : 'STATIC'}</p>
            <p className="text-xs opacity-60">Data Mode</p>
          </div>
        </div>

        {/* Connection Quality */}
        <div className="mt-4 p-3 bg-gray-500/10 rounded-lg">
          <div className="flex items-center justify-between text-sm"><span>Admin Data Refresh:</span><span className="text-blue-400">3 seconds</span></div>
          <div className="flex items-center justify-between text-sm"><span>Connection Quality:</span><span className="text-green-400">Excellent</span></div>
          <div className="flex items-center justify-between text-sm"><span>Last Sync:</span><span className="text-purple-400">{new Date(realTimeData.lastUpdated || Date.now()).toLocaleTimeString()}</span></div>
        </div>
      </div>

      {/* FIXED: Recent Activity with real-time data */}
      <div className={`p-4 sm:p-6 ${cardClasses}`}>
        <h3 className="text-lg font-semibold mb-4">Recent Activity</h3>
        <div className="space-y-4">
          {realTimeData.recentActivity?.slice(0, 5).map((activity, index) => (
            <div key={index} className="flex items-center space-x-4 p-3 rounded-xl bg-gray-500/10">
              <div className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0"></div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">
                  {activity.user_name} ({activity.role}) - {activity.department}
                </p>
                <p className="text-xs opacity-60 truncate">{activity.message_preview}</p>
              </div>
              <span className="text-xs opacity-40 flex-shrink-0">
                {new Date(activity.timestamp).toLocaleTimeString()}
              </span>
            </div>
          )) || (
            <p className="text-gray-500 text-center py-8">No recent activity</p>
          )}
        </div>
      </div>
    </div>
  );

  // FIXED: Users section with real-time data
  const renderUsers = () => (
    <div className="space-y-6">
      <DataTable 
        title="User Management (Real-time)"
        data={realTimeData.users || []}
        columns={[
          { key: 'name', label: 'Name' },
          { key: 'email', label: 'Email' },
          { 
            key: 'role', 
            label: 'Role',
            render: (user) => (
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                user.role === 'c_level' ? 'bg-yellow-500/20 text-yellow-400' :
                user.role === 'finance' ? 'bg-purple-500/20 text-purple-400' :
                user.role === 'hr' ? 'bg-pink-500/20 text-pink-400' :
                user.role === 'engineering' ? 'bg-green-500/20 text-green-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                {user.role?.toUpperCase() || 'EMPLOYEE'}
              </span>
            )
          },
          { key: 'department', label: 'Department' },
          { 
            key: 'is_online', 
            label: 'Status',
            render: (user) => (
              <div className="flex flex-col space-y-1">
                {user.is_online ? (
                  <span className="flex items-center space-x-1 text-green-400">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-xs">Online</span>
                  </span>
                ) : user.is_active ? (
                  <span className="flex items-center space-x-1 text-blue-400">
                    <CheckCircle className="w-3 h-3" />
                    <span className="text-xs">Active</span>
                  </span>
                ) : (
                  <span className="flex items-center space-x-1 text-red-400">
                    <XCircle className="w-3 h-3" />
                    <span className="text-xs">Inactive</span>
                  </span>
                )}
                {user.total_messages > 0 && (
                  <span className="text-xs opacity-60">{user.total_messages} msgs</span>
                )}
              </div>
            )
          },
          {
            key: 'actions',
            label: 'Actions',
            render: (user) => (
              <div className="flex space-x-2">
                <button 
                  onClick={() => {
                    setSelectedUser(user);
                    setShowUserDetails(true);
                  }}
                  className={buttonClasses('blue', 'secondary')}
                  title="View Details"
                >
                  <Eye className="w-4 h-4" />
                </button>
                <button 
                  onClick={() => {
                    setSelectedUser(user);
                    setShowEditUser(true);
                  }}
                  className={buttonClasses('orange', 'secondary')}
                  title="Edit User"
                >
                  <Edit3 className="w-4 h-4" />
                </button>
                <button 
                  onClick={() => toggleUserStatus(user.id, user.is_active)}
                  className={buttonClasses(user.is_active ? 'red' : 'green', 'secondary')}
                  title={user.is_active ? 'Deactivate' : 'Activate'}
                >
                  {user.is_active ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />}
                </button>
                <button 
                  onClick={() => showConfirmationDialog(() => handleDeleteUser(user.id), user)}
                  className={buttonClasses('red', 'secondary')}
                  title="Delete User"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            )
          }
        ]}
        actions={
          <button 
            onClick={() => setShowCreateUser(true)}
            className={buttonClasses('blue', 'primary')}
          >
            <Plus className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">Add User</span>
          </button>
        }
      />
    </div>
  );

  // FIXED: Documents section with real-time data
  const renderDocuments = () => (
    <div className="space-y-6">
      <DataTable 
        title="Document Management (Real-time)"
        data={realTimeData.documents || []}
        columns={[
          { key: 'filename', label: 'Document Name' },
          { key: 'file_type', label: 'Type' },
          { key: 'department', label: 'Department' },
          { 
            key: 'file_size', 
            label: 'Size',
            render: (doc) => `${(doc.file_size / 1024).toFixed(1)} KB`
          },
          { 
            key: 'created_at', 
            label: 'Uploaded',
            render: (doc) => new Date(doc.created_at).toLocaleDateString()
          },
          {
            key: 'actions',
            label: 'Actions',
            render: (doc) => (
              <div className="flex space-x-2">
                <button 
                  onClick={() => handleViewDocument(doc)}
                  className={buttonClasses('blue', 'secondary')} 
                  title="View"
                >
                  <Eye className="w-4 h-4" />
                </button>
                <button 
                  onClick={() => handleDownloadDocument(doc)}
                  className={buttonClasses('green', 'secondary')} 
                  title="Download"
                >
                  <Download className="w-4 h-4" />
                </button>
                <button 
                  onClick={() => showConfirmationDialog(() => handleDeleteDocument(doc.id), doc)}
                  className={buttonClasses('red', 'secondary')} 
                  title="Delete"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            )
          }
        ]}
        actions={
          <div className="flex space-x-2">
            <input
              ref={fileUploadRef}
              type="file"
              multiple
              accept=".csv,.txt,.md,.xlsx,.xls,.pdf"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
            <button 
              onClick={() => fileUploadRef.current?.click()}
              className={buttonClasses('blue', 'primary')}
              disabled={isLoading || isUploading}
            >
              <Upload className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">Upload Data</span>
            </button>
            <button 
              onClick={reloadDocuments}
              disabled={isLoading || isUploading}
              className={buttonClasses('green', 'primary')}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">Reload</span>
            </button>
          </div>
        }
      />
      {showDepartmentSelector && <DepartmentSelectorModal />}
    </div>
  );

  // Keep existing analytics, audit, and system render functions...
  const renderAnalytics = () => (
    <div className="space-y-8">
      {/* FIXED: Analytics Overview with real-time data */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        <StatCard 
          title="Success Rate" 
          value={realTimeData.analytics?.success_rate || 'N/A'} 
          icon={CheckCircle} 
          color="green"
          description="Request success rate"
        />
        <StatCard 
          title="Avg Response Time" 
          value={realTimeData.analytics?.avg_response_time || 'N/A'} 
          icon={Clock} 
          color="blue"
          description="Average processing time"
        />
        <StatCard 
          title="Total Requests" 
          value={realTimeData.analytics?.total_requests || 0} 
          icon={Activity} 
          color="purple"
          description="All-time requests"
        />
      </div>

      {/* Department Breakdown */}
      <div className={`p-6 ${cardClasses}`}>
        <h3 className="text-lg font-semibold mb-4">Department Activity</h3>
        <div className="space-y-4">
          {Object.entries(realTimeData.analytics?.department_breakdown || {}).map(([dept, count]) => (
            <div key={dept} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="capitalize">{dept}</span>
                <span>{count} users</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                  style={{ width: `${(count / Math.max(...Object.values(realTimeData.analytics?.department_breakdown || {}))) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderAuditLogs = () => (
    <div className="space-y-6">
      <DataTable 
        title="Audit Trail (Real-time)"
        data={realTimeData.auditLogs || []}
        columns={[
          { 
            key: 'created_at', 
            label: 'Time',
            render: (log) => (
              <div className="text-xs">
                <div>{new Date(log.created_at).toLocaleDateString()}</div>
                <div className="opacity-60">{new Date(log.created_at).toLocaleTimeString()}</div>
              </div>
            )
          },
          { key: 'email', label: 'User' },
          { 
            key: 'event_type', 
            label: 'Event',
            render: (log) => (
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                log.event_type.includes('SUCCESS') ? 'bg-green-500/20 text-green-400' :
                log.event_type.includes('DENIED') || log.event_type.includes('FAILURE') ? 'bg-red-500/20 text-red-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                {log.event_type}
              </span>
            )
          },
          { 
            key: 'event_details', 
            label: 'Details',
            render: (log) => (
              <span className="truncate max-w-xs block">
                {log.sensitive ? '[REDACTED]' : log.event_details}
              </span>
            )
          },
          { 
            key: 'sensitive', 
            label: 'Level',
            render: (log) => log.sensitive ? (
              <span className="flex items-center space-x-1 text-red-400">
                <AlertTriangle className="w-4 h-4" />
                <span className="hidden sm:inline">High</span>
              </span>
            ) : (
              <span className="flex items-center space-x-1 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span className="hidden sm:inline">Normal</span>
              </span>
            )
          }
        ]}
      />
    </div>
  );

  // Keep existing system, modals, etc. sections unchanged...
  const renderSystem = () => (
    <div className="space-y-8">
      {/* System Status */}
      <div className={`p-4 sm:p-6 ${cardClasses}`}>
        <h3 className="text-lg font-semibold mb-4">System Configuration</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Database Connection</span>
              <span className="flex items-center space-x-2 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span>Connected</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">AI Model Status</span>
              <span className="flex items-center space-x-2 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span>Online</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Memory System</span>
              <span className="flex items-center space-x-2 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span>Active</span>
              </span>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Security Level</span>
              <span className="flex items-center space-x-2 text-green-400">
                <ShieldCheck className="w-4 h-4" />
                <span>High</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Backup Status</span>
              <span className="flex items-center space-x-2 text-green-400">
                <CheckCircle className="w-4 h-4" />
                <span>Up to date</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">API Rate Limiting</span>
              <span className="flex items-center space-x-2 text-yellow-400">
                <AlertCircle className="w-4 h-4" />
                <span>Moderate</span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* System Actions */}
      <div className={`p-4 sm:p-6 ${cardClasses}`}>
        <h3 className="text-lg font-semibold mb-4">System Actions</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <button 
            onClick={reloadDocuments}
            disabled={isLoading}
            className="flex items-center space-x-3 p-4 rounded-xl bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Reload Documents</span>
          </button>
          <button 
            onClick={handleSystemBackup}
            disabled={isLoading}
            className="flex items-center space-x-3 p-4 rounded-xl bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors disabled:opacity-50"
          >
            <Database className="w-5 h-5" />
            <span>Backup Database</span>
          </button>
          <button 
            onClick={handleSystemHealthCheck}
            disabled={isLoading}
            className="flex items-center space-x-3 p-4 rounded-xl bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 transition-colors disabled:opacity-50"
          >
            <Monitor className="w-5 h-5" />
            <span>System Health</span>
          </button>
        </div>
      </div>
    </div>
  );

  const renderMemory = () => (
    <ConversationMemory
      isOpen={true}
      onClose={() => setActiveTab('overview')}
      darkMode={darkMode}
      user={user}
      apiCall={apiCall}
      addNotification={addNotification}
    />
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return renderOverview();
      case 'users': return renderUsers();
      case 'organizations': return <div className="text-center py-8 text-gray-500">Organizations management coming soon...</div>;
      case 'documents': return renderDocuments();
      case 'analytics': return renderAnalytics();
      case 'audit': return renderAuditLogs();
      case 'memory': return renderMemory();
      case 'system': return renderSystem();
      default: return renderOverview();
    }
  };

  // FIXED: Department Selector Modal with role selection
  const DepartmentSelectorModal = () => (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className={`w-full max-w-lg p-6 rounded-2xl shadow-2xl border ${
        darkMode 
          ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
          : 'bg-white/95 text-gray-900 border-gray-200/60'
      }`}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Upload className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-bold">Upload Files to Department</h2>
          </div>
          <button 
            onClick={() => {
              setShowDepartmentSelector(false);
              setFilesToUpload([]);
              setSelectedDepartment('');
              setSelectedTargetRole('');
            }}
            className="p-2 rounded-full hover:bg-gray-500/20"
            disabled={isUploading}
          >
            <X className="w-5 h-5 opacity-60" />
          </button>
        </div>

        <div className="space-y-6">
          {/* File List */}
          <div>
            <h3 className="text-sm font-medium mb-2">Files to Upload ({filesToUpload.length})</h3>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {filesToUpload.map((file, index) => (
                <div key={index} className={`flex items-center justify-between p-2 rounded-lg ${
                  darkMode ? 'bg-gray-800/60' : 'bg-gray-100/60'
                }`}>
                  <div className="flex items-center space-x-2">
                    <FileText className="w-4 h-4 opacity-60" />
                    <span className="text-sm truncate">{file.name}</span>
                  </div>
                  <span className="text-xs opacity-60">{(file.size / 1024).toFixed(1)} KB</span>
                </div>
              ))}
            </div>
          </div>

          {/* Department Selection */}
          <div>
            <label className="block text-sm font-medium mb-2">Target Department</label>
            <select
              value={selectedDepartment}
              onChange={(e) => {
                setSelectedDepartment(e.target.value);
                setSelectedTargetRole('');
              }}
              className={`w-full p-3 rounded-xl border ${
                darkMode 
                  ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                  : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
              }`}
              required
            >
              <option value="">Select Department</option>
              {departmentOptions.map(dept => (
                <option key={dept.value} value={dept.value}>{dept.label}</option>
              ))}
            </select>
          </div>

          {/* Role Selection */}
          {selectedDepartment && (
            <div>
              <label className="block text-sm font-medium mb-2">Target Role (Optional)</label>
              <select
                value={selectedTargetRole}
                onChange={(e) => setSelectedTargetRole(e.target.value)}
                className={`w-full p-3 rounded-xl border ${
                  darkMode 
                    ? 'bg-gray-900/90 border-gray-800/60 text-gray-100' 
                    : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                }`}
              >
                <option value="">All roles in department</option>
                {roleOptions
                  .filter(role => {
                    const dept = departmentOptions.find(d => d.value === selectedDepartment);
                    return dept?.roles.includes(role.value);
                  })
                  .map(role => (
                    <option key={role.value} value={role.value}>{role.label}</option>
                  ))
                }
              </select>
              <p className="text-xs opacity-60 mt-1">
                Leave empty to make accessible to all roles in the {selectedDepartment} department
              </p>
            </div>
          )}

          {/* Access Preview */}
          {selectedDepartment && (
            <div className={`p-3 rounded-lg ${
              darkMode ? 'bg-blue-500/10 border border-blue-500/20' : 'bg-blue-50 border border-blue-200'
            }`}>
              <h4 className="text-sm font-medium text-blue-400 mb-1">Access Preview</h4>
              <p className="text-xs opacity-80">
                These files will be searchable by: <strong>
                  {selectedTargetRole 
                    ? `${roleOptions.find(r => r.value === selectedTargetRole)?.label} only` 
                    : `All ${departmentOptions.find(d => d.value === selectedDepartment)?.label} users`}
                </strong>
              </p>
            </div>
          )}

          {/* Upload Progress */}
          {uploadProgress > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Uploading...</span>
                <span className="text-sm opacity-60">{uploadProgress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <button
              onClick={() => {
                setShowDepartmentSelector(false);
                setFilesToUpload([]);
                setSelectedDepartment('');
                setSelectedTargetRole('');
              }}
              className="flex-1 py-3 px-4 rounded-xl border border-gray-500/50 hover:bg-gray-500/10 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => uploadToDepartment()}
              disabled={!selectedDepartment || isUploading}
              className="flex-1 py-3 px-4 rounded-xl bg-blue-500 hover:bg-blue-600 text-white transition-colors disabled:opacity-50"
            >
              {isUploading ? 'Uploading...' : 'Upload Files'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // NEW: Create User Modal
  const CreateUserModal = () => {
    const [formData, setFormData] = useState({
      email: '',
      name: '',
      role: 'employee',
      department: '',
      password: ''
    });
    const [isSubmitting, setIsSubmitting] = useState(false);
    const emailRef = useRef(null);

    useEffect(() => {
      emailRef.current?.focus();
    }, []);

    const submit = async () => {
      if (!formData.email || !formData.name || !formData.password) {
        addNotification('❌ Please fill in all required fields', 'error');
        return;
      }
      try {
        setIsSubmitting(true);
        const res = await apiCall('/admin/users', {
          method: 'POST',
          body: JSON.stringify({ ...formData, email: formData.email.trim() })
        });
        if (res?.success) {
          addNotification('✅ User created successfully', 'success');
          await loadAdminData();
          setShowCreateUser(false);
        }
      } catch (err) {
        console.error('Create user failed', err);
        addNotification('❌ Failed to create user', 'error');
      } finally {
        setIsSubmitting(false);
      }
    };

    return (
      <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
        <div className={`w-full max-w-md p-6 rounded-2xl shadow-2xl border ${
          darkMode ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' : 'bg-white/95 text-gray-900 border-gray-200/60'
        }`}>
          <h3 className="text-lg font-semibold mb-4">Create New User</h3>
          <div className="space-y-4">
            <input
              ref={emailRef}
              type="email"
              placeholder="Email"
              value={formData.email}
              onChange={(e) => setFormData(p => ({ ...p, email: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            />
            <input
              type="text"
              placeholder="Full Name"
              value={formData.name}
              onChange={(e) => setFormData(p => ({ ...p, name: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            />
            <select
              value={formData.role}
              onChange={(e) => setFormData(p => ({ ...p, role: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            >
              {roleOptions.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
            <select
              value={formData.department}
              onChange={(e) => setFormData(p => ({ ...p, department: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            >
              <option value="">Select Department</option>
              {departmentOptions.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
            </select>
            <input
              type="password"
              placeholder="Password"
              value={formData.password}
              onChange={(e) => setFormData(p => ({ ...p, password: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            />
          </div>
          <div className="flex space-x-3 mt-6">
            <button
              onClick={() => setShowCreateUser(false)}
              className="flex-1 py-3 px-4 rounded-xl border border-gray-500/50 hover:bg-gray-500/10 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={isSubmitting}
              className="flex-1 py-3 px-4 rounded-xl bg-blue-500 hover:bg-blue-600 text-white transition-colors disabled:opacity-50"
            >
              {isSubmitting ? 'Creating...' : 'Create User'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Modal for editing an existing user
  const EditUserModal = () => {
    const [isSubmitting, setIsSubmitting] = useState(false);

    const submit = async () => {
      setIsSubmitting(true);
      await handleEditUser();
      setIsSubmitting(false);
    };

    if (!selectedUser) return null;

    return (
      <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
        <div className={`w-full max-w-md p-6 rounded-2xl shadow-2xl border ${
          darkMode ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' : 'bg-white/95 text-gray-900 border-gray-200/60'
        }`}>
          <h3 className="text-lg font-semibold mb-4">Edit User</h3>
          <div className="space-y-4">
            <input
              type="text"
              placeholder="Full Name"
              value={selectedUser.name}
              onChange={(e) => setSelectedUser((p) => ({ ...p, name: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            />
            <select
              value={selectedUser.role}
              onChange={(e) => setSelectedUser((p) => ({ ...p, role: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            >
              {roleOptions.map((r) => (
                <option key={r.value} value={r.value}>
                  {r.label}
                </option>
              ))}
            </select>
            <select
              value={selectedUser.department}
              onChange={(e) => setSelectedUser((p) => ({ ...p, department: e.target.value }))}
              className="w-full p-3 rounded-xl border bg-gray-100/50 border-gray-200/40 dark:bg-gray-900/80 dark:border-gray-800/60"
            >
              {departmentOptions.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
            <label className="flex items-center space-x-2 text-sm">
              <input
                type="checkbox"
                checked={selectedUser.is_active}
                onChange={(e) => setSelectedUser((p) => ({ ...p, is_active: e.target.checked }))}
              />
              <span>Active</span>
            </label>
          </div>
          <div className="flex space-x-3 mt-6">
            <button
              onClick={() => {
                setShowEditUser(false);
                setSelectedUser(null);
              }}
              className="flex-1 py-3 px-4 rounded-xl border border-gray-500/50 hover:bg-gray-500/10 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={isSubmitting || isLoading}
              className="flex-1 py-3 px-4 rounded-xl bg-orange-500 hover:bg-orange-600 text-white transition-colors disabled:opacity-50"
            >
              {isSubmitting || isLoading ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Modal for viewing user details (read-only)
  const UserDetailsModal = () => {
    if (!selectedUser) return null;

    return (
      <div className="fixed inset-0 z-70 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
        <div className={`w-full max-w-md p-6 rounded-2xl shadow-2xl border ${
          darkMode ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' : 'bg-white/95 text-gray-900 border-gray-200/60'
        }`}>
          <h3 className="text-lg font-semibold mb-4">User Details</h3>
          <ul className="space-y-2 text-sm">
            <li><strong>Name:</strong> {selectedUser.name}</li>
            <li><strong>Email:</strong> {selectedUser.email}</li>
            <li><strong>Role:</strong> {selectedUser.role}</li>
            <li><strong>Department:</strong> {selectedUser.department}</li>
            <li><strong>Status:</strong> {selectedUser.is_active ? 'Active' : 'Inactive'}</li>
            <li><strong>Messages:</strong> {selectedUser.total_messages}</li>
          </ul>
          <div className="flex justify-end mt-6">
            <button
              onClick={() => {
                setShowUserDetails(false);
                setSelectedUser(null);
              }}
              className="py-2 px-4 rounded-xl bg-blue-500 hover:bg-blue-600 text-white transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex bg-black/50 backdrop-blur-sm">
      <div className={`w-full h-full flex flex-col rounded-lg sm:rounded-3xl sm:m-4 shadow-2xl border transition-all duration-300 ${
        darkMode 
          ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
          : 'bg-white/95 text-gray-900 border-gray-200/60'
      } overflow-hidden`}>
        
        {/* FIXED: Header with real-time indicator */}
        <div className="flex items-center justify-between p-4 sm:p-6 border-b border-gray-800/60 flex-shrink-0">
          <div className="flex items-center space-x-3">
            <Shield className="w-6 h-6 sm:w-8 sm:h-8 text-red-500" />
            <div>
              <h1 className="text-xl sm:text-2xl font-bold">Admin Dashboard</h1>
              <p className="text-sm opacity-60 hidden sm:block">
                Real-time system monitoring and administration
                {autoRefresh && <span className="ml-2 text-green-400">● Live</span>}
              </p>
            </div>
          </div>
          <div className="flex space-x-2">
            {onLogout && (
              <button
                onClick={onLogout}
                className="p-2 rounded-full hover:bg-red-500/20 text-red-400 transition-colors duration-200"
                title="Admin Logout"
              >
                <LogOut className="w-5 h-5" />
              </button>
            )}
            <button 
              onClick={onClose}
              className="p-2 rounded-full hover:bg-gray-500/20 transition-colors duration-200"
              title="Close Dashboard"
            >
              <X className="w-5 h-5 sm:w-6 sm:h-6 opacity-60" />
            </button>
          </div>
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
                      ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                      : 'hover:bg-gray-500/10'
                  }`}
                  title={tab.label}
                >
                  <tab.icon className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
                  <span className="font-medium hidden sm:inline truncate">{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-8">
            {isLoading && (
              <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-60">
                <div className={`p-6 rounded-2xl ${cardClasses} flex items-center space-x-3`}>
                  <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
                  <span>Loading real-time data...</span>
                </div>
              </div>
            )}
            {renderContent()}
          </div>
        </div>

        {/* Keep all existing modals unchanged */}
        {showDocumentViewer && selectedDocument && (
          <DocumentViewerModal
            document={selectedDocument}
            onClose={() => setShowDocumentViewer(false)}
            darkMode={darkMode}
          />
        )}

        {/* NEW: Generic confirmation modal (used for deletes) */}
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
                  <h3 className="text-lg font-semibold">Confirm Action</h3>
                  <p className="text-sm opacity-60">This action cannot be undone</p>
                </div>
              </div>

              <p className="text-sm mb-6 opacity-80">
                Are you sure you want to proceed? The selected item will be permanently deleted.
              </p>

              <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-3">
                <button
                  onClick={() => {
                    setShowConfirmDialog(false);
                    setConfirmAction(null);
                  }}
                  className="flex-1 py-3 px-4 rounded-xl border border-gray-500/50 hover:bg-gray-500/10 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmAction}
                  disabled={isLoading}
                  className="flex-1 py-3 px-4 rounded-xl bg-red-500 hover:bg-red-600 text-white transition-colors disabled:opacity-50"
                >
                  {isLoading ? 'Processing...' : 'Confirm'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* New: render modals */}
        {showDepartmentSelector && <DepartmentSelectorModal />}
        {showCreateUser && <CreateUserModal />}
        {showEditUser && selectedUser && <EditUserModal />}
        {showUserDetails && selectedUser && <UserDetailsModal />}
      </div>
    </div>
  );
};

export default AdminDashboard;