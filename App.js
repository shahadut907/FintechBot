import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import LoginForm from './components/LoginForm';
import AdminDashboard from './components/AdminDashboard';
import { Sun, Moon, Shield, LogOut } from 'lucide-react';

const FinSolveAssistant = () => {
  // Core Authentication State
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [authToken, setAuthToken] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // UI State
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('finsolve-dark-mode');
    return saved !== null ? JSON.parse(saved) : true;
  });

  // FIXED: Separate Admin State - Independent of user roles
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false);
  const [adminUser, setAdminUser] = useState(null);
  const [showAdminLogin, setShowAdminLogin] = useState(false);

  // Notifications
  const [notifications, setNotifications] = useState([]);

  // API Configuration
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const TOKEN_KEY = 'finsolve_auth_token';
  const USER_KEY = 'finsolve_user_data';
  const ADMIN_TOKEN_KEY = 'finsolve_admin_token';
  const ADMIN_USER_KEY = 'finsolve_admin_user';

  // Utility Functions
  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    const notification = { id, message, type, timestamp: new Date() };
    setNotifications(prev => [...prev, notification]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 4000);
  };

  const apiCall = async (endpoint, options = {}) => {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      const token = authToken || localStorage.getItem(TOKEN_KEY);
      const adminToken = localStorage.getItem(ADMIN_TOKEN_KEY);
      
      // If endpoint is admin → admin token.
      // For non-admin endpoints use user token; if it doesn't exist but admin token is present (admin dashboard context) use the admin token instead so authenticated requests still succeed.
      const useToken = endpoint.startsWith('/admin/') ? adminToken : (token || adminToken);
      
      const config = {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...(useToken && { 'Authorization': `Bearer ${useToken}` }),
        },
        ...options,
      };
      
      const response = await fetch(url, config);
      
      if (response.status === 401) {
        console.warn('Token expired, logging out...');
        if (endpoint.startsWith('/admin/')) {
          handleAdminLogout();
        } else {
          handleLogout();
        }
        throw new Error('Authentication failed');
      }
      
      if (!response.ok) {
        let errorText = `Unknown API Error: ${response.status}`;
        if (response && typeof response.text === 'function') {
          try {
            errorText = await response.text();
          } catch (e) {
            console.error("Failed to read response text:", e);
          }
        }
        throw new Error(`API Error: ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Call failed:`, error);
      throw error;
    }
  };

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('finsolve-dark-mode', JSON.stringify(newDarkMode));

    // Apply theme immediately
    const root = document.documentElement;
    if (newDarkMode) {
      root.classList.add('dark');
      root.style.setProperty('--bg-primary', '#000000');
      root.style.setProperty('--text-primary', '#ffffff');
    } else {
      root.classList.remove('dark');
      root.style.setProperty('--bg-primary', '#ffffff');
      root.style.setProperty('--text-primary', '#000000');
    }
  };

  useEffect(() => {
    const initializeApp = () => {
      // Apply theme on mount or darkMode change
      const root = document.documentElement;
      if (darkMode) {
        root.classList.add('dark');
        root.style.setProperty('--bg-primary', '#000000');
        root.style.setProperty('--text-primary', '#ffffff');
      } else {
        root.classList.remove('dark');
        root.style.setProperty('--bg-primary', '#ffffff');
        root.style.setProperty('--text-primary', '#000000');
      }
    };

    initializeApp();
  }, [darkMode]);

  const handleLogin = async (credentials) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await apiCall('/auth/login', {
        method: 'POST',
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password
        })
      });
      
      if (response.access_token) {
        setAuthToken(response.access_token);
        localStorage.setItem(TOKEN_KEY, response.access_token);
        localStorage.setItem(USER_KEY, JSON.stringify(response.user));
        
        setUser(response.user);
        setIsAuthenticated(true);
        
        addNotification(`🎉 Welcome back, ${response.user.name.split(' ')[0]}!`, 'success');
      }
    } catch (error) {
      console.error('Login error:', error);
      setError('Login failed. Please check your credentials.');
      addNotification('❌ Login failed. Please try again.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    setAuthToken(null);
    setUser(null);
    setIsAuthenticated(false);
    addNotification('👋 Successfully logged out', 'info');
  };

  // FIXED: Separate admin authentication functions
  const handleAdminLogin = async (credentials) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await apiCall('/admin/auth/login', {
        method: 'POST',
        body: JSON.stringify({
          username: credentials.username,
          password: credentials.password
        })
      });
      
      if (response.access_token) {
        localStorage.setItem(ADMIN_TOKEN_KEY, response.access_token);
        localStorage.setItem(ADMIN_USER_KEY, JSON.stringify(response.admin));
        
        if (response.admin && (response.admin.role === 'c_level' || response.admin.role === 'admin')) {
          setAdminUser(response.admin);
          setIsAdminAuthenticated(true);
          setShowAdminLogin(false);
          addNotification('🎉 Welcome Admin!', 'success');
        } else {
          addNotification('❌ Not an admin user', 'error');
        }
      }
    } catch (error) {
      console.error('Admin login error:', error);
      addNotification('❌ Invalid admin credentials', 'error');
    } finally {
      setIsLoading(false); 
    }
  };

  const handleAdminLogout = () => {
    localStorage.removeItem(ADMIN_TOKEN_KEY);
    localStorage.removeItem(ADMIN_USER_KEY);
    setAdminUser(null);
    setIsAdminAuthenticated(false);
    addNotification('🔒 Admin logged out', 'info');
  };

  // Initialize app
  useEffect(() => {
    const initializeApp = () => {
      // Apply dark mode
      if (darkMode) {
        document.documentElement.classList.add('dark');
        document.body.style.backgroundColor = '#000000';
        document.body.style.color = '#ffffff';
      } else {
        document.documentElement.classList.remove('dark');
        document.body.style.backgroundColor = '#ffffff';
        document.body.style.color = '#000000';
      }
      
      // Check stored tokens
      const storedToken = localStorage.getItem(TOKEN_KEY);
      const storedUser = localStorage.getItem(USER_KEY);
      const storedAdminToken = localStorage.getItem(ADMIN_TOKEN_KEY);
      const storedAdminUser = localStorage.getItem(ADMIN_USER_KEY);
      
      // Restore admin session
      if (storedAdminToken && storedAdminUser) {
        try {
          const adminData = JSON.parse(storedAdminUser);
          if (adminData && (adminData.role === 'c_level' || adminData.role === 'admin')) {
            setIsAdminAuthenticated(true);
            setAdminUser(adminData);
          } else {
            localStorage.removeItem(ADMIN_TOKEN_KEY);
            localStorage.removeItem(ADMIN_USER_KEY);
          }
        } catch (e) {
          localStorage.removeItem(ADMIN_TOKEN_KEY);
          localStorage.removeItem(ADMIN_USER_KEY);
        }
      }
      
      // Restore user session
      if (storedToken && storedUser) {
        try {
          setAuthToken(storedToken);
          setUser(JSON.parse(storedUser));
          setIsAuthenticated(true);
        } catch (e) {
          localStorage.removeItem(TOKEN_KEY);
          localStorage.removeItem(USER_KEY);
        }
      }
    };

    initializeApp();
  }, [darkMode]);

  // Notification Component
  const NotificationContainer = () => (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={`max-w-sm p-4 rounded-lg shadow-lg border transition-all duration-300 transform translate-x-0 ${
            notification.type === 'success' ? 'bg-green-500/20 border-green-500/50 text-green-100' :
            notification.type === 'error' ? 'bg-red-500/20 border-red-500/50 text-red-100' :
            notification.type === 'warning' ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-100' :
            'bg-blue-500/20 border-blue-500/50 text-blue-100'
          }`}
        >
          <p className="text-sm font-medium">{notification.message}</p>
        </div>
      ))}
    </div>
  );

  // UPDATED: Admin login button visible ONLY on main login screen
  const AdminControls = () => {
    if (isAuthenticated || isAdminAuthenticated) return null; // hide inside conversation/dashboard
    return (
      <div className="fixed top-4 left-4 z-60">
        <button
          onClick={() => setShowAdminLogin(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg backdrop-blur-sm border border-red-500/40 shadow-lg"
        >
          <Shield className="w-4 h-4" />
          <span className="text-sm font-semibold">Admin Login</span>
        </button>
      </div>
    );
  };

  // FIXED: Admin Login Modal
  const AdminLoginModal = () => {
    const [adminCredentials, setAdminCredentials] = useState({ username: '', password: '' });

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
        <div className={`w-full max-w-md p-6 rounded-2xl shadow-2xl border ${
          darkMode 
            ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
            : 'bg-white/95 text-gray-900 border-gray-200/60'
        }`}>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Shield className="w-6 h-6 text-red-500" />
              <h2 className="text-xl font-bold">Admin Login</h2>
            </div>
            <button 
              onClick={() => setShowAdminLogin(false)}
              className="p-2 rounded-full hover:bg-gray-500/20"
            >
              <LogOut className="w-5 h-5 opacity-60" />
            </button>
          </div>

          <form onSubmit={(e) => {
            e.preventDefault();
            handleAdminLogin(adminCredentials);
          }} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Admin Username</label>
              <input
                type="text"
                value={adminCredentials.username}
                onChange={(e) => setAdminCredentials(prev => ({ ...prev, username: e.target.value }))}
                className={`w-full p-3 rounded-xl border ${
                  darkMode 
                    ? 'bg-gray-900/80 border-gray-800/60 text-gray-100' 
                    : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                }`}
                placeholder="Enter admin username"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Admin Password</label>
              <input
                type="password"
                value={adminCredentials.password}
                onChange={(e) => setAdminCredentials(prev => ({ ...prev, password: e.target.value }))}
                className={`w-full p-3 rounded-xl border ${
                  darkMode 
                    ? 'bg-gray-900/80 border-gray-800/60 text-gray-100' 
                    : 'bg-gray-100/50 border-gray-200/40 text-gray-900'
                }`}
                placeholder="Enter admin password"
                required
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 bg-red-500 hover:bg-red-600 text-white rounded-xl transition-colors disabled:opacity-50"
            >
              {isLoading ? 'Authenticating...' : 'Login as Admin'}
            </button>
          </form>

          <div className="mt-4 p-3 bg-blue-500/10 rounded-lg">
            <p className="text-xs text-blue-400">Demo Admin Credentials:</p>
            <p className="text-xs font-mono">Username: admin | Password: admin123</p>
            <p className="text-xs font-mono">Username: demo_admin | Password: demo123</p>
          </div>
        </div>
      </div>
    );
  };

  // Admin sees dashboard exclusively
  if (isAdminAuthenticated) {
    return (
      <div className={`min-h-screen ${darkMode ? 'bg-black text-white' : 'bg-white text-black'}`}>
        <AdminDashboard
          isOpen={true}
          darkMode={darkMode}
          user={adminUser}
          apiCall={apiCall}
          addNotification={addNotification}
          onLogout={handleAdminLogout}
        />
        <NotificationContainer />
      </div>
    );
  }

  // Main application flow
  if (!isAuthenticated && !isAdminAuthenticated) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${
        darkMode ? 'bg-black' : 'bg-gray-50'
      }`}>
        <AdminControls />
        <LoginForm
          onLogin={handleLogin}
          isLoading={isLoading}
          error={error}
          darkMode={darkMode}
        />
        <NotificationContainer />
        {showAdminLogin && <AdminLoginModal />}
      </div>
    );
  }

  // Main Chat Interface
  return (
    <div className={`min-h-screen ${darkMode ? 'bg-black' : 'bg-white'}`}>
      <AdminControls />
      <ChatInterface
        user={user}
        darkMode={darkMode}
        apiCall={apiCall}
        addNotification={addNotification}
        authToken={authToken}
        onLogout={handleLogout}
        onToggleTheme={toggleDarkMode}
      />
      <NotificationContainer />
      {showAdminLogin && <AdminLoginModal />}
    </div>
  );
};

export default FinSolveAssistant;