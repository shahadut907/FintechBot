import React, { useState } from 'react';
import { Eye, EyeOff, LogIn, User, Lock, Brain, Shield, Building } from 'lucide-react';

const LoginForm = ({ onLogin, isLoading, error, darkMode }) => {
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [selectedDemo, setSelectedDemo] = useState('');

  // Demo users (disabled in production)
  const SHOW_DEMO_USERS = false;
  const demoUsers = SHOW_DEMO_USERS ? [
    { username: 'john.doe@finsolve.com', password: 'password123', role: 'Finance Manager', department: 'Finance' },
    { username: 'jane.smith@finsolve.com', password: 'password123', role: 'HR Director', department: 'Human Resources' },
    { username: 'mike.johnson@finsolve.com', password: 'password123', role: 'Senior Engineer', department: 'Engineering' },
    { username: 'sarah.wilson@finsolve.com', password: 'password123', role: 'Marketing Lead', department: 'Marketing' },
    { username: 'david.brown@finsolve.com', password: 'password123', role: 'CEO', department: 'Executive' },
    { username: 'employee@finsolve.com', password: 'password123', role: 'General Employee', department: 'General' }
  ] : [];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (credentials.username && credentials.password) {
      onLogin(credentials);
    }
  };

  const handleDemoSelect = (user) => {
    setCredentials({
      username: user.username,
      password: user.password
    });
    setSelectedDemo(user.username);
  };

  const handleQuickLogin = (user) => {
    onLogin({
      username: user.username,
      password: user.password
    });
  };

  return (
    <div className="w-full max-w-md mx-auto p-4">
      {/* Logo and Header */}
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-gradient-to-r from-blue-500 via-purple-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-2xl">
            <Brain className="w-8 h-8 text-white" />
          </div>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold mb-2">FinSolve AI</h1>
        <p className={`text-sm sm:text-base ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Intelligent Role-Based Assistant
        </p>
      </div>

      {/* Main Login Form */}
      <div className={`p-6 sm:p-8 rounded-3xl shadow-2xl border backdrop-blur-sm ${
        darkMode 
          ? 'bg-gray-950/95 text-white border-gray-800/60'
          : 'bg-white/95 text-gray-900 border-gray-200/60'
      }`}>
        
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-400/40 rounded-xl text-red-200 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Username/Email Field */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Email Address
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className={`w-5 h-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              </div>
              <input
                type="email"
                value={credentials.username}
                onChange={(e) => setCredentials(prev => ({ ...prev, username: e.target.value }))}
                className={`block w-full pl-10 pr-3 py-3 border rounded-xl transition-all duration-200 ${
                  darkMode 
                    ? 'bg-gray-900/80 border-gray-800/60 text-gray-100 placeholder-gray-400 focus:bg-gray-900 focus:border-gray-700' 
                    : 'bg-gray-50 border-gray-200/60 text-gray-900 placeholder-gray-500 focus:bg-white focus:border-gray-300'
                } focus:outline-none focus:ring-2 focus:ring-blue-500/50`}
                placeholder="Enter your email"
                required
              />
            </div>
          </div>

          {/* Password Field */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className={`w-5 h-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                value={credentials.password}
                onChange={(e) => setCredentials(prev => ({ ...prev, password: e.target.value }))}
                className={`block w-full pl-10 pr-12 py-3 border rounded-xl transition-all duration-200 ${
                  darkMode 
                    ? 'bg-gray-900/80 border-gray-800/60 text-gray-100 placeholder-gray-400 focus:bg-gray-900 focus:border-gray-700' 
                    : 'bg-gray-50 border-gray-200/60 text-gray-900 placeholder-gray-500 focus:bg-white focus:border-gray-300'
                } focus:outline-none focus:ring-2 focus:ring-blue-500/50`}
                placeholder="Enter your password"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                {showPassword ? (
                  <EyeOff className={`w-5 h-5 ${darkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-500 hover:text-gray-700'} transition-colors`} />
                ) : (
                  <Eye className={`w-5 h-5 ${darkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-500 hover:text-gray-700'} transition-colors`} />
                )}
              </button>
            </div>
          </div>

          {/* Login Button */}
          <button
            type="submit"
            disabled={isLoading || !credentials.username || !credentials.password}
            className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-xl font-medium transition-all duration-200 ${
              isLoading || !credentials.username || !credentials.password
                ? 'bg-gray-500/50 cursor-not-allowed text-gray-400'
                : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
            }`}
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Signing In...</span>
              </>
            ) : (
              <>
                <LogIn className="w-5 h-5" />
                <span>Sign In</span>
              </>
            )}
          </button>
        </form>

        {/* Demo Users Section (disabled) */}
        {SHOW_DEMO_USERS && (
          <div className="mt-8">
            <div className="flex items-center mb-4">
              <div className={`flex-1 h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-200'}`}></div>
              <span className={`px-3 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Demo Accounts</span>
              <div className={`flex-1 h-px ${darkMode ? 'bg-gray-800' : 'bg-gray-200'}`}></div>
            </div>

            <div className="space-y-3">
              {demoUsers.map((user, index) => (
                <div 
                  key={index}
                  className={`p-3 rounded-xl border transition-all duration-200 cursor-pointer ${
                    selectedDemo === user.username
                      ? 'border-blue-500/50 bg-blue-500/10'
                      : darkMode
                        ? 'border-gray-800/60 bg-gray-900/50 hover:bg-gray-800/50'
                        : 'border-gray-200/60 bg-gray-100/50 hover:bg-gray-100'
                  }`}
                  onClick={() => handleDemoSelect(user)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          user.department === 'Finance' ? 'bg-purple-500/20 text-purple-400' :
                          user.department === 'Human Resources' ? 'bg-pink-500/20 text-pink-400' :
                          user.department === 'Engineering' ? 'bg-green-500/20 text-green-400' :
                          user.department === 'Marketing' ? 'bg-orange-500/20 text-orange-400' :
                          user.department === 'Executive' ? 'bg-yellow-500/20 text-yellow-400' :
                          'bg-blue-500/20 text-blue-400'
                        }`}>
                          {user.department === 'Executive' ? <Building className="w-4 h-4" /> :
                           user.department === 'Engineering' ? <Brain className="w-4 h-4" /> :
                           <User className="w-4 h-4" />}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate">{user.role}</p>
                          <p className={`text-xs truncate ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            {user.username}
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDemoSelect(user);
                        }}
                        className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                          selectedDemo === user.username
                            ? 'bg-blue-500/30 text-blue-300'
                            : 'bg-gray-500/20 text-gray-400 hover:bg-gray-500/30'
                        }`}
                      >
                        Select
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleQuickLogin(user);
                        }}
                        className="px-3 py-1 text-xs bg-green-500/20 text-green-400 hover:bg-green-500/30 rounded-lg transition-colors"
                      >
                        Quick Login
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer Info */}
        <div className={`mt-6 pt-6 border-t text-center text-xs ${
          darkMode ? 'border-gray-800 text-gray-400' : 'border-gray-200 text-gray-600'
        }`}>
          <p className="mb-2">
            🔒 Secure Role-Based Access Control
          </p>
          <p>
            Each role has access to different data and functionality
          </p>
        </div>
      </div>

      {/* Additional Info */}
      <div className="mt-6 text-center">
        <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-xl ${
          darkMode ? 'bg-gray-900/50 text-gray-400' : 'bg-gray-100/50 text-gray-600'
        }`}>
          <Shield className="w-4 h-4" />
          <span className="text-sm">Enterprise Security Enabled</span>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;