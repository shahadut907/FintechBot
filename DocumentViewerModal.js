import React from 'react';
import { X } from 'lucide-react';

const DocumentViewerModal = ({ onClose, document, darkMode }) => {
  if (!document) return null;

  const cardClasses = `rounded-2xl border transition-all duration-300 ${
    darkMode 
      ? 'bg-gray-950/98 text-gray-100 border-gray-800/60' 
      : 'bg-white/95 text-gray-900 border-gray-200/60'
  }`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className={`w-full max-w-3xl max-h-[90vh] flex flex-col ${cardClasses} shadow-2xl overflow-hidden`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 sm:p-6 border-b border-gray-800/60 flex-shrink-0">
          <h2 className="text-xl sm:text-2xl font-bold truncate">
            {document.filename || 'Document Preview'}
          </h2>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-500/20 transition-colors duration-200"
          >
            <X className="w-5 h-5 sm:w-6 sm:h-6 opacity-60" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 p-4 sm:p-6 overflow-y-auto whitespace-pre-wrap font-mono text-sm">
          {document.content || 'No content available for this document.'}
        </div>
      </div>
    </div>
  );
};

export default DocumentViewerModal;
