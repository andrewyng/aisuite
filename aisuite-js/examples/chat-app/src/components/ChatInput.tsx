import React, { useState, KeyboardEvent } from 'react';
import { Send, RotateCcw } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onResetChat: () => void;
  isLoading: boolean;
  placeholder?: string;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onResetChat,
  isLoading,
  placeholder = "Enter your query...",
  disabled = false
}) => {
  const [message, setMessage] = useState('');

  const handleSend = () => {
    if (message.trim() && !isLoading && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t bg-background p-4">
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={isLoading || disabled}
            className="w-full resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            rows={3}
            style={{ minHeight: '60px', maxHeight: '120px' }}
          />
        </div>
        
        <div className="flex flex-col gap-2">
          <button
            onClick={handleSend}
            disabled={!message.trim() || isLoading || disabled}
            className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
          
          <button
            onClick={onResetChat}
            disabled={isLoading}
            className="flex items-center justify-center w-10 h-10 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Reset Chat"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>
      
      {isLoading && (
        <div className="mt-2 text-sm text-muted-foreground flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-muted-foreground border-t-transparent rounded-full animate-spin" />
          Processing...
        </div>
      )}
    </div>
  );
}; 