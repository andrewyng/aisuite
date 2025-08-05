import React from 'react';
import { Message } from '../types/chat';
import { User, Bot } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
  modelName?: string;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, modelName }) => {
  const isUser = message.role === 'user';
  const roleDisplay = isUser ? 'User' : modelName || 'Assistant';

  return (
    <div className={`flex gap-3 p-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center">
          <Bot className="w-4 h-4 text-primary-foreground" />
        </div>
      )}
      
      <div className={`max-w-[80%] ${isUser ? 'order-first' : ''}`}>
        <div className={`rounded-lg p-3 ${
          isUser 
            ? 'bg-primary text-primary-foreground' 
            : 'bg-muted text-foreground'
        }`}>
          <div className="text-sm font-medium mb-1 opacity-70">
            {roleDisplay}
          </div>
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        </div>
        {message.timestamp && (
          <div className={`text-xs text-muted-foreground mt-1 ${
            isUser ? 'text-right' : 'text-left'
          }`}>
            {message.timestamp.toLocaleTimeString()}
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
          <User className="w-4 h-4 text-secondary-foreground" />
        </div>
      )}
    </div>
  );
}; 