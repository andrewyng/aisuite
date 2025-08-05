import React, { useRef, useEffect } from 'react';
import { Message } from '../types/chat';
import { ChatMessage } from './ChatMessage';

interface ChatContainerProps {
  messages: Message[];
  modelName: string;
  isLoading?: boolean;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({ 
  messages, 
  modelName, 
  isLoading = false 
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="min-h-full">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <div className="text-center">
                <div className="text-lg font-medium mb-2">No messages yet</div>
                <div className="text-sm">Start a conversation with {modelName}</div>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <ChatMessage 
                key={index} 
                message={message} 
                modelName={modelName}
              />
            ))
          )}
          
          {isLoading && (
            <div className="flex gap-3 p-4 justify-start">
              <div className="flex-shrink-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
              </div>
              <div className="max-w-[80%]">
                <div className="rounded-lg p-3 bg-muted text-foreground">
                  <div className="text-sm font-medium mb-1 opacity-70">
                    {modelName}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
}; 