import React, { useState, useEffect } from 'react';
import { Settings, AlertCircle } from 'lucide-react';
import { Message, AISuiteConfig } from './types/chat';
import { configuredLLMs, getLLMConfigByName } from './config/llm-config';
import { aiSuiteService } from './services/aisuite-service';
import { ChatContainer } from './components/ChatContainer';
import { ChatInput } from './components/ChatInput';
import { ModelSelector } from './components/ModelSelector';
import { ProviderSelector } from './components/ProviderSelector';
import { ApiKeyModal } from './components/ApiKeyModal';

function App() {
  const [chatHistory1, setChatHistory1] = useState<Message[]>([]);
  const [chatHistory2, setChatHistory2] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [useComparisonMode, setUseComparisonMode] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel1, setSelectedModel1] = useState('');
  const [selectedModel2, setSelectedModel2] = useState('');
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiConfig, setApiConfig] = useState<AISuiteConfig>({});
  const [error, setError] = useState<string | null>(null);

  // Initialize AISuite service when API config changes
  useEffect(() => {
    if (Object.keys(apiConfig).length > 0) {
      try {
        aiSuiteService.initialize(apiConfig);
        setError(null);
      } catch (err) {
        setError('Failed to initialize AISuite client');
      }
    }
  }, [apiConfig]);

  // Load API config from localStorage on mount
  useEffect(() => {
    const savedConfig = localStorage.getItem('aisuite-config');
    if (savedConfig) {
      try {
        const config = JSON.parse(savedConfig);
        setApiConfig(config);
      } catch (err) {
        console.error('Failed to load saved config');
      }
    }
  }, []);

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;

    // Check if provider is selected
    if (!selectedProvider) {
      setError('Please select a provider first');
      return;
    }

    // Check if API key is configured for the selected provider
    if (!apiConfig[selectedProvider as keyof AISuiteConfig]?.apiKey) {
      setError(`API key for ${selectedProvider} is not configured. Please configure it first.`);
      setShowApiKeyModal(true);
      return;
    }

    const userMessage: Message = {
      role: 'user',
      content: message,
      timestamp: new Date()
    };

    setIsProcessing(true);
    setError(null);

    try {
      // Add user message to both chat histories
      setChatHistory1(prev => [...prev, userMessage]);
      if (useComparisonMode) {
        setChatHistory2(prev => [...prev, userMessage]);
      }

      // Get model configurations
      const modelConfig1 = getLLMConfigByName(selectedModel1);
      if (!modelConfig1) {
        throw new Error(`Model ${selectedModel1} not found`);
      }

      // Query first model
      const response1 = await aiSuiteService.queryLLM(modelConfig1, [...chatHistory1, userMessage]);
      const assistantMessage1: Message = {
        role: 'assistant',
        content: response1,
        timestamp: new Date()
      };
      setChatHistory1(prev => [...prev, assistantMessage1]);

      // Query second model if in comparison mode
      if (useComparisonMode) {
        const modelConfig2 = getLLMConfigByName(selectedModel2);
        if (!modelConfig2) {
          throw new Error(`Model ${selectedModel2} not found`);
        }

        const response2 = await aiSuiteService.queryLLM(modelConfig2, [...chatHistory2, userMessage]);
        const assistantMessage2: Message = {
          role: 'assistant',
          content: response2,
          timestamp: new Date()
        };
        setChatHistory2(prev => [...prev, assistantMessage2]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleResetChat = () => {
    setChatHistory1([]);
    setChatHistory2([]);
    setError(null);
  };

  const handleSaveApiConfig = (config: AISuiteConfig) => {
    setApiConfig(config);
    localStorage.setItem('aisuite-config', JSON.stringify(config));
  };

  // Get all available providers (show all by default)
  const allProviders = ['openai', 'anthropic', 'groq', 'mistral'];
  const availableProviders = allProviders;
  
  // Get configured providers (those with API keys)
  const configuredProviders = Object.keys(apiConfig).filter(provider => 
    apiConfig[provider as keyof AISuiteConfig]?.apiKey
  );

  // Get models for the selected provider
  const availableModels = selectedProvider 
    ? configuredLLMs.filter(model => model.provider === selectedProvider)
    : [];

  // Reset model selections when provider changes
  useEffect(() => {
    if (selectedProvider) {
      const providerModels = configuredLLMs.filter(model => model.provider === selectedProvider);
      if (providerModels.length > 0) {
        setSelectedModel1(providerModels[0].name);
        if (useComparisonMode && providerModels.length > 1) {
          setSelectedModel2(providerModels[1].name);
        } else {
          setSelectedModel2('');
        }
      } else {
        setSelectedModel1('');
        setSelectedModel2('');
      }
    } else {
      setSelectedModel1('');
      setSelectedModel2('');
    }
  }, [selectedProvider, useComparisonMode]);

  const hasConfiguredProviders = Object.keys(apiConfig).length > 0;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">AISuite Chat</h1>
            <button
              onClick={() => setShowApiKeyModal(true)}
              className="flex items-center gap-2 rounded-lg border border-input bg-background px-3 py-2 text-sm font-medium ring-offset-background placeholder:text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <Settings className="w-4 h-4" />
              Configure API Keys
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {!hasConfiguredProviders ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">No API Keys Configured</h2>
              <p className="text-muted-foreground mb-4">
                Please configure your API keys to start chatting with AI models.
              </p>
              <button
                onClick={() => setShowApiKeyModal(true)}
                className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                Configure API Keys
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Error Display */}
            {error && (
              <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
                <div className="flex items-center gap-2 text-destructive">
                  <AlertCircle className="w-4 h-4" />
                  <span className="text-sm font-medium">{error}</span>
                </div>
              </div>
            )}

            {/* Controls */}
            <div className="space-y-4">
                             {/* Provider Selection */}
               <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                 <ProviderSelector
                   selectedProvider={selectedProvider}
                   onProviderChange={setSelectedProvider}
                   availableProviders={availableProviders}
                   configuredProviders={configuredProviders}
                   label="Select AI Provider"
                   disabled={isProcessing}
                 />
                 <div className="flex items-center justify-center">
                   <label className="flex items-center gap-2">
                     <input
                       type="checkbox"
                       checked={useComparisonMode}
                       onChange={(e) => setUseComparisonMode(e.target.checked)}
                       className="rounded border-input"
                     />
                     <span className="text-sm font-medium">Comparison Mode</span>
                   </label>
                 </div>
               </div>

              {/* Model Selection - Only show if provider is selected */}
              {selectedProvider && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ModelSelector
                    selectedModel={selectedModel1}
                    onModelChange={setSelectedModel1}
                    availableModels={availableModels}
                    label="Choose LLM Model 1"
                    disabled={isProcessing}
                  />
                  {useComparisonMode && availableModels.length > 1 && (
                    <ModelSelector
                      selectedModel={selectedModel2}
                      onModelChange={setSelectedModel2}
                      availableModels={availableModels}
                      label="Choose LLM Model 2"
                      disabled={isProcessing}
                    />
                  )}
                </div>
              )}
            </div>

            {/* Chat Containers */}
            {selectedProvider && selectedModel1 && (
              <div className="grid grid-cols-1 gap-6" style={{ 
                gridTemplateColumns: useComparisonMode && selectedModel2 ? '1fr 1fr' : '1fr' 
              }}>
                <div className="border rounded-lg bg-card h-[500px] flex flex-col">
                  <div className="border-b p-4">
                    <h3 className="font-medium">{selectedModel1}</h3>
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <ChatContainer
                      messages={chatHistory1}
                      modelName={selectedModel1}
                      isLoading={isProcessing}
                    />
                  </div>
                </div>

                {useComparisonMode && selectedModel2 && (
                  <div className="border rounded-lg bg-card h-[500px] flex flex-col">
                    <div className="border-b p-4">
                      <h3 className="font-medium">{selectedModel2}</h3>
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <ChatContainer
                        messages={chatHistory2}
                        modelName={selectedModel2}
                        isLoading={isProcessing}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* No Provider Selected State */}
            {!selectedProvider && hasConfiguredProviders && (
              <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-center">
                  <AlertCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <h2 className="text-xl font-semibold mb-2">Select a Provider</h2>
                  <p className="text-muted-foreground mb-4">
                    Please select an AI provider to start chatting.
                  </p>
                </div>
              </div>
            )}

            {/* Chat Input */}
            <ChatInput
              onSendMessage={handleSendMessage}
              onResetChat={handleResetChat}
              isLoading={isProcessing}
              placeholder={selectedProvider ? "Enter your query..." : "Select a provider to start chatting..."}
              disabled={!selectedProvider}
            />
          </div>
        )}
      </main>

      {/* API Key Modal */}
      <ApiKeyModal
        isOpen={showApiKeyModal}
        onClose={() => setShowApiKeyModal(false)}
        onSave={handleSaveApiConfig}
        initialConfig={apiConfig}
      />
    </div>
  );
}

export default App; 