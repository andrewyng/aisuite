import React, { useState } from 'react';
import { X, Eye, EyeOff } from 'lucide-react';
import { AISuiteConfig } from '../types/chat';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (config: AISuiteConfig) => void;
  initialConfig?: AISuiteConfig;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({
  isOpen,
  onClose,
  onSave,
  initialConfig = {}
}) => {
  const [config, setConfig] = useState<AISuiteConfig>(initialConfig);
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});

  const toggleKeyVisibility = (provider: string) => {
    setShowKeys(prev => ({
      ...prev,
      [provider]: !prev[provider]
    }));
  };

  const handleSave = () => {
    // Filter out empty API keys
    const filteredConfig: AISuiteConfig = {};
    Object.entries(config).forEach(([provider, providerConfig]) => {
      if (providerConfig?.apiKey?.trim()) {
        providerConfig.dangerouslyAllowBrowser = true;
        filteredConfig[provider as keyof AISuiteConfig] = providerConfig;
      }
    });
    onSave(filteredConfig);
    onClose();
  };

  const updateConfig = (provider: string, field: string, value: string) => {
    setConfig(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider as keyof AISuiteConfig],
        [field]: value
      }
    }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Configure API Keys</h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          {/* OpenAI */}
          <div className="space-y-2">
            <label className="text-sm font-medium">OpenAI</label>
            <div className="relative">
              <input
                type={showKeys.openai ? 'text' : 'password'}
                placeholder="sk-..."
                value={config.openai?.apiKey || ''}
                onChange={(e) => updateConfig('openai', 'apiKey', e.target.value)}
                className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              />
              <button
                type="button"
                onClick={() => toggleKeyVisibility('openai')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showKeys.openai ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Anthropic */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Anthropic</label>
            <div className="relative">
              <input
                type={showKeys.anthropic ? 'text' : 'password'}
                placeholder="sk-ant-..."
                value={config.anthropic?.apiKey || ''}
                onChange={(e) => updateConfig('anthropic', 'apiKey', e.target.value)}
                className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              />
              <button
                type="button"
                onClick={() => toggleKeyVisibility('anthropic')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showKeys.anthropic ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Groq */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Groq</label>
            <div className="relative">
              <input
                type={showKeys.groq ? 'text' : 'password'}
                placeholder="gsk_..."
                value={config.groq?.apiKey || ''}
                onChange={(e) => updateConfig('groq', 'apiKey', e.target.value)}
                className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              />
              <button
                type="button"
                onClick={() => toggleKeyVisibility('groq')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showKeys.groq ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Mistral */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Mistral</label>
            <div className="relative">
              <input
                type={showKeys.mistral ? 'text' : 'password'}
                placeholder="..."
                value={config.mistral?.apiKey || ''}
                onChange={(e) => updateConfig('mistral', 'apiKey', e.target.value)}
                className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              />
              <button
                type="button"
                onClick={() => toggleKeyVisibility('mistral')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showKeys.mistral ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>

        <div className="mt-6 flex gap-2">
          <button
            onClick={onClose}
            className="flex-1 rounded-lg border border-input bg-background px-3 py-2 text-sm font-medium ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex-1 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}; 