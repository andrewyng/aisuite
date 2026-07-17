import React from 'react';

interface ProviderSelectorProps {
  selectedProvider: string;
  onProviderChange: (provider: string) => void;
  availableProviders: string[];
  configuredProviders: string[];
  label: string;
  disabled?: boolean;
}

export const ProviderSelector: React.FC<ProviderSelectorProps> = ({
  selectedProvider,
  onProviderChange,
  availableProviders,
  configuredProviders,
  label,
  disabled = false
}) => {
  const providerNames = {
    openai: 'OpenAI',
    anthropic: 'Anthropic',
    groq: 'Groq',
    mistral: 'Mistral'
  };

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-foreground">
        {label}
      </label>
      <select
        value={selectedProvider}
        onChange={(e) => onProviderChange(e.target.value)}
        disabled={disabled}
        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <option value="">Select a provider</option>
        {availableProviders.map((provider) => {
          const isConfigured = configuredProviders.includes(provider);
          const displayName = providerNames[provider as keyof typeof providerNames] || provider;
          return (
            <option key={provider} value={provider}>
              {displayName} {!isConfigured ? '(API key needed)' : ''}
            </option>
          );
        })}
      </select>
    </div>
  );
}; 