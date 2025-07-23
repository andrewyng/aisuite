export interface ProviderConfigs {
  openai?: OpenAIConfig;
  anthropic?: AnthropicConfig;
}

export interface OpenAIConfig {
  apiKey: string;
  baseURL?: string;
  organization?: string;
}

export interface AnthropicConfig {
  apiKey: string;
  baseURL?: string;
}