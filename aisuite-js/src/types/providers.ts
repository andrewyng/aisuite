export interface ProviderConfigs {
  openai?: OpenAIConfig;
  anthropic?: AnthropicConfig;
  mistral?: MistralConfig;
  groq?: GroqConfig;
  deepgram?: DeepgramConfig;
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

export interface MistralConfig {
  apiKey: string;
  baseURL?: string;
}

export interface GroqConfig {
  apiKey: string;
  baseURL?: string;
}

export interface DeepgramConfig {
  apiKey: string;
  baseURL?: string;
}
