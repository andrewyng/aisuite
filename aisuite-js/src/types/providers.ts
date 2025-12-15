export interface ProviderConfigs {
  openai?: OpenAIConfig;
  anthropic?: AnthropicConfig;
  mistral?: MistralConfig;
  groq?: GroqConfig;
  deepgram?: DeepgramConfig;
  google?: GoogleConfig;
}

export interface OpenAIConfig {
  apiKey?: string;  // Optional - SDK reads from OPENAI_API_KEY env var if not provided
  baseURL?: string;
  organization?: string;
}

export interface AnthropicConfig {
  apiKey?: string;  // Optional - SDK reads from ANTHROPIC_API_KEY env var if not provided
  baseURL?: string;
}

export interface MistralConfig {
  apiKey?: string;  // Optional - SDK reads from MISTRAL_API_KEY env var if not provided
  baseURL?: string;
}

export interface GroqConfig {
  apiKey?: string;  // Optional - SDK reads from GROQ_API_KEY env var if not provided
  baseURL?: string;
}

export interface DeepgramConfig {
  apiKey?: string;  // Optional - SDK reads from DEEPGRAM_API_KEY env var if not provided
  baseURL?: string;
}

export interface GoogleConfig {
  apiKey?: string;  // Optional - SDK reads from GOOGLE_API_KEY env var if not provided
}
