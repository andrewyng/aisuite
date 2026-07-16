export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

export interface ChatHistory {
  id: string;
  messages: Message[];
  modelName: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface LLMConfig {
  name: string;
  provider: string;
  model: string;
}

export interface ChatState {
  chatHistory1: Message[];
  chatHistory2: Message[];
  isProcessing: boolean;
  useComparisonMode: boolean;
  selectedModel1: string;
  selectedModel2: string;
}

export interface AISuiteConfig {
  openai?: {
    apiKey: string;
    baseURL?: string;
    organization?: string;
  };
  anthropic?: {
    apiKey: string;
    baseURL?: string;
  };
  groq?: {
    apiKey: string;
    baseURL?: string;
    dangerouslyAllowBrowser?: boolean;
  };
  mistral?: {
    apiKey: string;
    baseURL?: string;
  };
} 