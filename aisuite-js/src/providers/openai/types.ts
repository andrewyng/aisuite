import { OpenAIConfig } from '../../types';

export { OpenAIConfig };

// Re-export OpenAI types that we need
export type { 
  ChatCompletion,
  ChatCompletionChunk as OpenAIChunk,
  ChatCompletionCreateParams
} from 'openai/resources/chat/completions';