import { Client } from '../../../../src/client';
import { Message, LLMConfig, AISuiteConfig } from '../types/chat';

class AISuiteService {
  private client: Client | null = null;
  private config: AISuiteConfig | null = null;

  initialize(config: AISuiteConfig) {
    this.config = config;
    this.client = new Client(config);
  }

  async queryLLM(modelConfig: LLMConfig, messages: Message[]): Promise<string> {
    if (!this.client) {
      throw new Error('AISuite client not initialized. Please check your API keys.');
    }

    try {
      const model = `${modelConfig.provider}:${modelConfig.model}`;
      const response = await this.client.chat.completions.create({
        model,
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        temperature: 0.7,
        max_tokens: 1000,
        stream: false, // Explicitly set stream to false to get ChatCompletionResponse
      });

      // Type guard to ensure we have a ChatCompletionResponse
      if ('choices' in response && Array.isArray(response.choices)) {
        return response.choices[0].message.content || 'No response from model';
      } else {
        throw new Error('Unexpected response format from model');
      }
    } catch (error) {
      console.error(`Error querying ${modelConfig.name}:`, error);
      throw new Error(`Error with ${modelConfig.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  getAvailableProviders(): string[] {
    if (!this.client) {
      return [];
    }
    return this.client.listProviders();
  }

  isProviderConfigured(provider: string): boolean {
    if (!this.client) {
      return false;
    }
    return this.client.isProviderConfigured(provider);
  }

  getConfig(): AISuiteConfig | null {
    return this.config;
  }
}

export const aiSuiteService = new AISuiteService(); 