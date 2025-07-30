import OpenAI from 'openai';
import { BaseProvider } from '../../core/base-provider';
import { 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ChatCompletionChunk,
  RequestOptions 
} from '../../types';
import { OpenAIConfig } from './types';
import { adaptRequest, adaptResponse, adaptChunk } from './adapters';
import { AISuiteError } from '../../core/errors';

export class OpenAIProvider extends BaseProvider {
  public readonly name = 'openai';
  private client: OpenAI;

  constructor(config: OpenAIConfig) {
    super();
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      organization: config.organization,
    });
  }

  async chatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): Promise<ChatCompletionResponse> {
    try {
      // For now, we don't support streaming in non-streaming method
      if (request.stream) {
        throw new AISuiteError(
          'Streaming is not yet supported. Set stream: false or use streamChatCompletion method.',
          this.name,
          'STREAMING_NOT_SUPPORTED'
        );
      }

      const openaiRequest = adaptRequest(request);
      const completion = await this.client.chat.completions.create(
        openaiRequest,
        options
      ) as any;  // Type assertion needed because OpenAI SDK returns a union type

      return adaptResponse(completion);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `OpenAI API error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        this.name,
        'API_ERROR'
      );
    }
  }

  async *streamChatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): AsyncIterable<ChatCompletionChunk> {
    try {
      const openaiRequest = adaptRequest(request);
      const stream = await this.client.chat.completions.create(
        {
          ...openaiRequest,
          stream: true,
        },
        options
      );

      for await (const chunk of stream) {
        yield adaptChunk(chunk);
      }
    } catch (error) {
      throw new AISuiteError(
        `OpenAI streaming error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        this.name,
        'STREAMING_ERROR'
      );
    }
  }
}