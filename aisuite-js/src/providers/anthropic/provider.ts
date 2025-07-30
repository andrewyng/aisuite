import Anthropic from '@anthropic-ai/sdk';
import { BaseProvider } from '../../core/base-provider';
import { 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ChatCompletionChunk,
  RequestOptions 
} from '../../types';
import { AnthropicConfig } from './types';
import { adaptRequest, adaptResponse, adaptStreamEvent } from './adapters';
import { AISuiteError } from '../../core/errors';
import { generateId } from '../../utils/streaming';

export class AnthropicProvider extends BaseProvider {
  public readonly name = 'anthropic';
  private client: Anthropic;

  constructor(config: AnthropicConfig) {
    super();
    this.client = new Anthropic({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
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

      const anthropicRequest = adaptRequest(request);
      const message = await this.client.messages.create(
        anthropicRequest,
        options
      ) as any;  // Type assertion needed because Anthropic SDK returns a union type

      return adaptResponse(message, request.model);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `Anthropic API error: ${error instanceof Error ? error.message : 'Unknown error'}`,
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
      const anthropicRequest = adaptRequest(request);
      const stream = await this.client.messages.create(
        {
          ...anthropicRequest,
          stream: true,
        },
        options
      );

      const streamId = generateId();

      // Handle abort signal
      if (options?.signal) {
        options.signal.addEventListener('abort', () => {
          if (stream && typeof (stream as any).controller?.abort === 'function') {
            (stream as any).controller.abort();
          }
        });
      }

      for await (const event of stream) {
        const chunk = adaptStreamEvent(event, streamId, request.model);
        if (chunk) {
          yield chunk;
        }
      }
    } catch (error) {
      throw new AISuiteError(
        `Anthropic streaming error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        this.name,
        'STREAMING_ERROR'
      );
    }
  }
}