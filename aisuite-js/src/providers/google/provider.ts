import { GoogleGenAI } from '@google/genai';
import { Provider } from '../../core/base-provider';
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  RequestOptions
} from '../../types';
import { GoogleConfig } from './types';
import { adaptRequest, adaptResponse, adaptStreamEvent } from './adapters';
import { AISuiteError } from '../../core/errors';
import { generateId, createChunk } from '../../utils/streaming';

export class GoogleProvider implements Provider {
  public readonly name = 'google';
  private client: GoogleGenAI;

  constructor(config: GoogleConfig) {
    this.client = new GoogleGenAI({
      apiKey: config.apiKey,
    });
  }

  async chatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): Promise<ChatCompletionResponse> {
    try {
      if (request.stream) {
        throw new AISuiteError(
          'Streaming is not supported in non-streaming method. Set stream: false or use streamChatCompletion method.',
          this.name,
          'STREAMING_NOT_SUPPORTED'
        );
      }

      const geminiRequest = adaptRequest(request);
      const response = await this.client.models.generateContent(geminiRequest);

      return adaptResponse(response, request.model);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `Google Gemini API error: ${error instanceof Error ? error.message : 'Unknown error'}`,
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
      const geminiRequest = adaptRequest(request);
      const stream = await this.client.models.generateContentStream(geminiRequest);

      const streamId = generateId();

      for await (const chunk of stream) {
        const adaptedChunk = adaptStreamEvent(chunk, streamId, request.model);
        if (adaptedChunk) {
          yield adaptedChunk;
        }
      }

      // Emit final chunk with finish_reason
      yield createChunk(streamId, request.model, undefined, 'stop');
    } catch (error) {
      throw new AISuiteError(
        `Google Gemini streaming error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        this.name,
        'STREAMING_ERROR'
      );
    }
  }
}
