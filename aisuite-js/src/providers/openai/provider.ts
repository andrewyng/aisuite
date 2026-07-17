import OpenAI from "openai";
import { Provider } from "../../core/base-provider";
import { ASRProvider } from "../../core/base-asr-provider";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  RequestOptions,
  TranscriptionRequest,
  TranscriptionResult,
} from "../../types";
import { OpenAIASRResponse, OpenAIConfig } from "./types";
import {
  adaptRequest,
  adaptResponse,
  adaptChunk,
  adaptASRRequest,
  adaptASRResponse,
} from "./adapters";
import { AISuiteError } from "../../core/errors";

export class OpenAIProvider implements Provider, ASRProvider {
  public readonly name = "openai";
  private client: OpenAI;

  constructor(config: OpenAIConfig) {
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
          "Streaming is not yet supported. Set stream: false or use streamChatCompletion method.",
          this.name,
          "STREAMING_NOT_SUPPORTED"
        );
      }

      const openaiRequest = adaptRequest(request);
      const completion = (await this.client.chat.completions.create(
        openaiRequest,
        options
      )) as any; // Type assertion needed because OpenAI SDK returns a union type

      return adaptResponse(completion);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `OpenAI API error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        this.name,
        "API_ERROR"
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
        `OpenAI streaming error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        this.name,
        "STREAMING_ERROR"
      );
    }
  }

  async transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult> {
    try {
      this.validateParams(request);

      const adaptedRequest = adaptASRRequest(request);
      const otherParams = this.translateParams(request);
      const response = await this.client.audio.transcriptions.create({
        ...adaptedRequest,
        response_format: "verbose_json",
        stream: false,
        ...otherParams,
      });

      return adaptASRResponse(response as OpenAIASRResponse);
    } catch (error: any) {
      throw new AISuiteError(
        `OpenAI ASR transcription failed: ${error.message}`,
        this.name,
        "PROVIDER_ERROR"
      );
    }
  }

  validateParams(params: { [key: string]: any }): void {
    if (!params.model) {
      throw new AISuiteError(
        "Model parameter is required",
        this.name,
        "MODEL_PARAMETER_REQUIRED"
      );
    }

    if (!params.file) {
      throw new AISuiteError(
        "File parameter is required",
        this.name,
        "MODEL_PARAMETER_REQUIRED"
      );
    }
  }

  translateParams(params: { [key: string]: any }): { [key: string]: any } {
    const { model: _, file: __, ...rest } = params;
    return rest;
  }
}
