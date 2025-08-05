import Groq from "groq-sdk";
import { BaseProvider } from "../../core/base-provider";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  RequestOptions,
} from "../../types";
import { GroqConfig } from "./types";
import { adaptRequest, adaptResponse, adaptStreamResponse } from "./adapters";
import { AISuiteError } from "../../core/errors";
import { generateId } from "../../utils/streaming";

export class GroqProvider extends BaseProvider {
  public readonly name = "groq";
  private client: Groq;

  constructor(config: GroqConfig) {
    super();

    this.client = new Groq({
      apiKey: config.apiKey,
      dangerouslyAllowBrowser: config.dangerouslyAllowBrowser || false, // Allow browser usage for chat app
    });
    if (config.baseURL) {
      (this.client as any).baseURL = config.baseURL;
    }
  }

  async chatCompletion(
    request: ChatCompletionRequest,
    options?: RequestOptions
  ): Promise<ChatCompletionResponse> {
    try {
      if (request.stream) {
        throw new AISuiteError(
          "Streaming is not supported in non-streaming method. Set stream: false or use streamChatCompletion method.",
          this.name,
          "STREAMING_NOT_SUPPORTED"
        );
      }

      const groqRequest = adaptRequest(request);
      const completion = await this.client.chat.completions.create(groqRequest);

      return adaptResponse(completion);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `Groq API error: ${
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
      const groqRequest = adaptRequest(request);
      const stream = await this.client.chat.completions.create(groqRequest);
      const streamId = generateId();

      // Handle abort signal
      if (options?.signal) {
        options.signal.addEventListener("abort", () => {
          if (
            stream &&
            typeof (stream as any).controller?.abort === "function"
          ) {
            (stream as any).controller.abort();
          }
        });
      }

      for await (const chunk of stream as any) {
        yield adaptStreamResponse(chunk, streamId);
      }
    } catch (error) {
      throw new AISuiteError(
        `Groq streaming error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        this.name,
        "STREAMING_ERROR"
      );
    }
  }
}
