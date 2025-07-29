import MistralClient from "@mistralai/mistralai";
import { BaseProvider } from "../../core/base-provider";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  RequestOptions,
} from "../../types";
import { MistralConfig } from "./types";
import { adaptRequest, adaptResponse, adaptStreamResponse } from "./adapters";
import { AISuiteError } from "../../core/errors";
import { generateId } from "../../utils/streaming";

export class MistralProvider extends BaseProvider {
  public readonly name = "mistral";
  private client: MistralClient;

  constructor(config: MistralConfig) {
    super();
    this.client = new MistralClient(config.apiKey);
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

      const mistralRequest = adaptRequest(request);
      const completion = await this.client.chat(mistralRequest);

      return adaptResponse(completion);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `Mistral API error: ${
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
      const mistralRequest = adaptRequest(request);
      const stream = await this.client.chatStream(mistralRequest);
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

      for await (const chunk of stream) {
        yield adaptStreamResponse(chunk, streamId);
      }
    } catch (error) {
      throw new AISuiteError(
        `Mistral streaming error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        this.name,
        "STREAMING_ERROR"
      );
    }
  }
}
