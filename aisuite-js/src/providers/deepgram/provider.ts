import { createClient } from "@deepgram/sdk";
import { BaseASRProvider } from "../../core/base-asr-provider";
import {
  TranscriptionRequest,
  TranscriptionResult,
  RequestOptions,
} from "../../types";
import { DeepgramConfig } from "./types";
import { adaptRequest, adaptResponse } from "./adapters";
import { AISuiteError } from "../../core/errors";
import * as fs from "fs";

export class DeepgramASRProvider extends BaseASRProvider {
  public readonly name = "deepgram";
  private client: any;

  constructor(config: DeepgramConfig) {
    super();

    // Use the new createClient API instead of the deprecated Deepgram constructor
    this.client = createClient({
      key: config.apiKey,
      ...(config.baseURL && { baseUrl: config.baseURL }),
    });
  }

  validateParams(model: string, params: { [key: string]: any }): void {
    const supported = new Set([
      "language",
      "timestamps",
      "word_confidence",
      "speaker_labels",
      "smart_format",
      "punctuate",
      "diarize",
      "utterances",
      "temperature",
    ]);

    for (const [key, value] of Object.entries(params)) {
      if (!supported.has(key) && !key.startsWith("deepgram_")) {
        console.warn(`Parameter '${key}' may not be supported by Deepgram ASR`);
      }
    }
  }

  translateParams(
    model: string,
    params: { [key: string]: any }
  ): { [key: string]: any } {
    const translated: { [key: string]: any } = {};

    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "language":
          translated.language = value;
          break;
        case "timestamps":
          if (value) {
            translated.utterances = true;
          }
          break;
        case "word_confidence":
          if (value) {
            translated.smart_format = true;
          }
          break;
        case "speaker_labels":
          if (value) {
            translated.diarize = true;
          }
          break;
        default:
          // Pass through Deepgram-specific parameters
          if (key.startsWith("deepgram_")) {
            translated[key.substring(9)] = value;
          } else {
            translated[key] = value;
          }
      }
    }

    return translated;
  }

  async transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult> {
    try {
      // Extract parameters excluding model and file
      const { model, file, ...params } = request;
      this.validateParams(model, params);
      const translatedParams = this.translateParams(model, params);

      // Handle different input types
      let audioData: Buffer;
      if (typeof request.file === "string") {
        audioData = fs.readFileSync(request.file);
      } else if (Buffer.isBuffer(request.file)) {
        audioData = request.file;
      } else if (request.file instanceof Uint8Array) {
        audioData = Buffer.from(request.file);
      } else {
        throw new AISuiteError(
          "Unsupported audio input type",
          this.name,
          "INVALID_INPUT"
        );
      }

      // Create options object for Deepgram SDK
      const deepgramOptions: any = {
        model: request.model,
        ...translatedParams,
      };

      // Handle timeout if provided
      if (options?.timeout) {
        deepgramOptions.timeout = options.timeout;
      }

      // Deepgram SDK expects an object with buffer and mimetype for prerecorded files
      const filePayload = { buffer: audioData, mimetype: "audio/wav" };

      const response = await this.client.listen.prerecorded.transcribeFile(
        filePayload,
        deepgramOptions
      );

      // Check for errors in the response
      if (response.error) {
        throw new AISuiteError(
          `Deepgram API error: ${response.error.message}`,
          this.name,
          "API_ERROR"
        );
      }

      return adaptResponse(response.result);
    } catch (error) {
      if (error instanceof AISuiteError) {
        throw error;
      }
      throw new AISuiteError(
        `Deepgram ASR error: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
        this.name,
        "API_ERROR"
      );
    }
  }
}
