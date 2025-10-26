import { createClient, DeepgramClient } from "@deepgram/sdk";
import { BaseASRProvider } from "../../core/base-asr-provider";
import {
  TranscriptionRequest,
  TranscriptionResult,
  RequestOptions,
} from "../../types";
import { DeepgramConfig } from "./types";
import { adaptResponse } from "./adapters";
import { AISuiteError } from "../../core/errors";
import * as fs from "fs";

export class DeepgramASRProvider extends BaseASRProvider {
  public readonly name = "deepgram";
  private client: DeepgramClient;

  constructor(config: DeepgramConfig) {
    super();

    // Use the new createClient API instead of the deprecated Deepgram constructor
    this.client = createClient({
      key: config.apiKey,
      ...(config.baseURL && { baseUrl: config.baseURL }),
    });
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
    return Object.entries(rest).reduce((translated, [key, value]) => {
      switch (key) {
        case "timestamps":
          if (value) translated.utterances = true;
          break;
        default:
          translated[key] = value;
      }
      return translated;
    }, {} as { [key: string]: any });
  }

  async transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult> {
    try {
      // Extract parameters excluding model and file
      const { model, file, ...params } = request;
      this.validateParams(request);
      const translatedParams = this.translateParams(params);

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

      // Set up transcription options for v4 SDK format
      const transcriptionOptions = {
        model: request.model,
        ...translatedParams
      };

      // Use the v4 SDK format for transcription
      const response = await this.client.listen.prerecorded
        .transcribeFile(audioData, {
          ...transcriptionOptions
        });

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
