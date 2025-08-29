import { BaseASRProvider } from "../../../core/base-asr-provider";
import {
  TranscriptionRequest,
  TranscriptionResult,
  RequestOptions,
} from "../../../types";
import { OpenAIASRConfig, OpenAIASRResponse } from "./types";
import { adaptRequest, adaptResponse } from "./adapters";
import { AISuiteError } from "../../../core/errors";
import OpenAI from "openai";


export class OpenAIASRProvider extends BaseASRProvider {
  public readonly name = "openai";
  private client: OpenAI;

  constructor(config: OpenAIASRConfig) {
    super();
    this.client = new OpenAI({
      apiKey: config.apiKey,
      organization: config.organization,
      baseURL: config.baseURL,
    });
  }

  validateParams(model: string, params: { [key: string]: any }): void {
    const supported = new Set([
      "language",
      "prompt",
      "response_format",
      "temperature",
      "timestamps",
    ]);

    for (const [key, value] of Object.entries(params)) {
      if (!supported.has(key) && !key.startsWith("openai_")) {
        console.warn(`Parameter '${key}' may not be supported by OpenAI ASR`);
      }
    }
  }

  translateParams(
    model: string,
    params: { [key: string]: any }
  ): { [key: string]: any } {
    const adaptedParams: { [key: string]: any } = {};

    // Map standard parameters to OpenAI-specific parameters
    if (params.language) adaptedParams.language = params.language;
    if (params.prompt) adaptedParams.prompt = params.prompt;
    if (params.response_format)
      adaptedParams.response_format = params.response_format;
    if (params.temperature) adaptedParams.temperature = params.temperature;
    if (params.timestamps) adaptedParams.timestamp_granularities = ["word"];

    // Include any OpenAI-specific parameters (prefixed with openai_)
    for (const [key, value] of Object.entries(params)) {
      if (key.startsWith("openai_")) {
        adaptedParams[key.replace("openai_", "")] = value;
      }
    }

    return adaptedParams;
  }

  async transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult> {
    try {
      if (!request.file || !(request.file instanceof Buffer)) {
        throw new AISuiteError(
          "File must be provided as a Buffer",
          "VALIDATION_ERROR"
        );
      }

      const adaptedRequest = adaptRequest(request);
      const response = await this.client.audio.transcriptions.create({
        ...adaptedRequest,
        response_format: "verbose_json",
      });

      return adaptResponse(response as OpenAIASRResponse);
    } catch (error: any) {
      throw new AISuiteError(
        `OpenAI ASR transcription failed: ${error.message}`,
        "PROVIDER_ERROR"
      );
    }
  }
}
