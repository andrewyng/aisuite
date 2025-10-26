import { BaseASRProvider } from "../../core/base-asr-provider";
import {
  TranscriptionRequest,
  TranscriptionResult,
  RequestOptions,
} from "../../types";
import { OpenAIASRConfig, OpenAIASRResponse } from "./types";
import { adaptRequest, adaptResponse } from "./adapters";
import { AISuiteError } from "../../core/errors";
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

  validateParams(params: TranscriptionRequest): void {
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

  async transcribe(
    request: TranscriptionRequest,
    options?: RequestOptions
  ): Promise<TranscriptionResult> {
    try {
      this.validateParams(request);

      const adaptedRequest = adaptRequest(request);
      const otherParams = this.translateParams(request);
      const response = await this.client.audio.transcriptions.create({
        ...adaptedRequest,
        response_format: "verbose_json",
        stream: false,
        ...otherParams,
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
