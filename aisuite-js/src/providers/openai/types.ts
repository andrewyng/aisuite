import OpenAI from "openai";
import { OpenAIConfig } from "../../types";

export { OpenAIConfig };

// Re-export OpenAI types that we need
export type {
  ChatCompletion,
  ChatCompletionChunk as OpenAIChunk,
  ChatCompletionCreateParams,
} from "openai/resources/chat/completions";

export interface OpenAIASRRequest {
  file: File;
  model: string;
  language?: string;
  prompt?: string;
  response_format?: "json" | "text" | "srt" | "verbose_json" | "vtt";
  temperature?: number;
  timestamp_granularities?: Array<"word" | "segment">;
}

export interface OpenAIASRResponse extends OpenAI.Audio.Transcription {
  text: string;
  language?: string;
  duration?: number;
  segments?: Array<{
    id: number;
    seek: number;
    start: number;
    end: number;
    text: string;
    tokens: number[];
    temperature: number;
    avg_logprob: number;
    compression_ratio: number;
    no_speech_prob: number;
  }>;
  words?: Array<{
    text: string;
    start: number;
    end: number;
    confidence?: number;
  }>;
}
