import { TranscriptionRequest, TranscriptionResult } from "../../types";
import { OpenAIASRResponse } from "./types";
import { Uploadable } from "openai/uploads";

export function adaptRequest(request: TranscriptionRequest): {
  file: Uploadable;
  model: string;
} {
  if (!(request.file instanceof Buffer)) {
    throw new Error("File must be provided as a Buffer");
  }

  const file = new File([request.file], "audio.mp3", {
    type: "audio/mpeg",
  }) as unknown as Uploadable;

  return {
    file,
    model: request.model,
  };
}

export function adaptResponse(
  response: OpenAIASRResponse
): TranscriptionResult {
  return {
    text: response.text,
    language: response.language || "en", // Default to English if not provided
    confidence: response.segments?.[0]?.avg_logprob,
    words:
      response.words?.map((word) => ({
        text: word.text || "",
        start: word.start,
        end: word.end,
        confidence: word?.confidence, // Default confidence if not provided
      })) ?? [],
    segments:
      response.segments?.map((segment) => ({
        text: segment.text,
        start: segment.start,
        end: segment.end,
      })) ?? [],
  };
}
