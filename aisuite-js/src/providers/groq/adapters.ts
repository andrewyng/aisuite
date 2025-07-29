import type { ChatCompletion as GroqChatCompletion } from "groq-sdk/resources/chat/completions";
import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk as AISuiteChatCompletionChunk,
  Usage,
} from "../../types";

export function adaptRequest(request: ChatCompletionRequest): any {
  return {
    model: request.model.replace("groq:", ""),
    messages: request.messages,
    temperature: request.temperature,
    max_tokens: request.max_tokens,
    stream: request.stream,
    tools: request.tools,
    tool_choice: request.tool_choice,
  };
}

export function adaptResponse(
  response: GroqChatCompletion
): ChatCompletionResponse {
  return {
    id: response.id,
    object: response.object,
    created: response.created,
    model: `groq:${response.model}`,
    choices: response.choices.map((choice) => ({
      index: choice.index,
      message: choice.message,
      finish_reason: choice.finish_reason,
    })),
    usage: response.usage ?? {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}

export function adaptStreamResponse(
  chunk: any,
  streamId: string
): AISuiteChatCompletionChunk {
  return {
    id: streamId,
    object: "chat.completion.chunk",
    created: Date.now(),
    model: `groq:${chunk.model}`,
    choices: chunk.choices.map((choice: any) => ({
      index: choice.index,
      delta: choice.delta,
      finish_reason: choice.finish_reason,
    })),
  };
}
