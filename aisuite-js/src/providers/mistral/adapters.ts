import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatMessage,
} from "../../types";
import type {
  ChatCompletionResponse as MistralResponse,
  ChatCompletionResponseChunk as MistralStreamResponse,
} from "@mistralai/mistralai";

export function adaptRequest(request: ChatCompletionRequest): any {
  // Transform the request into Mistral's format
  const tools =
    Array.isArray(request.tools) && request.tools.length > 0
      ? request.tools
      : undefined;

  return {
    model: request.model,
    messages: request.messages.map(adaptMessage),
    tools,
    temperature: request.temperature,
    max_tokens: request.max_tokens,
    top_p: request.top_p,
    stream: request.stream,
  };
}

function adaptMessage(message: ChatMessage): any {
  return {
    role: message.role,
    content: message.content,
    tool_calls: message.tool_calls,
  };
}

export function adaptResponse(
  response: MistralResponse
): ChatCompletionResponse {
  return {
    id: response.id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: response.choices[0].message.content,
        },
        finish_reason: response.choices[0].finish_reason,
      },
    ],
    usage: {
      prompt_tokens: response.usage.prompt_tokens,
      completion_tokens: response.usage.completion_tokens,
      total_tokens: response.usage.total_tokens,
    },
  };
}

export function adaptStreamResponse(
  response: MistralStreamResponse,
  streamId: string
): ChatCompletionChunk {
  return {
    id: streamId,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          content: response.choices[0].delta.content,
          tool_calls: response.choices[0].delta.tool_calls,
        },
        finish_reason: response.choices[0].finish_reason,
      },
    ],
  };
}
