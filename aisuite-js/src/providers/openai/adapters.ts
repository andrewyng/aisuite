import { 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ChatCompletionChunk,
  ChatMessage,
  ToolCall,
  Tool
} from '../../types';
import type { 
  ChatCompletion, 
  ChatCompletionChunk as OpenAIChunk,
  ChatCompletionCreateParams
} from 'openai/resources/chat/completions';

export function adaptRequest(request: ChatCompletionRequest): ChatCompletionCreateParams {
  // OpenAI is our base format, so minimal transformation needed
  // Don't pass stream parameter to avoid accidental streaming
  const { stream, ...requestWithoutStream } = request;
  
  return {
    model: requestWithoutStream.model,
    messages: requestWithoutStream.messages.map(adaptMessage),
    tools: requestWithoutStream.tools,
    tool_choice: requestWithoutStream.tool_choice,
    temperature: requestWithoutStream.temperature,
    max_tokens: requestWithoutStream.max_tokens,
    top_p: requestWithoutStream.top_p,
    frequency_penalty: requestWithoutStream.frequency_penalty,
    presence_penalty: requestWithoutStream.presence_penalty,
    stop: requestWithoutStream.stop,
    user: requestWithoutStream.user,
  };
}

function adaptMessage(message: ChatMessage): any {
  return {
    role: message.role,
    content: message.content,
    name: message.name,
    tool_call_id: message.tool_call_id,
    tool_calls: message.tool_calls,
  };
}

export function adaptResponse(response: ChatCompletion): ChatCompletionResponse {
  return {
    id: response.id,
    object: 'chat.completion',
    created: response.created,
    model: response.model,
    choices: response.choices.map(choice => ({
      index: choice.index,
      message: {
        role: choice.message.role as any,
        content: choice.message.content,
        tool_calls: choice.message.tool_calls?.map(adaptToolCall),
      },
      finish_reason: choice.finish_reason || 'stop',
    })),
    usage: {
      prompt_tokens: response.usage?.prompt_tokens || 0,
      completion_tokens: response.usage?.completion_tokens || 0,
      total_tokens: response.usage?.total_tokens || 0,
    },
    system_fingerprint: response.system_fingerprint,
  };
}

export function adaptChunk(chunk: OpenAIChunk): ChatCompletionChunk {
  return {
    id: chunk.id,
    object: 'chat.completion.chunk',
    created: chunk.created,
    model: chunk.model,
    choices: chunk.choices.map(choice => ({
      index: choice.index,
      delta: {
        role: choice.delta.role as any,
        content: choice.delta.content || undefined,
        tool_calls: choice.delta.tool_calls?.map(adaptToolCall),
      },
      finish_reason: choice.finish_reason || undefined,
    })),
    usage: chunk.usage ? {
      prompt_tokens: chunk.usage.prompt_tokens || 0,
      completion_tokens: chunk.usage.completion_tokens || 0,
      total_tokens: chunk.usage.total_tokens || 0,
    } : undefined,
  };
}

function adaptToolCall(toolCall: any): ToolCall {
  return {
    id: toolCall.id,
    type: 'function',
    function: {
      name: toolCall.function.name,
      arguments: toolCall.function.arguments,
    },
  };
}