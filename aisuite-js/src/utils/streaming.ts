import { ChatCompletionChunk } from '../types';

export function createChunk(
  id: string,
  model: string,
  content?: string,
  finishReason?: string,
  toolCalls?: any[]
): ChatCompletionChunk {
  return {
    id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      delta: {
        role: 'assistant',
        content,
        tool_calls: toolCalls
      },
      finish_reason: finishReason || undefined
    }]
  };
}

export function generateId(): string {
  return `chatcmpl-${Math.random().toString(36).substr(2, 9)}`;
}