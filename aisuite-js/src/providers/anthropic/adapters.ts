import { 
  ChatCompletionRequest, 
  ChatCompletionResponse, 
  ChatCompletionChunk,
  ChatMessage,
  Tool,
  ToolCall
} from '../../types';
import type { 
  Message, 
  MessageCreateParams,
  MessageStreamEvent
} from '@anthropic-ai/sdk/resources/messages';
import { generateId, createChunk } from '../../utils/streaming';

export function adaptRequest(request: ChatCompletionRequest): MessageCreateParams {
  const { systemMessage, userMessages } = transformMessages(request.messages);
  
  // Don't pass stream parameter to avoid accidental streaming
  const params: MessageCreateParams = {
    model: request.model,
    max_tokens: request.max_tokens || 1024,
    messages: userMessages,
    temperature: request.temperature,
    top_p: request.top_p,
    stop_sequences: Array.isArray(request.stop) ? request.stop : request.stop ? [request.stop] : undefined,
  };

  if (systemMessage) {
    params.system = systemMessage;
  }

  if (request.tools) {
    params.tools = request.tools.map(adaptTool);
  }

  return params;
}

function transformMessages(messages: ChatMessage[]) {
  const systemMessages = messages.filter(msg => msg.role === 'system');
  const otherMessages = messages.filter(msg => msg.role !== 'system');
  
  const systemMessage = systemMessages.map(msg => msg.content).join('\n') || undefined;
  
  const userMessages = otherMessages.map(msg => {
    if (msg.role === 'tool') {
      // Transform tool response to user message with tool_result
      return {
        role: 'user' as const,
        content: [
          {
            type: 'tool_result' as const,
            tool_use_id: msg.tool_call_id!,
            content: msg.content!,
          }
        ]
      };
    }
    
    if (msg.role === 'assistant' && msg.tool_calls) {
      // Transform assistant message with tool calls
      const content: any[] = [];
      
      if (msg.content) {
        content.push({
          type: 'text',
          text: msg.content
        });
      }
      
      msg.tool_calls.forEach(toolCall => {
        content.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.function.name,
          input: JSON.parse(toolCall.function.arguments)
        });
      });
      
      return {
        role: 'assistant' as const,
        content
      };
    }
    
    return {
      role: msg.role as 'user' | 'assistant',
      content: msg.content!,
    };
  });
  
  return {
    systemMessage,
    userMessages,
  };
}

function adaptTool(tool: Tool): any {
  return {
    name: tool.function.name,
    description: tool.function.description,
    input_schema: {
      type: 'object',
      properties: tool.function.parameters.properties,
      required: tool.function.parameters.required,
    },
  };
}

export function adaptResponse(response: Message, originalModel: string): ChatCompletionResponse {
  const content = Array.isArray(response.content) 
    ? response.content.find(block => block.type === 'text')?.text || ''
    : response.content;

  const toolCalls: ToolCall[] = [];
  if (Array.isArray(response.content)) {
    response.content.forEach(block => {
      if (block.type === 'tool_use') {
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input),
          },
        });
      }
    });
  }

  return {
    id: response.id,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: originalModel,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      },
      finish_reason: response.stop_reason || 'stop',
    }],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
      total_tokens: response.usage.input_tokens + response.usage.output_tokens,
    },
  };
}

export function adaptStreamEvent(
  event: MessageStreamEvent, 
  streamId: string, 
  originalModel: string
): ChatCompletionChunk | null {
  switch (event.type) {
    case 'content_block_delta':
      if (event.delta.type === 'text_delta') {
        return createChunk(streamId, originalModel, event.delta.text);
      }
      break;
      
    case 'content_block_start':
      if (event.content_block.type === 'tool_use') {
        return createChunk(streamId, originalModel, undefined, undefined, [{
          id: event.content_block.id,
          type: 'function',
          function: {
            name: event.content_block.name,
            arguments: '',
          },
        }]);
      }
      break;
      
    case 'message_stop':
      return createChunk(streamId, originalModel, undefined, 'stop');
      
    default:
      return null;
  }
  
  return null;
}