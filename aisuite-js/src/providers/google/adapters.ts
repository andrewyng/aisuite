import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatMessage,
  Tool,
  ToolCall,
  ToolChoice
} from '../../types';
import { generateId, createChunk } from '../../utils/streaming';
import type {
  Content,
  Part,
  GenerateContentConfig,
  FunctionDeclaration,
  FunctionCallingConfigMode
} from '@google/genai';

export interface GeminiRequest {
  model: string;
  contents: Content[];
  config?: GenerateContentConfig;
}

export function adaptRequest(request: ChatCompletionRequest): GeminiRequest {
  const { systemInstruction, contents } = transformMessages(request.messages);

  const config: GenerateContentConfig = {};

  // Add generation parameters directly to config (not wrapped in generationConfig)
  if (request.temperature !== undefined) {
    config.temperature = request.temperature;
  }
  if (request.top_p !== undefined) {
    config.topP = request.top_p;
  }
  if (request.max_tokens !== undefined) {
    config.maxOutputTokens = request.max_tokens;
  }
  if (request.stop !== undefined) {
    config.stopSequences = Array.isArray(request.stop) ? request.stop : [request.stop];
  }

  if (systemInstruction) {
    config.systemInstruction = systemInstruction;
  }

  if (request.tools && request.tools.length > 0) {
    config.tools = [{
      functionDeclarations: request.tools.map(adaptTool)
    }];
  }

  if (request.tool_choice) {
    config.toolConfig = {
      functionCallingConfig: adaptToolChoice(request.tool_choice),
    };
  }

  return {
    model: request.model,
    contents,
    config: Object.keys(config).length > 0 ? config : undefined,
  };
}

function transformMessages(messages: ChatMessage[]): {
  systemInstruction?: string;
  contents: Content[]
} {
  const systemMessages = messages.filter(msg => msg.role === 'system');
  const otherMessages = messages.filter(msg => msg.role !== 'system');

  const systemInstruction = systemMessages
    .map(msg => msg.content)
    .filter(Boolean)
    .join('\n') || undefined;

  const contents: Content[] = [];

  for (const msg of otherMessages) {
    if (msg.role === 'tool') {
      // Tool response becomes a user message with functionResponse part
      contents.push({
        role: 'user',
        parts: [{
          functionResponse: {
            name: msg.name || 'unknown',
            response: { result: msg.content },
          },
        }],
      });
    } else if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
      // Assistant message with tool calls
      const parts: Part[] = [];

      if (msg.content) {
        parts.push({ text: msg.content });
      }

      for (const toolCall of msg.tool_calls) {
        let args: Record<string, unknown> = {};
        try {
          args = JSON.parse(toolCall.function.arguments);
        } catch {
          // If parsing fails, use empty object
        }
        parts.push({
          functionCall: {
            name: toolCall.function.name,
            args,
          },
        });
      }

      contents.push({ role: 'model', parts });
    } else {
      // Regular user or assistant message
      contents.push({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content || '' }],
      });
    }
  }

  return { systemInstruction, contents };
}

function adaptTool(tool: Tool): FunctionDeclaration {
  return {
    name: tool.function.name,
    description: tool.function.description,
    // Use parametersJsonSchema which accepts raw JSON schema
    // This avoids needing to convert to the SDK's Schema type
    parametersJsonSchema: tool.function.parameters,
  };
}

function adaptToolChoice(toolChoice: ToolChoice): {
  mode?: FunctionCallingConfigMode;
  allowedFunctionNames?: string[]
} {
  if (toolChoice === 'auto') {
    return { mode: 'AUTO' as FunctionCallingConfigMode };
  } else if (toolChoice === 'none') {
    return { mode: 'NONE' as FunctionCallingConfigMode };
  } else if (typeof toolChoice === 'object' && toolChoice.function) {
    return {
      mode: 'ANY' as FunctionCallingConfigMode,
      allowedFunctionNames: [toolChoice.function.name],
    };
  }
  return { mode: 'AUTO' as FunctionCallingConfigMode };
}

export function adaptResponse(
  response: any,
  originalModel: string
): ChatCompletionResponse {
  // Extract text content
  let textContent = '';
  try {
    textContent = response.text || '';
  } catch {
    // If .text accessor fails, try to extract from candidates
    if (response.candidates?.[0]?.content?.parts) {
      const textPart = response.candidates[0].content.parts.find(
        (p: any) => p.text !== undefined
      );
      textContent = textPart?.text || '';
    }
  }

  // Extract function calls
  const toolCalls: ToolCall[] = [];
  const functionCalls = response.functionCalls || [];

  for (const fc of functionCalls) {
    toolCalls.push({
      id: generateId(),
      type: 'function',
      function: {
        name: fc.name,
        arguments: JSON.stringify(fc.args || {}),
      },
    });
  }

  // Determine finish reason
  let finishReason = 'stop';
  if (toolCalls.length > 0) {
    finishReason = 'tool_calls';
  } else if (response.candidates?.[0]?.finishReason) {
    const geminiReason = response.candidates[0].finishReason;
    // Map Gemini finish reasons to OpenAI format
    if (geminiReason === 'STOP') {
      finishReason = 'stop';
    } else if (geminiReason === 'MAX_TOKENS') {
      finishReason = 'length';
    } else if (geminiReason === 'SAFETY') {
      finishReason = 'content_filter';
    }
  }

  // Extract usage metadata
  const usageMetadata = response.usageMetadata || {};

  return {
    id: generateId(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: originalModel,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: textContent,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      },
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: usageMetadata.promptTokenCount || 0,
      completion_tokens: usageMetadata.candidatesTokenCount || 0,
      total_tokens: usageMetadata.totalTokenCount || 0,
    },
  };
}

export function adaptStreamEvent(
  chunk: any,
  streamId: string,
  originalModel: string
): ChatCompletionChunk | null {
  // Handle text content in stream
  let textContent: string | undefined;
  try {
    textContent = chunk.text;
  } catch {
    // Try to extract from candidates
    if (chunk.candidates?.[0]?.content?.parts) {
      const textPart = chunk.candidates[0].content.parts.find(
        (p: any) => p.text !== undefined
      );
      textContent = textPart?.text;
    }
  }

  if (textContent) {
    return createChunk(streamId, originalModel, textContent);
  }

  // Handle function calls in stream
  const functionCalls = chunk.functionCalls || [];
  if (functionCalls.length > 0) {
    const toolCalls = functionCalls.map((fc: any) => ({
      id: generateId(),
      type: 'function' as const,
      function: {
        name: fc.name,
        arguments: JSON.stringify(fc.args || {}),
      },
    }));
    return createChunk(streamId, originalModel, undefined, undefined, toolCalls);
  }

  return null;
}
