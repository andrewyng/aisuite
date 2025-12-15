import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatMessage,
  Tool,
  ToolCall,
  ToolChoice,
  RequestOptions
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

export function adaptRequest(request: ChatCompletionRequest, options?: RequestOptions): GeminiRequest {
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

  // Add Gemini 3 thinking/reasoning configuration from RequestOptions
  if (options?.thinking_level) {
    (config as any).thinkingConfig = {
      thinkingLevel: options.thinking_level.toUpperCase() // 'low' -> 'LOW', 'high' -> 'HIGH'
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

      const content: Content = { role: 'model', parts };
      // Include thought signature from provider_data for Gemini 3 multi-turn
      if (msg.provider_data?.thought_signature) {
        (content as any).thoughtSignature = msg.provider_data.thought_signature;
      }
      contents.push(content);
    } else {
      // Regular user or assistant message
      const content: Content = {
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content || '' }],
      };
      // Include thought signature from provider_data for Gemini 3 multi-turn
      if (msg.role === 'assistant' && msg.provider_data?.thought_signature) {
        (content as any).thoughtSignature = msg.provider_data.thought_signature;
      }
      contents.push(content);
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
  // Extract function calls FIRST to avoid triggering SDK warning when accessing .text
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

  // Extract text content - only use .text accessor if no function calls
  // (accessing .text when there are function calls triggers a SDK warning)
  let textContent = '';
  if (functionCalls.length === 0) {
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
  } else {
    // When there are function calls, extract text from candidates directly
    if (response.candidates?.[0]?.content?.parts) {
      const textPart = response.candidates[0].content.parts.find(
        (p: any) => p.text !== undefined
      );
      textContent = textPart?.text || '';
    }
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

  // Extract Gemini 3 thinking content and thought signature for provider_data
  const providerData: Record<string, any> = {};

  // Extract thinking content from response parts (Gemini 3 thinking feature)
  if (response.candidates?.[0]?.content?.parts) {
    const thinkingParts = response.candidates[0].content.parts.filter(
      (p: any) => p.thought !== undefined
    );
    if (thinkingParts.length > 0) {
      providerData.thinking = thinkingParts.map((p: any) => p.thought).join('');
    }
  }

  // Extract thought signature for multi-turn conversations (Gemini 3)
  // Thought signature can be at candidate level OR inside parts
  if (response.candidates?.[0]?.thoughtSignature) {
    providerData.thought_signature = response.candidates[0].thoughtSignature;
  } else if (response.candidates?.[0]?.content?.parts) {
    // Check for thoughtSignature inside parts (Gemini 3 format)
    const partWithSignature = response.candidates[0].content.parts.find(
      (p: any) => p.thoughtSignature !== undefined
    );
    if (partWithSignature?.thoughtSignature) {
      providerData.thought_signature = partWithSignature.thoughtSignature;
    }
  }

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
    provider_data: Object.keys(providerData).length > 0 ? providerData : undefined,
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

  // Handle Gemini 3 thinking content in stream
  let thinkingContent: string | undefined;
  if (chunk.candidates?.[0]?.content?.parts) {
    const thinkingPart = chunk.candidates[0].content.parts.find(
      (p: any) => p.thought !== undefined
    );
    thinkingContent = thinkingPart?.thought;
  }

  // Extract thought signature if present (typically in final chunks)
  // Thought signature can be at candidate level OR inside parts
  let thoughtSignature: string | undefined;
  if (chunk.candidates?.[0]?.thoughtSignature) {
    thoughtSignature = chunk.candidates[0].thoughtSignature;
  } else if (chunk.candidates?.[0]?.content?.parts) {
    const partWithSignature = chunk.candidates[0].content.parts.find(
      (p: any) => p.thoughtSignature !== undefined
    );
    if (partWithSignature?.thoughtSignature) {
      thoughtSignature = partWithSignature.thoughtSignature;
    }
  }

  // Build provider_data if we have thinking content or thought signature
  const providerData: Record<string, any> = {};
  if (thinkingContent) {
    providerData.thinking_delta = thinkingContent;
  }
  if (thoughtSignature) {
    providerData.thought_signature = thoughtSignature;
  }

  // Handle function calls in stream (check first to include provider_data)
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
    const result = createChunk(streamId, originalModel, undefined, undefined, toolCalls);
    if (Object.keys(providerData).length > 0) {
      result.provider_data = providerData;
    }
    return result;
  }

  if (textContent) {
    const result = createChunk(streamId, originalModel, textContent);
    if (Object.keys(providerData).length > 0) {
      result.provider_data = providerData;
    }
    return result;
  }

  // Handle thinking content without text (stream thinking separately)
  if (thinkingContent || thoughtSignature) {
    const result = createChunk(streamId, originalModel, undefined);
    result.provider_data = providerData;
    return result;
  }

  return null;
}
