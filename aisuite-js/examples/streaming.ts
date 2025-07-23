import 'dotenv/config';
import { Client, ChatCompletionChunk } from '../src';

async function main() {
  const client = new Client({
    openai: { apiKey: process.env.OPENAI_API_KEY! },
    anthropic: { apiKey: process.env.ANTHROPIC_API_KEY! },
  });

  console.log('🚀 AISuite Streaming Examples\n');

  // Example 1: Basic OpenAI Streaming
  console.log('--- OpenAI Streaming ---');
  try {
    const stream = await client.chat.completions.create({
      model: 'openai:gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Write a haiku about TypeScript' }
      ],
      stream: true,
      temperature: 0.7,
      max_tokens: 100,
    }) as AsyncIterable<ChatCompletionChunk>;

    console.log('Response: ');
    let fullContent = '';
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      process.stdout.write(content);
      fullContent += content;
    }
    console.log('\n\nFull response:', fullContent);
  } catch (error) {
    console.error('OpenAI streaming error:', error);
  }

  // Example 2: Basic Anthropic Streaming
  console.log('\n--- Anthropic Streaming ---');
  try {
    const stream = await client.chat.completions.create({
      model: 'anthropic:claude-3-haiku-20240307',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Write a haiku about JavaScript' }
      ],
      stream: true,
      temperature: 0.7,
      max_tokens: 100,
    }) as AsyncIterable<ChatCompletionChunk>;

    console.log('Response: ');
    let fullContent = '';
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      process.stdout.write(content);
      fullContent += content;
    }
    console.log('\n\nFull response:', fullContent);
  } catch (error) {
    console.error('Anthropic streaming error:', error);
  }

  // Example 3: Streaming with Progress Indicator
  console.log('\n--- Streaming with Progress ---');
  try {
    const stream = await client.chat.completions.create({
      model: 'openai:gpt-3.5-turbo',
      messages: [
        { role: 'user', content: 'Count from 1 to 10 slowly' }
      ],
      stream: true,
      temperature: 0,
      max_tokens: 100,
    }) as AsyncIterable<ChatCompletionChunk>;

    console.log('Response: ');
    let charCount = 0;
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      process.stdout.write(content);
      charCount += content.length;
      
      // Show progress in title (if supported)
      if (process.stdout.isTTY) {
        process.stdout.write(`\x1b]0;Streaming: ${charCount} chars\x07`);
      }
    }
    console.log(`\n\nTotal characters: ${charCount}`);
  } catch (error) {
    console.error('Streaming error:', error);
  }

  // Example 4: Abort Controller
  console.log('\n--- Streaming with Abort Controller ---');
  try {
    const controller = new AbortController();
    
    // Abort after 2 seconds
    const timeout = setTimeout(() => {
      console.log('\n\n⏹️  Aborting stream...');
      controller.abort();
    }, 2000);

    const stream = await client.chat.completions.create({
      model: 'anthropic:claude-3-haiku-20240307',
      messages: [
        { role: 'user', content: 'Tell me a very long story about a programmer' }
      ],
      stream: true,
      temperature: 0.7,
      max_tokens: 500,
    }, { signal: controller.signal }) as AsyncIterable<ChatCompletionChunk>;

    console.log('Response (will abort after 2 seconds): ');
    try {
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        process.stdout.write(content);
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('\n\n✅ Stream successfully aborted');
      } else {
        throw error;
      }
    } finally {
      clearTimeout(timeout);
    }
  } catch (error) {
    console.error('Abort controller error:', error);
  }

  // Example 5: Streaming with Tool Calls
  console.log('\n--- Streaming with Tool Calls ---');
  try {
    const tools = [{
      type: 'function' as const,
      function: {
        name: 'get_weather',
        description: 'Get current weather',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string' }
          },
          required: ['location']
        }
      }
    }];

    const stream = await client.chat.completions.create({
      model: 'openai:gpt-4o-mini',
      messages: [
        { role: 'user', content: 'What\'s the weather in Tokyo?' }
      ],
      tools,
      tool_choice: 'auto',
      stream: true,
    }) as AsyncIterable<ChatCompletionChunk>;

    console.log('Streaming response with potential tool calls:\n');
    
    let currentToolCall: any = null;
    let toolCalls: any[] = [];
    
    for await (const chunk of stream) {
      // Handle text content
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        process.stdout.write(content);
      }
      
      // Handle tool calls
      const deltaToolCalls = chunk.choices[0]?.delta?.tool_calls;
      if (deltaToolCalls) {
        for (const toolCall of deltaToolCalls) {
          if (toolCall.id) {
            // New tool call
            currentToolCall = {
              id: toolCall.id,
              type: toolCall.type,
              function: {
                name: toolCall.function?.name || '',
                arguments: toolCall.function?.arguments || ''
              }
            };
            toolCalls.push(currentToolCall);
          } else if (currentToolCall && toolCall.function?.arguments) {
            // Accumulate arguments
            currentToolCall.function.arguments += toolCall.function.arguments;
          }
        }
      }
      
      // Check if we're done
      if (chunk.choices[0]?.finish_reason === 'tool_calls') {
        console.log('\n\nTool calls detected:');
        for (const tc of toolCalls) {
          console.log(`- ${tc.function.name}: ${tc.function.arguments}`);
        }
      }
    }
  } catch (error) {
    console.error('Streaming with tools error:', error);
  }

  console.log('\n\n✨ Streaming examples completed!');
}

// Run examples
main().catch(console.error);