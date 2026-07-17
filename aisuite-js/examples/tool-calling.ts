import 'dotenv/config';
import { Client, ChatMessage } from '../src';

// Mock function for weather
function getWeather(location: string, unit: 'celsius' | 'fahrenheit' = 'celsius') {
  // Mock implementation
  return {
    location,
    temperature: unit === 'celsius' ? 22 : 72,
    condition: 'sunny',
    unit
  };
}

async function main() {
  const client = new Client({
    openai: { apiKey: process.env.OPENAI_API_KEY! },
    anthropic: { apiKey: process.env.ANTHROPIC_API_KEY! },
  });

  // Define tools in OpenAI format
  const tools = [
    {
      type: 'function' as const,
      function: {
        name: 'get_weather',
        description: 'Get the current weather for a location',
        parameters: {
          type: 'object',
          properties: {
            location: {
              type: 'string',
              description: 'The city and state, e.g. San Francisco, CA'
            },
            unit: {
              type: 'string',
              enum: ['celsius', 'fahrenheit'],
              description: 'The temperature unit'
            }
          },
          required: ['location']
        }
      }
    }
  ];

  // Example 1: OpenAI Tool Calling
  console.log('--- OpenAI Tool Calling ---');
  try {
    // Step 1: Initial request with tools
    const response = await client.chat.completions.create({
      model: 'openai:gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a helpful weather assistant.' },
        { role: 'user', content: "What's the weather like in San Francisco?" }
      ],
      tools,
      tool_choice: 'auto'
    });

    const message = response.choices[0]?.message;
    console.log('Step 1 - Initial response:', JSON.stringify(message, null, 2));

    if (message?.tool_calls) {
      // Step 2: Execute tool calls and send results back
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful weather assistant.' },
        { role: 'user', content: "What's the weather like in San Francisco?" },
        message // The assistant's message with tool calls
      ];

      console.log('\nTool calls detected:');
      for (const toolCall of message.tool_calls) {
        console.log(`- Function: ${toolCall.function.name}`);
        console.log(`  Arguments: ${toolCall.function.arguments}`);
        
        // Execute the function
        const args = JSON.parse(toolCall.function.arguments);
        const result = getWeather(args.location, args.unit);
        console.log(`  Result:`, result);

        // Add tool result to messages
        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(result)
        });
      }

      // Step 3: Get final response with tool results
      console.log('\nStep 2 - Sending tool results back...');
      const finalResponse = await client.chat.completions.create({
        model: 'openai:gpt-4o-mini',
        messages,
        temperature: 0.7,
        max_tokens: 200
      });

      console.log('\nStep 3 - Final response:', finalResponse.choices[0].message.content);
    }
  } catch (error) {
    console.error('OpenAI Tool Calling Error:', error);
  }

  // Example 2: Anthropic Tool Calling
  console.log('\n--- Anthropic Tool Calling ---');
  try {
    // Step 1: Initial request with tools
    const response = await client.chat.completions.create({
      model: 'anthropic:claude-3-haiku-20240307',
      messages: [
        { role: 'system', content: 'You are a helpful weather assistant.' },
        { role: 'user', content: "What's the weather like in New York?" }
      ],
      tools,
      tool_choice: 'auto'
    });

    const message = response.choices[0]?.message;
    console.log('Step 1 - Initial response:', JSON.stringify(message, null, 2));

    if (message?.tool_calls) {
      // Step 2: Execute tool calls and send results back
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful weather assistant.' },
        { role: 'user', content: "What's the weather like in New York?" },
        message // The assistant's message with tool calls
      ];

      console.log('\nTool calls detected:');
      for (const toolCall of message.tool_calls) {
        console.log(`- Function: ${toolCall.function.name}`);
        console.log(`  Arguments: ${toolCall.function.arguments}`);
        
        // Execute the function
        const args = JSON.parse(toolCall.function.arguments);
        const result = getWeather(args.location, args.unit);
        console.log(`  Result:`, result);

        // Add tool result to messages
        messages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(result)
        });
      }

      // Step 3: Get final response with tool results
      console.log('\nStep 2 - Sending tool results back...');
      const finalResponse = await client.chat.completions.create({
        model: 'anthropic:claude-3-haiku-20240307',
        messages,
        temperature: 0.7,
        max_tokens: 200
      });

      console.log('\nStep 3 - Final response:', finalResponse.choices[0].message.content);
    }
  } catch (error) {
    console.error('Anthropic Tool Calling Error:', error);
  }
}

main().catch(console.error);