import 'dotenv/config';
import { Client, AISuiteError, ProviderNotConfiguredError, ChatCompletionChunk } from '../src';

// Test configuration
const ENABLE_OPENAI_TESTS = !!process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'your-openai-api-key-here';
const ENABLE_ANTHROPIC_TESTS = !!process.env.ANTHROPIC_API_KEY && process.env.ANTHROPIC_API_KEY !== 'your-anthropic-api-key-here';

console.log('🧪 AISuite Test Suite\n');
console.log(`OpenAI tests: ${ENABLE_OPENAI_TESTS ? '✅ Enabled' : '❌ Disabled (no API key)'}`);
console.log(`Anthropic tests: ${ENABLE_ANTHROPIC_TESTS ? '✅ Enabled' : '❌ Disabled (no API key)'}`);
console.log('');

// Helper function to run a test
async function runTest(name: string, fn: () => Promise<void>) {
  try {
    await fn();
    console.log(`✅ ${name}`);
  } catch (error) {
    console.error(`❌ ${name}`);
    console.error(`   Error: ${error instanceof Error ? error.message : error}`);
  }
}

async function main() {
  // Initialize client
  const client = new Client({
    ...(ENABLE_OPENAI_TESTS && { openai: { apiKey: process.env.OPENAI_API_KEY! } }),
    ...(ENABLE_ANTHROPIC_TESTS && { anthropic: { apiKey: process.env.ANTHROPIC_API_KEY! } }),
  });

  console.log('📋 Available providers:', client.listProviders());
  console.log('');

  // Test 1: Basic functionality
  console.log('🔍 Testing basic functionality...\n');
  
  await runTest('Model parser - valid format', async () => {
    const { parseModel } = await import('../src');
    const result = parseModel('openai:gpt-4');
    if (result.provider !== 'openai' || result.model !== 'gpt-4') {
      throw new Error('Model parser failed');
    }
  });

  await runTest('Model parser - invalid format', async () => {
    const { parseModel } = await import('../src');
    try {
      parseModel('invalid-format');
      throw new Error('Should have thrown error');
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes('Invalid model format')) {
        throw error;
      }
    }
  });

  await runTest('Provider not configured error', async () => {
    try {
      await client.chat.completions.create({
        model: 'invalid:model',
        messages: [{ role: 'user', content: 'test' }]
      });
      throw new Error('Should have thrown error');
    } catch (error) {
      if (!(error instanceof ProviderNotConfiguredError)) {
        throw error;
      }
    }
  });

  // Test 2: OpenAI Provider
  if (ENABLE_OPENAI_TESTS) {
    console.log('\n🤖 Testing OpenAI provider...\n');

    await runTest('OpenAI - Basic chat completion', async () => {
      const response = await client.chat.completions.create({
        model: 'openai:gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a test assistant. Respond with exactly: "Test successful"' },
          { role: 'user', content: 'Hello' }
        ],
        temperature: 0,
        max_tokens: 50,
      });

      if (!response.choices[0]?.message?.content) {
        throw new Error('No response content');
      }
      console.log(`   Response: ${response.choices[0].message.content.trim()}`);
    });

    await runTest('OpenAI - Multiple messages', async () => {
      const response = await client.chat.completions.create({
        model: 'openai:gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Say "A"' },
          { role: 'assistant', content: 'A' },
          { role: 'user', content: 'Say "B"' }
        ],
        temperature: 0,
        max_tokens: 10,
      });

      if (!response.choices[0]?.message?.content?.includes('B')) {
        throw new Error('Multi-turn conversation failed');
      }
    });

    await runTest('OpenAI - Tool calling', async () => {
      const response = await client.chat.completions.create({
        model: 'openai:gpt-3.5-turbo',
        messages: [
          { role: 'user', content: 'What is the weather in San Francisco?' }
        ],
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather for a location',
            parameters: {
              type: 'object',
              properties: {
                location: { type: 'string' }
              },
              required: ['location']
            }
          }
        }],
        tool_choice: 'auto'
      });

      const toolCalls = response.choices[0]?.message?.tool_calls;
      if (!toolCalls || toolCalls.length === 0) {
        throw new Error('No tool calls in response');
      }
      console.log(`   Tool called: ${toolCalls[0].function.name}`);
    });

    await runTest('OpenAI - Streaming support', async () => {
      const stream = await client.chat.completions.create({
        model: 'openai:gpt-3.5-turbo',
        messages: [{ role: 'user', content: 'Say exactly: "Stream test"' }],
        stream: true,
        temperature: 0,
        max_tokens: 20,
      }) as AsyncIterable<ChatCompletionChunk>;

      let content = '';
      for await (const chunk of stream) {
        content += chunk.choices[0]?.delta?.content || '';
      }
      
      if (!content.toLowerCase().includes('stream')) {
        throw new Error('Streaming response did not contain expected content');
      }
      console.log(`   Streamed: ${content.trim()}`);
    });
  }

  // Test 3: Anthropic Provider
  if (ENABLE_ANTHROPIC_TESTS) {
    console.log('\n🔮 Testing Anthropic provider...\n');

    await runTest('Anthropic - Basic chat completion', async () => {
      const response = await client.chat.completions.create({
        model: 'anthropic:claude-3-haiku-20240307',
        messages: [
          { role: 'system', content: 'You are a test assistant. Respond with exactly: "Test successful"' },
          { role: 'user', content: 'Hello' }
        ],
        temperature: 0,
        max_tokens: 50,
      });

      if (!response.choices[0]?.message?.content) {
        throw new Error('No response content');
      }
      console.log(`   Response: ${response.choices[0].message.content.trim()}`);
    });

    await runTest('Anthropic - System message handling', async () => {
      const response = await client.chat.completions.create({
        model: 'anthropic:claude-3-haiku-20240307',
        messages: [
          { role: 'system', content: 'Always respond in uppercase.' },
          { role: 'system', content: 'Also end with an exclamation mark.' },
          { role: 'user', content: 'say hello' }
        ],
        temperature: 0,
        max_tokens: 50,
      });

      const content = response.choices[0]?.message?.content || '';
      if (!content.includes('!') || content !== content.toUpperCase()) {
        console.log(`   Warning: System message may not be properly handled. Response: ${content}`);
      }
    });

    await runTest('Anthropic - Tool calling', async () => {
      const response = await client.chat.completions.create({
        model: 'anthropic:claude-3-haiku-20240307',
        messages: [
          { role: 'user', content: 'What is the weather in Paris?' }
        ],
        tools: [{
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather for a location',
            parameters: {
              type: 'object',
              properties: {
                location: { type: 'string', description: 'The city name' }
              },
              required: ['location']
            }
          }
        }],
        tool_choice: 'auto'
      });

      const toolCalls = response.choices[0]?.message?.tool_calls;
      if (!toolCalls || toolCalls.length === 0) {
        throw new Error('No tool calls in response');
      }
      console.log(`   Tool called: ${toolCalls[0].function.name}`);
      console.log(`   Arguments: ${toolCalls[0].function.arguments}`);
    });

    await runTest('Anthropic - Streaming support', async () => {
      const stream = await client.chat.completions.create({
        model: 'anthropic:claude-3-haiku-20240307',
        messages: [{ role: 'user', content: 'Say exactly: "Stream works"' }],
        stream: true,
        temperature: 0,
        max_tokens: 20,
      }) as AsyncIterable<ChatCompletionChunk>;

      let content = '';
      for await (const chunk of stream) {
        content += chunk.choices[0]?.delta?.content || '';
      }
      
      if (!content.toLowerCase().includes('stream')) {
        throw new Error('Streaming response did not contain expected content');
      }
      console.log(`   Streamed: ${content.trim()}`);
    });
  }

  // Test 4: Cross-provider compatibility
  if (ENABLE_OPENAI_TESTS && ENABLE_ANTHROPIC_TESTS) {
    console.log('\n🔄 Testing cross-provider compatibility...\n');

    await runTest('Same prompt - different providers', async () => {
      const prompt = {
        messages: [
          { role: 'system', content: 'You are a helpful assistant. Be concise.' },
          { role: 'user', content: 'What is 2+2?' }
        ] as const,
        temperature: 0,
        max_tokens: 50,
      };

      const openaiResponse = await client.chat.completions.create({
        ...prompt,
        model: 'openai:gpt-3.5-turbo',
      });

      const anthropicResponse = await client.chat.completions.create({
        ...prompt,
        model: 'anthropic:claude-3-haiku-20240307',
      });

      console.log(`   OpenAI: ${openaiResponse.choices[0].message.content?.trim()}`);
      console.log(`   Anthropic: ${anthropicResponse.choices[0].message.content?.trim()}`);
      
      // Both should mention "4" in their response
      if (!openaiResponse.choices[0].message.content?.includes('4') || 
          !anthropicResponse.choices[0].message.content?.includes('4')) {
        throw new Error('Providers gave inconsistent results');
      }
    });
  }

  console.log('\n✨ Test suite completed!\n');
}

// Run tests
main().catch(error => {
  console.error('\n💥 Test suite failed:', error);
  process.exit(1);
});