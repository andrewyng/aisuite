import 'dotenv/config';
import { Client } from '../src';

async function main() {
  // Initialize the client with API keys
  const client = new Client({
    openai: { apiKey: process.env.OPENAI_API_KEY! },
    anthropic: { apiKey: process.env.ANTHROPIC_API_KEY! },
  });

  console.log('Available providers:', client.listProviders());

  // Example 1: OpenAI Chat Completion
  console.log('\n--- OpenAI Example ---');
  try {
    const openaiResponse = await client.chat.completions.create({
      model: 'openai:gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is TypeScript in one sentence?' }
      ],
      temperature: 0.7,
      max_tokens: 100,
    });

    console.log('OpenAI Response:', openaiResponse.choices[0].message.content);
    console.log('Usage:', openaiResponse.usage);
    console.log('Full response:', JSON.stringify(openaiResponse, null, 2));
  } catch (error) {
    console.error('OpenAI Error:', error);
  }

  // Example 2: Anthropic Chat Completion
  console.log('\n--- Anthropic Example ---');
  try {
    const anthropicResponse = await client.chat.completions.create({
      model: 'anthropic:claude-3-haiku-20240307',
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is TypeScript in one sentence?' }
      ],
      temperature: 0.7,
      max_tokens: 100,
    });

    console.log('Anthropic Response:', anthropicResponse.choices[0].message.content);
    console.log('Usage:', anthropicResponse.usage);
    console.log('Full response:', JSON.stringify(anthropicResponse, null, 2));
  } catch (error) {
    console.error('Anthropic Error:', error);
  }

  // Example 3: Error handling - invalid provider
  console.log('\n--- Error Handling Example ---');
  try {
    await client.chat.completions.create({
      model: 'invalid:model',
      messages: [{ role: 'user', content: 'Hello' }]
    });
  } catch (error) {
    console.error('Expected error:', error);
  }
}

// Run the examples
main().catch(console.error);