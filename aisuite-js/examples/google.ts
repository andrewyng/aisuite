import { Client } from '../src';
import 'dotenv/config';

async function main() {
  const apiKey = process.env.GOOGLE_API_KEY;

  if (!apiKey) {
    console.error('Error: GOOGLE_API_KEY environment variable is not set');
    console.log('\nTo get an API key:');
    console.log('1. Go to https://aistudio.google.com/apikey');
    console.log('2. Create an API key');
    console.log('3. Run: export GOOGLE_API_KEY="your-api-key"');
    process.exit(1);
  }

  const client = new Client({
    google: { apiKey }
  });

  // Test non-streaming
  console.log('=== Testing Non-Streaming ===');
  const response = await client.chat.completions.create({
    model: 'google:gemini-2.0-flash',
    messages: [{ role: 'user', content: 'What is 2 + 2? Reply in one word.' }]
  });
  console.log('Response:', response.choices[0].message.content);
  console.log('Usage:', response.usage);

  // Test streaming
  console.log('\n=== Testing Streaming ===');
  const stream = await client.chat.completions.create({
    model: 'google:gemini-2.0-flash',
    messages: [{ role: 'user', content: 'Count from 1 to 5, one number per line.' }],
    stream: true
  });

  process.stdout.write('Streaming response: ');
  for await (const chunk of stream as AsyncIterable<any>) {
    if (chunk.choices[0]?.delta?.content) {
      process.stdout.write(chunk.choices[0].delta.content);
    }
  }
  console.log();

  // Test with system instruction
  console.log('\n=== Testing System Instruction ===');
  const pirateResponse = await client.chat.completions.create({
    model: 'google:gemini-2.0-flash',
    messages: [
      { role: 'system', content: 'You are a pirate. Respond in pirate speak.' },
      { role: 'user', content: 'Hello, how are you?' }
    ]
  });
  console.log('Pirate response:', pirateResponse.choices[0].message.content);

  console.log('\n=== All tests passed! ===');
}

main().catch(error => {
  console.error('Error:', error.message);
  process.exit(1);
});
