# AISuite

AISuite is a unified TypeScript library that provides a single, consistent interface for interacting with multiple Large Language Model (LLM) providers. The library uses OpenAI's API format as the standard interface while supporting OpenAI and Anthropic Claude.

npm pacakge - `npm i aisuite`

## Features

- **Unified API**: Single interface compatible with OpenAI's API structure
- **Multi-Provider Support**: Currently supports OpenAI and Anthropic
- **Provider Selection**: Use `provider:model` format (e.g., `openai:gpt-4o`, `anthropic:claude-3-haiku-20240307`)
- **Tool Calling**: Transparent tool/function calling across all providers
- **Streaming**: Real-time streaming responses with consistent API
- **Type Safety**: Full TypeScript support with comprehensive type definitions
- **Error Handling**: Unified error handling across providers

## Installation

```bash
npm install aisuite
```

## Quick Start

```typescript
import { Client } from 'aisuite';

const client = new Client({
  openai: { apiKey: process.env.OPENAI_API_KEY },
  anthropic: { apiKey: process.env.ANTHROPIC_API_KEY },
});

// Use any provider with identical interface
const response = await client.chat.completions.create({
  model: 'openai:gpt-4o',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' }
  ],
});

console.log(response.choices[0].message.content);
```

## Usage Examples

### Basic Chat Completion

```typescript
// OpenAI
const openaiResponse = await client.chat.completions.create({
  model: 'openai:gpt-4o',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is TypeScript?' }
  ],
  temperature: 0.7,
  max_tokens: 1000,
});

// Anthropic - exact same interface
const anthropicResponse = await client.chat.completions.create({
  model: 'anthropic:claude-3-haiku-20240307',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is TypeScript?' }
  ],
  temperature: 0.7,
  max_tokens: 1000,
});
```

### Tool/Function Calling

```typescript
const tools = [
  {
    type: 'function' as const,
    function: {
      name: 'get_weather',
      description: 'Get current weather for a location',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City name' }
        },
        required: ['location']
      }
    }
  }
];

// Works identically across all providers
const response = await client.chat.completions.create({
  model: 'anthropic:claude-3-haiku-20240307',
  messages: [{ role: 'user', content: 'What\'s the weather in NYC?' }],
  tools,
  tool_choice: 'auto'
});

if (response.choices[0].message.tool_calls) {
  console.log('Tool calls:', response.choices[0].message.tool_calls);
}
```

### Streaming Responses

```typescript
const stream = await client.chat.completions.create({
  model: 'openai:gpt-4o',
  messages: [{ role: 'user', content: 'Tell me a story' }],
  stream: true
});

// TypeScript: cast to AsyncIterable<ChatCompletionChunk>
for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

### Streaming with Abort Controller

```typescript
const controller = new AbortController();

// Abort after 5 seconds
setTimeout(() => controller.abort(), 5000);

const stream = await client.chat.completions.create({
  model: 'anthropic:claude-3-haiku-20240307',
  messages: [{ role: 'user', content: 'Write a long story' }],
  stream: true
}, { signal: controller.signal });

try {
  for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
} catch (error) {
  if (error.name === 'AbortError') {
    console.log('Stream aborted');
  }
}
```

### Error Handling

```typescript
import { AISuiteError, ProviderNotConfiguredError } from 'aisuite';

try {
  const response = await client.chat.completions.create({
    model: 'invalid:model',
    messages: [{ role: 'user', content: 'Hello' }]
  });
} catch (error) {
  if (error instanceof ProviderNotConfiguredError) {
    console.error('Provider not configured:', error.message);
  } else if (error instanceof AISuiteError) {
    console.error('AISuite error:', error.message, error.provider);
  } else {
    console.error('Unknown error:', error);
  }
}
```

## API Reference

### Client Configuration

```typescript
const client = new Client({
  openai?: {
    apiKey: string;
    baseURL?: string;
    organization?: string;
  },
  anthropic?: {
    apiKey: string;
    baseURL?: string;
  }
});
```

### Chat Completion Request

All providers use the standard OpenAI chat completion format:

```typescript
interface ChatCompletionRequest {
  model: string;              // "provider:model" format
  messages: ChatMessage[];
  tools?: Tool[];
  tool_choice?: ToolChoice;
  temperature?: number;
  max_tokens?: number;
  stop?: string | string[];
  stream?: boolean;
}
```

### Helper Methods

```typescript
// List configured providers
client.listProviders(); // ['openai', 'anthropic']

// Check if a provider is configured
client.isProviderConfigured('openai'); // true
```

## Current Limitations

- Only OpenAI and Anthropic providers are currently supported (Gemini, Mistral, and Bedrock coming soon)
- Tool calling requires handling tool responses manually
- Streaming tool calls require manual accumulation of arguments

## Development

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test

# Run examples
#Run basic usage example only:
npm run example:basic
# Run tool calling example only:
npm run example:tools
# Run the full test suite:
npm run test:examples
```

## License

MIT
