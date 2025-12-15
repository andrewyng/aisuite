/**
 * Integration tests for all providers.
 * These tests make REAL API calls and require API keys to be set in environment variables.
 * Tests are skipped automatically if the required API key is not available.
 *
 * Run with: npm run test:integration
 *
 * Set API keys in .env file at project root:
 *   OPENAI_API_KEY=your-key
 *   GOOGLE_API_KEY=your-key
 *   ANTHROPIC_API_KEY=your-key
 *   MISTRAL_API_KEY=your-key
 *   GROQ_API_KEY=your-key
 */

// Load .env file from project root
import * as dotenv from "dotenv";
dotenv.config();

import { Client } from "../../src/client";
import { ChatCompletionChunk, ChatCompletionResponse, Tool } from "../../src/types";

// All supported provider API keys
const API_KEYS = [
  "OPENAI_API_KEY",
  "GOOGLE_API_KEY",
  "ANTHROPIC_API_KEY",
  "MISTRAL_API_KEY",
  "GROQ_API_KEY",
];

// Check if at least one API key is set
const hasAnyApiKey = API_KEYS.some((key) => process.env[key]);

// Fail fast if no API keys are set
if (!hasAnyApiKey) {
  describe("Integration Tests", () => {
    it("should have at least one API key set", () => {
      throw new Error(
        `No API keys found! Set at least one of: ${API_KEYS.join(", ")}\n` +
        `Example: export GOOGLE_API_KEY=your-key && npm run test:integration`
      );
    });
  });
}

// Helper to conditionally run tests based on env var availability
const describeIfEnv = (envVar: string) =>
  process.env[envVar] ? describe : describe.skip;

// Shared tool definition for tool calling tests
const weatherTool: Tool = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the current weather for a location",
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city name, e.g. Paris, London, Tokyo",
        },
      },
      required: ["location"],
    },
  },
};

// ============================================================================
// OpenAI Integration Tests
// ============================================================================
describeIfEnv("OPENAI_API_KEY")("OpenAI Integration", () => {
  const client = new Client();
  const model = "openai:gpt-4o-mini";

  it("should complete a simple chat request", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: 'Say "hello" and nothing else' }],
    })) as ChatCompletionResponse;

    expect(response.choices[0].message.content?.toLowerCase()).toContain(
      "hello"
    );
  });

  it("should handle tool calls", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "What is the weather in Paris?" }],
      tools: [weatherTool],
    })) as ChatCompletionResponse;

    const toolCall = response.choices[0].message.tool_calls?.[0];
    expect(toolCall).toBeDefined();
    expect(toolCall?.function.name).toBe("get_weather");
    expect(toolCall?.function.arguments.toLowerCase()).toContain("paris");
  });

  it("should stream responses", async () => {
    const stream = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Count from 1 to 5" }],
      stream: true,
    });

    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    const fullContent = chunks
      .map((c) => c.choices[0]?.delta?.content || "")
      .join("");
    expect(fullContent).toBeTruthy();
  });
});

// ============================================================================
// Google Gemini Integration Tests
// ============================================================================
describeIfEnv("GOOGLE_API_KEY")("Google Gemini Integration", () => {
  const client = new Client();
  const model = "google:gemini-2.0-flash";

  it("should complete a simple chat request", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: 'Say "hello" and nothing else' }],
    })) as ChatCompletionResponse;

    expect(response.choices[0].message.content?.toLowerCase()).toContain(
      "hello"
    );
  });

  it("should handle tool calls", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "What is the weather in Paris?" }],
      tools: [weatherTool],
    })) as ChatCompletionResponse;

    const toolCall = response.choices[0].message.tool_calls?.[0];
    expect(toolCall).toBeDefined();
    expect(toolCall?.function.name).toBe("get_weather");
    expect(toolCall?.function.arguments.toLowerCase()).toContain("paris");
  });

  it("should stream responses", async () => {
    const stream = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Count from 1 to 5" }],
      stream: true,
    });

    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    const fullContent = chunks
      .map((c) => c.choices[0]?.delta?.content || "")
      .join("");
    expect(fullContent).toBeTruthy();
  });
});

// ============================================================================
// Anthropic Integration Tests
// ============================================================================
describeIfEnv("ANTHROPIC_API_KEY")("Anthropic Integration", () => {
  const client = new Client();
  const model = "anthropic:claude-3-haiku-20240307";

  it("should complete a simple chat request", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: 'Say "hello" and nothing else' }],
    })) as ChatCompletionResponse;

    expect(response.choices[0].message.content?.toLowerCase()).toContain(
      "hello"
    );
  });

  it("should handle tool calls", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "What is the weather in Paris?" }],
      tools: [weatherTool],
    })) as ChatCompletionResponse;

    const toolCall = response.choices[0].message.tool_calls?.[0];
    expect(toolCall).toBeDefined();
    expect(toolCall?.function.name).toBe("get_weather");
    expect(toolCall?.function.arguments.toLowerCase()).toContain("paris");
  });

  it("should stream responses", async () => {
    const stream = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Count from 1 to 5" }],
      stream: true,
    });

    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    const fullContent = chunks
      .map((c) => c.choices[0]?.delta?.content || "")
      .join("");
    expect(fullContent).toBeTruthy();
  });
});

// ============================================================================
// Mistral Integration Tests
// ============================================================================
describeIfEnv("MISTRAL_API_KEY")("Mistral Integration", () => {
  const client = new Client();
  const model = "mistral:mistral-small-latest";

  it("should complete a simple chat request", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: 'Say "hello" and nothing else' }],
    })) as ChatCompletionResponse;

    expect(response.choices[0].message.content?.toLowerCase()).toContain(
      "hello"
    );
  });

  it("should handle tool calls", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "What is the weather in Paris?" }],
      tools: [weatherTool],
    })) as ChatCompletionResponse;

    const toolCall = response.choices[0].message.tool_calls?.[0];
    expect(toolCall).toBeDefined();
    expect(toolCall?.function.name).toBe("get_weather");
    expect(toolCall?.function.arguments.toLowerCase()).toContain("paris");
  });

  it("should stream responses", async () => {
    const stream = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Count from 1 to 5" }],
      stream: true,
    });

    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    const fullContent = chunks
      .map((c) => c.choices[0]?.delta?.content || "")
      .join("");
    expect(fullContent).toBeTruthy();
  });
});

// ============================================================================
// Groq Integration Tests
// ============================================================================
describeIfEnv("GROQ_API_KEY")("Groq Integration", () => {
  const client = new Client();
  const model = "groq:llama-3.1-8b-instant";
  // Use tool-capable model for tool tests
  const toolModel = "groq:llama3-groq-70b-8192-tool-use-preview";

  it("should complete a simple chat request", async () => {
    const response = (await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: 'Say "hello" and nothing else' }],
    })) as ChatCompletionResponse;

    expect(response.choices[0].message.content?.toLowerCase()).toContain(
      "hello"
    );
  });

  it("should handle tool calls", async () => {
    const response = (await client.chat.completions.create({
      model: toolModel,
      messages: [{ role: "user", content: "What is the weather in Paris?" }],
      tools: [weatherTool],
    })) as ChatCompletionResponse;

    const toolCall = response.choices[0].message.tool_calls?.[0];
    expect(toolCall).toBeDefined();
    expect(toolCall?.function.name).toBe("get_weather");
    expect(toolCall?.function.arguments.toLowerCase()).toContain("paris");
  });

  it("should stream responses", async () => {
    const stream = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Count from 1 to 5" }],
      stream: true,
    });

    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of stream as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    const fullContent = chunks
      .map((c) => c.choices[0]?.delta?.content || "")
      .join("");
    expect(fullContent).toBeTruthy();
  });
});
