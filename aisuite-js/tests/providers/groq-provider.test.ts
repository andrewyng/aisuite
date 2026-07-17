import { GroqProvider } from "../../src/providers/groq/provider";
import { ChatCompletionRequest, ChatCompletionChunk } from "../../src/types";
import { AISuiteError } from "../../src/core/errors";

// Mock the Groq SDK
jest.mock("groq-sdk");

describe("GroqProvider", () => {
  let provider: GroqProvider;
  let mockGroqClient: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create mock Groq client
    mockGroqClient = {
      chat: {
        completions: {
          create: jest.fn(),
        },
      },
    };

    // Mock the Groq constructor
    const Groq = require("groq-sdk");
    Groq.mockImplementation(() => mockGroqClient);

    // Create provider instance
    provider = new GroqProvider({
      apiKey: "test-api-key",
    });
  });

  describe("constructor", () => {
    it("should initialize with basic config", () => {
      const config = { apiKey: "test-key" };
      const provider = new GroqProvider(config);

      expect(provider.name).toBe("groq");
    });

    it("should initialize with baseURL config", () => {
      const config = {
        apiKey: "test-key",
        baseURL: "https://custom.groq.com",
      };
      const provider = new GroqProvider(config);

      expect(provider.name).toBe("groq");
      expect(mockGroqClient.baseURL).toBe("https://custom.groq.com");
    });
  });

  describe("chatCompletion", () => {
    const baseRequest: ChatCompletionRequest = {
      model: "llama3-8b-8192",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should successfully complete chat", async () => {
      const mockResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "llama3-8b-8192",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Hello! How can I help you?",
            },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
        },
      };

      mockGroqClient.chat.completions.create.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(baseRequest);

      expect(mockGroqClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "llama3-8b-8192",
          messages: [{ role: "user", content: "Hello" }],
        })
      );
      expect(result).toEqual(
        expect.objectContaining({
          id: "chatcmpl-123",
          object: "chat.completion",
          model: "groq:llama3-8b-8192",
        })
      );
    });

    it("should throw error when streaming is enabled", async () => {
      const request: ChatCompletionRequest = {
        ...baseRequest,
        stream: true,
      };

      await expect(provider.chatCompletion(request)).rejects.toThrow(
        AISuiteError
      );
      await expect(provider.chatCompletion(request)).rejects.toThrow(
        "Streaming is not supported in non-streaming method"
      );
    });

    it("should handle API errors", async () => {
      const apiError = new Error("API rate limit exceeded");
      mockGroqClient.chat.completions.create.mockRejectedValue(apiError);

      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        AISuiteError
      );
      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        "Groq API error: API rate limit exceeded"
      );
    });

    it("should handle complex request with all parameters", async () => {
      const complexRequest: ChatCompletionRequest = {
        model: "llama3-8b-8192",
        messages: [
          { role: "system", content: "You are a helpful assistant" },
          { role: "user", content: "What is 2+2?" },
        ],
        temperature: 0.7,
        max_tokens: 100,
        top_p: 0.9,
        frequency_penalty: 0.1,
        presence_penalty: 0.1,
        stop: ["\n"],
        user: "user-123",
      };

      const mockResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "llama3-8b-8192",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "2+2 equals 4" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 15, completion_tokens: 5, total_tokens: 20 },
      };

      mockGroqClient.chat.completions.create.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(complexRequest);

      expect(mockGroqClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "llama3-8b-8192",
          messages: complexRequest.messages,
          temperature: 0.7,
          max_tokens: 100,
          stream: undefined,
          tool_choice: undefined,
          tools: undefined,
        })
      );
      expect(result).toEqual(
        expect.objectContaining({
          id: "chatcmpl-123",
          object: "chat.completion",
        })
      );
    });
  });

  describe("streamChatCompletion", () => {
    const baseRequest: ChatCompletionRequest = {
      model: "llama3-8b-8192",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should stream chat completion", async () => {
      const mockChunks = [
        {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "llama3-8b-8192",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Hello" },
              finish_reason: null,
            },
          ],
        },
        {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "llama3-8b-8192",
          choices: [
            {
              index: 0,
              delta: { content: "! How can I help?" },
              finish_reason: "stop",
            },
          ],
        },
      ];

      const mockStream = (async function* () {
        for (const chunk of mockChunks) {
          yield chunk;
        }
      })();

      mockGroqClient.chat.completions.create.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(baseRequest);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockGroqClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "llama3-8b-8192",
          messages: [{ role: "user", content: "Hello" }],
        })
      );
      expect(chunks).toHaveLength(2);
      expect(chunks[0]).toEqual(
        expect.objectContaining({
          object: "chat.completion.chunk",
        })
      );
    });

    it("should handle streaming errors", async () => {
      const streamError = new Error("Streaming connection failed");
      mockGroqClient.chat.completions.create.mockRejectedValue(streamError);

      await expect(async () => {
        const stream = provider.streamChatCompletion(baseRequest);
        for await (const chunk of stream) {
          // This should not be reached
        }
      }).rejects.toThrow(AISuiteError);

      await expect(async () => {
        const stream = provider.streamChatCompletion(baseRequest);
        for await (const chunk of stream) {
          // This should not be reached
        }
      }).rejects.toThrow("Groq streaming error: Streaming connection failed");
    });

    it("should handle abort signal", async () => {
      const mockStream = (async function* () {
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "llama3-8b-8192",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Hello!" },
              finish_reason: "stop",
            },
          ],
        };
      })();

      mockGroqClient.chat.completions.create.mockResolvedValue(mockStream);

      const abortController = new AbortController();
      const options = { signal: abortController.signal };

      const stream = provider.streamChatCompletion(baseRequest, options);
      const chunks: ChatCompletionChunk[] = [];

      // Start consuming the stream
      const consumePromise = (async () => {
        for await (const chunk of stream) {
          chunks.push(chunk);
        }
      })();

      // Abort after a short delay
      setTimeout(() => {
        abortController.abort();
      }, 10);

      await consumePromise;

      expect(chunks.length).toBeGreaterThan(0);
    });

    it("should handle complex streaming request", async () => {
      const complexRequest: ChatCompletionRequest = {
        model: "llama3-8b-8192",
        messages: [
          { role: "system", content: "You are a helpful assistant" },
          { role: "user", content: "Tell me a story" },
        ],
        temperature: 0.8,
        max_tokens: 200,
        top_p: 0.9,
        stop: ["END"],
      };

      const mockStream = (async function* () {
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "llama3-8b-8192",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Once upon a time" },
              finish_reason: null,
            },
          ],
        };
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "llama3-8b-8192",
          choices: [
            {
              index: 0,
              delta: { content: " there was a brave knight." },
              finish_reason: "stop",
            },
          ],
        };
      })();

      mockGroqClient.chat.completions.create.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(complexRequest);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockGroqClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "llama3-8b-8192",
          messages: complexRequest.messages,
          temperature: 0.8,
          max_tokens: 200,
          stream: undefined,
          tool_choice: undefined,
          tools: undefined,
        })
      );
      expect(chunks).toHaveLength(2);
    });
  });

  describe("error handling", () => {
    it("should preserve AISuiteError instances", async () => {
      const customError = new AISuiteError(
        "Custom error",
        "groq",
        "CUSTOM_ERROR"
      );

      mockGroqClient.chat.completions.create.mockRejectedValue(customError);

      await expect(
        provider.chatCompletion({
          model: "llama3-8b-8192",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow(customError);
    });

    it("should handle unknown error types", async () => {
      const unknownError = "Unknown error string";
      mockGroqClient.chat.completions.create.mockRejectedValue(unknownError);

      await expect(
        provider.chatCompletion({
          model: "llama3-8b-8192",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow("Groq API error: Unknown error");
    });

    it("should handle streaming unknown error types", async () => {
      const unknownError = "Unknown streaming error";
      mockGroqClient.chat.completions.create.mockRejectedValue(unknownError);

      const stream = provider.streamChatCompletion({
        model: "llama3-8b-8192",
        messages: [{ role: "user", content: "Hello" }],
      });

      await expect(async () => {
        for await (const chunk of stream) {
          // This should not be reached
        }
      }).rejects.toThrow("Groq streaming error: Unknown error");
    });
  });
});
