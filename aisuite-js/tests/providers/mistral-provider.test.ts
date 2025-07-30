import { MistralProvider } from "../../src/providers/mistral/provider";
import { ChatCompletionRequest, ChatCompletionChunk } from "../../src/types";
import { AISuiteError } from "../../src/core/errors";

// Mock the Mistral SDK
jest.mock("@mistralai/mistralai", () => {
  return {
    __esModule: true,
    default: jest.fn(),
  };
});

describe("MistralProvider", () => {
  let provider: MistralProvider;
  let mockMistralClient: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create mock Mistral client
    mockMistralClient = {
      chat: jest.fn(),
      chatStream: jest.fn(),
    };

    // Mock the MistralClient constructor
    const MistralClient = require("@mistralai/mistralai");
    MistralClient.default.mockImplementation(() => mockMistralClient);

    // Create provider instance
    provider = new MistralProvider({
      apiKey: "test-api-key",
    });
  });

  describe("constructor", () => {
    it("should initialize with basic config", () => {
      const config = { apiKey: "test-key" };
      const provider = new MistralProvider(config);

      expect(provider.name).toBe("mistral");
    });

    it("should initialize with baseURL config", () => {
      const config = {
        apiKey: "test-key",
        baseURL: "https://custom.mistral.com",
      };
      const provider = new MistralProvider(config);

      expect(provider.name).toBe("mistral");
      expect(mockMistralClient.baseURL).toBe("https://custom.mistral.com");
    });
  });

  describe("chatCompletion", () => {
    const baseRequest: ChatCompletionRequest = {
      model: "mistral-large",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should successfully complete chat", async () => {
      const mockResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "mistral-large",
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

      mockMistralClient.chat.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(baseRequest);

      expect(mockMistralClient.chat).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "mistral-large",
        })
      );
      expect(result).toEqual(
        expect.objectContaining({
          id: "chatcmpl-123",
          object: "chat.completion",
          model: "mistral-large",
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
      mockMistralClient.chat.mockRejectedValue(apiError);

      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        AISuiteError
      );
      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        "Mistral API error: API rate limit exceeded"
      );
    });

    it("should handle complex request with all parameters", async () => {
      const complexRequest: ChatCompletionRequest = {
        model: "mistral-large",
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
        model: "mistral-large",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "2+2 equals 4" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 15, completion_tokens: 5, total_tokens: 20 },
      };

      mockMistralClient.chat.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(complexRequest);

      expect(mockMistralClient.chat).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "mistral-large",
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
      model: "mistral-large",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should stream chat completion", async () => {
      const mockChunks = [
        {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "mistral-large",
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
          model: "mistral-large",
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

      mockMistralClient.chatStream.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(baseRequest);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockMistralClient.chatStream).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "mistral-large",
        })
      );
      expect(chunks).toHaveLength(2);
      expect(typeof chunks[0].id).toBe("string");
      expect(chunks[0].object).toBe("chat.completion.chunk");
    });

    it("should handle streaming errors", async () => {
      mockMistralClient.chatStream.mockRejectedValue(
        new Error("Streaming error")
      );
      const stream = provider.streamChatCompletion(baseRequest);
      const iterator = stream[Symbol.asyncIterator]();
      await expect(iterator.next()).rejects.toThrow(AISuiteError);
    });

    it("should handle abort signal", async () => {
      const mockStream = (async function* () {
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "mistral-large",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Hello!" },
              finish_reason: "stop",
            },
          ],
        };
      })();

      mockMistralClient.chatStream.mockResolvedValue(mockStream);

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
        model: "mistral-large",
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
          model: "mistral-large",
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
          model: "mistral-large",
          choices: [
            {
              index: 0,
              delta: { content: " there was a brave knight." },
              finish_reason: "stop",
            },
          ],
        };
      })();

      mockMistralClient.chatStream.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(complexRequest);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockMistralClient.chatStream).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "mistral-large",
        })
      );
      expect(chunks).toHaveLength(2);
    });
  });

  describe("error handling", () => {
    it("should preserve AISuiteError instances", async () => {
      const customError = new AISuiteError(
        "Custom error",
        "mistral",
        "CUSTOM_ERROR"
      );

      mockMistralClient.chat.mockRejectedValue(customError);

      await expect(
        provider.chatCompletion({
          model: "mistral-large",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow(customError);
    });

    it("should handle unknown error types", async () => {
      const unknownError = "Unknown error string";
      mockMistralClient.chat.mockRejectedValue(unknownError);

      await expect(
        provider.chatCompletion({
          model: "mistral-large",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow("Mistral API error: Unknown error");
    });

    it("should handle streaming unknown error types", async () => {
      const unknownError = "Unknown streaming error";
      mockMistralClient.chatStream.mockRejectedValue(unknownError);

      const stream = provider.streamChatCompletion({
        model: "mistral-large",
        messages: [{ role: "user", content: "Hello" }],
      });

      await expect(async () => {
        for await (const chunk of stream) {
          // This should not be reached
        }
      }).rejects.toThrow(/Mistral streaming error: Unknown/);
    });
  });
});
