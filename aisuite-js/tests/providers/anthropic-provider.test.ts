import { AnthropicProvider } from "../../src/providers/anthropic/provider";
import { ChatCompletionRequest, ChatCompletionChunk } from "../../src/types";
import { AISuiteError } from "../../src/core/errors";

// Mock the Anthropic SDK
jest.mock("@anthropic-ai/sdk", () => {
  return {
    __esModule: true,
    default: jest.fn(),
  };
});

describe("AnthropicProvider", () => {
  let provider: AnthropicProvider;
  let mockAnthropicClient: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create mock Anthropic client
    mockAnthropicClient = {
      messages: {
        create: jest.fn(),
      },
    };

    // Mock the Anthropic constructor
    const Anthropic = require("@anthropic-ai/sdk");
    Anthropic.default.mockImplementation(() => mockAnthropicClient);

    // Ensure the mock is properly structured
    mockAnthropicClient.messages = {
      create: jest.fn(),
    };

    // Create provider instance
    provider = new AnthropicProvider({
      apiKey: "test-api-key",
    });
  });

  describe("constructor", () => {
    it("should initialize with basic config", () => {
      const config = { apiKey: "test-key" };
      const provider = new AnthropicProvider(config);

      expect(provider.name).toBe("anthropic");
    });

    it("should initialize with full config", () => {
      const config = {
        apiKey: "test-key",
        baseURL: "https://custom.anthropic.com",
      };
      const provider = new AnthropicProvider(config);

      expect(provider.name).toBe("anthropic");
    });
  });

  describe("chatCompletion", () => {
    const baseRequest: ChatCompletionRequest = {
      model: "claude-3-sonnet",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should successfully complete chat", async () => {
      const mockResponse = {
        id: "msg_123",
        type: "message",
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Hello! How can I help you?",
          },
        ],
        model: "claude-3-sonnet-20240229",
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: {
          input_tokens: 10,
          output_tokens: 20,
        },
      };

      mockAnthropicClient.messages.create.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(baseRequest);

      expect(mockAnthropicClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: expect.any(String),
          messages: expect.arrayContaining([
            expect.objectContaining({ role: "user", content: "Hello" }),
          ]),
        }),
        undefined
      );
      expect(result).toEqual(
        expect.objectContaining({
          id: "msg_123",
          object: "chat.completion",
          model: "claude-3-sonnet",
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
        "Streaming is not yet supported"
      );
    });

    it("should handle API errors", async () => {
      const apiError = new Error("API rate limit exceeded");
      mockAnthropicClient.messages.create.mockRejectedValue(apiError);

      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        AISuiteError
      );
      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        "Anthropic API error: API rate limit exceeded"
      );
    });

    it("should pass options to the client", async () => {
      const options = { signal: new AbortController().signal };
      const mockResponse = {
        id: "msg_123",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "Hello!" }],
        model: "claude-3-sonnet-20240229",
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 5 },
      };

      mockAnthropicClient.messages.create.mockResolvedValue(mockResponse);

      await provider.chatCompletion(baseRequest, options);

      expect(mockAnthropicClient.messages.create).toHaveBeenCalledWith(
        expect.any(Object),
        options
      );
    });

    it("should handle complex request with all parameters", async () => {
      const complexRequest: ChatCompletionRequest = {
        model: "claude-3-sonnet",
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
        id: "msg_123",
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: "2+2 equals 4" }],
        model: "claude-3-sonnet-20240229",
        stop_reason: "end_turn",
        usage: { input_tokens: 15, output_tokens: 5 },
      };

      mockAnthropicClient.messages.create.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(complexRequest);

      expect(mockAnthropicClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: expect.any(String),
          messages: expect.arrayContaining([
            expect.objectContaining({ role: "user", content: "What is 2+2?" }),
          ]),
          system: "You are a helpful assistant",
          temperature: 0.7,
          max_tokens: 100,
          top_p: 0.9,
          stop_sequences: expect.any(Array),
        }),
        undefined
      );
      expect(result).toEqual(
        expect.objectContaining({
          id: "msg_123",
          object: "chat.completion",
        })
      );
    });
  });

  describe("streamChatCompletion", () => {
    const baseRequest: ChatCompletionRequest = {
      model: "claude-3-sonnet",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should stream chat completion", async () => {
      const mockEvents = [
        {
          type: "message_start",
          message: {
            id: "msg_123",
            type: "message",
            role: "assistant",
            content: [],
            model: "claude-3-sonnet-20240229",
          },
        },
        {
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "text",
            text: "",
          },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "text_delta",
            text: "Hello",
          },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "text_delta",
            text: "! How can I help?",
          },
        },
        {
          type: "content_block_stop",
          index: 0,
        },
        {
          type: "message_delta",
          delta: {
            stop_reason: "end_turn",
            stop_sequence: null,
          },
        },
        {
          type: "message_stop",
        },
      ];

      const mockStream = (async function* () {
        for (const event of mockEvents) {
          yield event;
        }
      })();

      mockAnthropicClient.messages.create.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(baseRequest);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockAnthropicClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: expect.any(String),
          messages: expect.any(Array),
        }),
        undefined
      );
      expect(chunks.length).toBeGreaterThan(0);
    });

    it("should handle streaming errors", async () => {
      mockAnthropicClient.messages.create.mockRejectedValue(
        new Error("Streaming error")
      );
      const stream = provider.streamChatCompletion(baseRequest);
      const iterator = stream[Symbol.asyncIterator]();
      await expect(iterator.next()).rejects.toThrow(AISuiteError);
    });

    it("should pass options to streaming request", async () => {
      const options = { signal: new AbortController().signal };
      const mockStream = (async function* () {
        yield {
          type: "message_start",
          message: {
            id: "msg_123",
            type: "message",
            role: "assistant",
            content: [],
            model: "claude-3-sonnet-20240229",
          },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "text_delta",
            text: "Hello!",
          },
        };
        yield {
          type: "message_stop",
        };
      })();

      mockAnthropicClient.messages.create.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(baseRequest, options);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockAnthropicClient.messages.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: expect.any(String),
          messages: expect.any(Array),
          stream: true,
        }),
        options
      );
      expect(chunks.length).toBeGreaterThan(0);
    });

    it("should handle abort signal", async () => {
      const mockStream = (async function* () {
        yield {
          type: "message_start",
          message: {
            id: "msg_123",
            type: "message",
            role: "assistant",
            content: [],
            model: "claude-3-sonnet-20240229",
          },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "text_delta",
            text: "Hello!",
          },
        };
      })();

      mockAnthropicClient.messages.create.mockResolvedValue(mockStream);

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
  });

  describe("error handling", () => {
    it("should preserve AISuiteError instances", async () => {
      const customError = new AISuiteError(
        "Custom error",
        "anthropic",
        "CUSTOM_ERROR"
      );

      mockAnthropicClient.messages.create.mockRejectedValue(customError);

      await expect(
        provider.chatCompletion({
          model: "claude-3-sonnet",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow(customError);
    });

    it("should handle unknown error types", async () => {
      const unknownError = "Unknown error string";
      mockAnthropicClient.messages.create.mockRejectedValue(unknownError);

      await expect(
        provider.chatCompletion({
          model: "claude-3-sonnet",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow("Anthropic API error: Unknown error");
    });
  });
});
