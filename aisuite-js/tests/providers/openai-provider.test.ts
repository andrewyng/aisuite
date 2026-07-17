import { OpenAIProvider } from "../../src/providers/openai/provider";
import { ChatCompletionRequest, ChatCompletionChunk } from "../../src/types";
import { AISuiteError } from "../../src/core/errors";

// Mock the OpenAI SDK
jest.mock("openai");

describe("OpenAIProvider", () => {
  let provider: OpenAIProvider;
  let mockOpenAIClient: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create mock OpenAI client
    mockOpenAIClient = {
      chat: {
        completions: {
          create: jest.fn(),
        },
      },
    };

    // Mock the OpenAI constructor
    const OpenAI = require("openai");
    OpenAI.mockImplementation(() => mockOpenAIClient);

    // Create provider instance
    provider = new OpenAIProvider({
      apiKey: "test-api-key",
    });
  });

  describe("constructor", () => {
    it("should initialize with basic config", () => {
      const config = { apiKey: "test-key" };
      const provider = new OpenAIProvider(config);

      expect(provider.name).toBe("openai");
    });

    it("should initialize with full config", () => {
      const config = {
        apiKey: "test-key",
        baseURL: "https://custom.openai.com",
        organization: "org-123",
      };
      const provider = new OpenAIProvider(config);

      expect(provider.name).toBe("openai");
    });
  });

  describe("chatCompletion", () => {
    const baseRequest: ChatCompletionRequest = {
      model: "gpt-4",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should successfully complete chat", async () => {
      const mockResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "gpt-4",
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

      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(baseRequest);

      expect(mockOpenAIClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-4",
          messages: [{ role: "user", content: "Hello" }],
        }),
        undefined
      );
      expect(result).toEqual(
        expect.objectContaining({
          id: "chatcmpl-123",
          object: "chat.completion",
          model: "gpt-4",
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
      mockOpenAIClient.chat.completions.create.mockRejectedValue(apiError);

      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        AISuiteError
      );
      await expect(provider.chatCompletion(baseRequest)).rejects.toThrow(
        "OpenAI API error: API rate limit exceeded"
      );
    });

    it("should pass options to the client", async () => {
      const options = { signal: new AbortController().signal };
      const mockResponse = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1234567890,
        model: "gpt-4",
        choices: [],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };

      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockResponse);

      await provider.chatCompletion(baseRequest, options);

      expect(mockOpenAIClient.chat.completions.create).toHaveBeenCalledWith(
        expect.any(Object),
        options
      );
    });

    it("should handle complex request with all parameters", async () => {
      const complexRequest: ChatCompletionRequest = {
        model: "gpt-4",
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
        model: "gpt-4",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "2+2 equals 4" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 15, completion_tokens: 5, total_tokens: 20 },
      };

      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockResponse);

      const result = await provider.chatCompletion(complexRequest);

      expect(mockOpenAIClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-4",
          messages: complexRequest.messages,
          temperature: 0.7,
          max_tokens: 100,
          top_p: 0.9,
          frequency_penalty: 0.1,
          presence_penalty: 0.1,
          stop: ["\n"],
          user: "user-123",
        }),
        undefined
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
      model: "gpt-4",
      messages: [{ role: "user", content: "Hello" }],
    };

    it("should stream chat completion", async () => {
      const mockChunks = [
        {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "gpt-4",
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
          model: "gpt-4",
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

      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(baseRequest);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockOpenAIClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-4",
          messages: [{ role: "user", content: "Hello" }],
          stream: true,
        }),
        undefined
      );
      expect(chunks).toHaveLength(2);
      expect(chunks[0]).toEqual(
        expect.objectContaining({
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
        })
      );
    });

    it("should handle streaming errors", async () => {
      const streamError = new Error("Streaming connection failed");
      mockOpenAIClient.chat.completions.create.mockRejectedValue(streamError);

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
      }).rejects.toThrow("OpenAI streaming error: Streaming connection failed");
    });

    it("should pass options to streaming request", async () => {
      const options = { signal: new AbortController().signal };
      const mockStream = (async function* () {
        yield {
          id: "chatcmpl-123",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "gpt-4",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Hello!" },
              finish_reason: "stop",
            },
          ],
        };
      })();

      mockOpenAIClient.chat.completions.create.mockResolvedValue(mockStream);

      const stream = provider.streamChatCompletion(baseRequest, options);
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(mockOpenAIClient.chat.completions.create).toHaveBeenCalledWith(
        expect.objectContaining({
          model: "gpt-4",
          messages: [{ role: "user", content: "Hello" }],
          stream: true,
        }),
        options
      );
      expect(chunks).toHaveLength(1);
    });
  });

  describe("error handling", () => {
    it("should preserve AISuiteError instances", async () => {
      const customError = new AISuiteError(
        "Custom error",
        "openai",
        "CUSTOM_ERROR"
      );

      mockOpenAIClient.chat.completions.create.mockRejectedValue(customError);

      await expect(
        provider.chatCompletion({
          model: "gpt-4",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow(customError);
    });

    it("should handle unknown error types", async () => {
      const unknownError = "Unknown error string";
      mockOpenAIClient.chat.completions.create.mockRejectedValue(unknownError);

      await expect(
        provider.chatCompletion({
          model: "gpt-4",
          messages: [{ role: "user", content: "Hello" }],
        })
      ).rejects.toThrow("OpenAI API error: Unknown error");
    });
  });
});
