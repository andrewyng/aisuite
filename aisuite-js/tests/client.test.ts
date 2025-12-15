import { Client } from "../src/client";
import {
  ProviderConfigs,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  RequestOptions,
  TranscriptionRequest,
} from "../src/types";
import { ProviderNotConfiguredError } from "../src/core/errors";
import { Provider } from "../src/core/base-provider";

// Mock the Mistral SDK
jest.mock("@mistralai/mistralai", () => {
  return {
    __esModule: true,
    default: jest.fn(),
  };
});

// Mock the providers
jest.mock("../src/providers/openai");
jest.mock("../src/providers/anthropic");
jest.mock("../src/providers/mistral");
jest.mock("../src/providers/groq");
jest.mock("../src/providers/google");
jest.mock("../src/asr-providers/deepgram");

describe("Client", () => {
  let mockOpenAIProvider: any;
  let mockAnthropicProvider: any;
  let mockMistralProvider: any;
  let mockGroqProvider: any;
  let mockGoogleProvider: any;
  let mockDeepgramProvider: any;
  let mockOpenAIASRProvider: any;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Create mock response
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

    // Create mock provider class
    class MockProvider implements Provider {
      public readonly name: string;
      public chatCompletion = jest.fn().mockResolvedValue(mockResponse);
      public streamChatCompletion = jest.fn().mockImplementation(async function* () {
        yield mockResponse as unknown as ChatCompletionChunk;
      });

      constructor(name: string) {
        this.name = name;
      }
    }

    // Create mock instances
    mockOpenAIProvider = new MockProvider("openai");
    mockAnthropicProvider = new MockProvider("anthropic");
    mockMistralProvider = new MockProvider("mistral");
    mockGroqProvider = new MockProvider("groq");
    mockGoogleProvider = new MockProvider("google");

    mockDeepgramProvider = {
      transcribe: jest.fn(),
    };

    mockOpenAIASRProvider = {
      transcribe: jest.fn(),
    };

    // Manually mock the provider constructors using jest.mock
    const openaiModule = jest.requireMock("../src/providers/openai");
    const anthropicModule = jest.requireMock("../src/providers/anthropic");
    const mistralModule = jest.requireMock("../src/providers/mistral");
    const groqModule = jest.requireMock("../src/providers/groq");
    const googleModule = jest.requireMock("../src/providers/google");
    const deepgramModule = jest.requireMock("../src/asr-providers/deepgram");

    openaiModule.OpenAIProvider = jest.fn().mockImplementation(() => mockOpenAIProvider);
    anthropicModule.AnthropicProvider = jest.fn().mockImplementation(() => mockAnthropicProvider);
    mistralModule.MistralProvider = jest.fn().mockImplementation(() => mockMistralProvider);
    groqModule.GroqProvider = jest.fn().mockImplementation(() => mockGroqProvider);
    googleModule.GoogleProvider = jest.fn().mockImplementation(() => mockGoogleProvider);
    deepgramModule.DeepgramASRProvider = jest.fn().mockImplementation(() => mockDeepgramProvider);
  });

  describe("constructor", () => {
    // All providers are now always registered (for env var support)
    const allChatProviders = ["openai", "anthropic", "mistral", "groq", "google"];
    const allASRProviders = ["openai", "deepgram"];

    it("should register all providers regardless of config", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        anthropic: { apiKey: "anthropic-key" },
        mistral: { apiKey: "mistral-key" },
        groq: { apiKey: "groq-key" },
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      // All providers are always registered (can use env vars if no explicit config)
      expect(client.listProviders()).toEqual(allChatProviders);
      expect(client.listASRProviders()).toEqual(allASRProviders);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(true);
      expect(client.isProviderConfigured("mistral")).toBe(true);
      expect(client.isProviderConfigured("groq")).toBe(true);
      expect(client.isProviderConfigured("google")).toBe(true);
      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
    });

    it("should register all providers even with partial config", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        groq: { apiKey: "groq-key" },
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      // All providers are registered (those without explicit config will use env vars)
      expect(client.listProviders()).toEqual(allChatProviders);
      expect(client.listASRProviders()).toEqual(allASRProviders);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(true);
      expect(client.isProviderConfigured("mistral")).toBe(true);
      expect(client.isProviderConfigured("groq")).toBe(true);
      expect(client.isProviderConfigured("google")).toBe(true);
      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
      expect(client.isASRProviderConfigured("unknown")).toBe(false);
    });

    it("should register all providers even with empty config", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      // All providers are registered (will use env vars for API keys)
      expect(client.listProviders()).toEqual(allChatProviders);
      expect(client.listASRProviders()).toEqual(allASRProviders);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
    });

    it("should work with no config argument", () => {
      const client = new Client();

      // All providers are registered (will use env vars for API keys)
      expect(client.listProviders()).toEqual(allChatProviders);
      expect(client.listASRProviders()).toEqual(allASRProviders);
    });
  });

  describe("chat.completions.create", () => {
    let client: Client;
    const baseConfig: ProviderConfigs = {
      openai: { apiKey: "openai-key" },
      anthropic: { apiKey: "anthropic-key" },
      mistral: { apiKey: "mistral-key" },
      groq: { apiKey: "groq-key" },
    };

    beforeEach(() => {
      client = new Client(baseConfig);
    });

    it("should call non-streaming chat completion", async () => {
      const request: ChatCompletionRequest = {
        model: "openai:gpt-4",
        messages: [{ role: "user", content: "Hello" }],
      };

      const mockResponse = {
        id: "test-id",
        object: "chat.completion",
        created: 1234567890,
        model: "gpt-4",
        choices: [],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };

      mockOpenAIProvider.chatCompletion.mockResolvedValue(mockResponse);

      const result = await client.chat.completions.create(request);

      expect(mockOpenAIProvider.chatCompletion).toHaveBeenCalledWith(
        { ...request, model: "gpt-4" },
        undefined
      );
      expect(result).toEqual(mockResponse);
    });

    it("should call streaming chat completion", async () => {
      const request: ChatCompletionRequest = {
        model: "anthropic:claude-3-sonnet",
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      };

      const mockStream = (async function* () {
        yield {
          id: "chunk-1",
          object: "chat.completion.chunk",
          created: 1234567890,
          model: "claude-3-sonnet",
          choices: [],
        };
      })();

      mockAnthropicProvider.streamChatCompletion.mockReturnValue(mockStream);

      const result = await client.chat.completions.create(request);

      expect(mockAnthropicProvider.streamChatCompletion).toHaveBeenCalledWith(
        { ...request, model: "claude-3-sonnet" },
        undefined
      );
      expect(result).toBe(mockStream);
    });

    it("should throw error for unconfigured provider", async () => {
      const request: ChatCompletionRequest = {
        model: "unknown:model",
        messages: [{ role: "user", content: "Hello" }],
      };

      await expect(client.chat.completions.create(request)).rejects.toThrow(
        ProviderNotConfiguredError
      );
    });

    it("should handle complex model names with multiple colons", async () => {
      const request: ChatCompletionRequest = {
        model: "openai:gpt-4:vision",
        messages: [{ role: "user", content: "Hello" }],
      };

      const mockResponse = {
        id: "test-id",
        object: "chat.completion",
        created: 1234567890,
        model: "gpt-4:vision",
        choices: [],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };

      mockOpenAIProvider.chatCompletion.mockResolvedValue(mockResponse);

      const result = await client.chat.completions.create(request);

      expect(mockOpenAIProvider.chatCompletion).toHaveBeenCalledWith(
        { ...request, model: "gpt-4:vision" },
        undefined
      );
      expect(result).toEqual(mockResponse);
    });

    it("should pass options to provider", async () => {
      const request: ChatCompletionRequest = {
        model: "mistral:mistral-large",
        messages: [{ role: "user", content: "Hello" }],
      };

      const options = { signal: new AbortController().signal };

      const mockResponse = {
        id: "test-id",
        object: "chat.completion",
        created: 1234567890,
        model: "mistral-large",
        choices: [],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };

      mockMistralProvider.chatCompletion.mockResolvedValue(mockResponse);

      const result = await client.chat.completions.create(request, options);

      expect(mockMistralProvider.chatCompletion).toHaveBeenCalledWith(
        { ...request, model: "mistral-large" },
        options
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe("audio.transcriptions.create", () => {
    describe("Deepgram Provider", () => {
      let client: Client;
      const baseConfig: ProviderConfigs = {
        deepgram: { apiKey: "deepgram-key" },
      };

      beforeEach(() => {
        client = new Client(baseConfig);
      });

      it("should call transcription with correct parameters", async () => {
        const audioBuffer = Buffer.from("test audio data");
        const request: TranscriptionRequest = {
          model: "deepgram:nova-2",
          file: audioBuffer,
          language: "en-US",
          timestamps: true,
          word_confidence: true,
          speaker_labels: true,
        };

        const mockResponse = {
          text: "Hello world",
          language: "en-US",
          confidence: 0.95,
          words: [
            {
              text: "Hello",
              start: 0.0,
              end: 0.5,
              confidence: 0.98,
            },
            {
              text: "world",
              start: 0.6,
              end: 1.0,
              confidence: 0.92,
            },
          ],
          segments: [
            {
              text: "Hello world",
              start: 0.0,
              end: 1.0,
            },
          ],
        };

        mockDeepgramProvider.transcribe.mockResolvedValue(mockResponse);

        const result = await client.audio.transcriptions.create(request);

        expect(mockDeepgramProvider.transcribe).toHaveBeenCalledWith(
          { ...request, model: "nova-2" },
          undefined
        );
        expect(result).toEqual(mockResponse);
      });

      it("should throw error for unconfigured ASR provider", async () => {
        const request: TranscriptionRequest = {
          model: "unknown:model",
          file: Buffer.from("test"),
        };

        await expect(
          client.audio.transcriptions.create(request)
        ).rejects.toThrow(ProviderNotConfiguredError);
      });

      it("should pass options to ASR provider", async () => {
        const audioBuffer = Buffer.from("test audio data");
        const request: TranscriptionRequest = {
          model: "deepgram:nova-2",
          file: audioBuffer,
          language: "en-US",
        };

        const options = { timeout: 30000 };

        const mockResponse = {
          text: "Test transcription",
          language: "en-US",
          confidence: 0.9,
          words: [],
          segments: [],
        };

        mockDeepgramProvider.transcribe.mockResolvedValue(mockResponse);

        const result = await client.audio.transcriptions.create(
          request,
          options
        );

        expect(mockDeepgramProvider.transcribe).toHaveBeenCalledWith(
          { ...request, model: "nova-2" },
          options
        );
        expect(result).toEqual(mockResponse);
      });

      it("should handle complex model names with multiple colons", async () => {
        const audioBuffer = Buffer.from("test audio data");
        const request: TranscriptionRequest = {
          model: "deepgram:nova-2:enhanced",
          file: audioBuffer,
          language: "en-US",
        };

        const mockResponse = {
          text: "Test transcription",
          language: "en-US",
          confidence: 0.9,
          words: [],
          segments: [],
        };

        mockDeepgramProvider.transcribe.mockResolvedValue(mockResponse);

        const result = await client.audio.transcriptions.create(request);

        expect(mockDeepgramProvider.transcribe).toHaveBeenCalledWith(
          { ...request, model: "nova-2:enhanced" },
          undefined
        );
        expect(result).toEqual(mockResponse);
      });
    });

    describe("OpenAI ASR Provider", () => {
      let client: Client;
      let mockOpenAIASRProvider: any;

      beforeEach(() => {
        mockOpenAIASRProvider = {
          name: "openai",
          transcribe: jest.fn(),
        };

        // Update to configure OpenAI provider for both chat and ASR
        const openaiModule = require("../src/providers/openai");
        openaiModule.OpenAIProvider.mockImplementation(() => ({
          ...mockOpenAIProvider,
          ...mockOpenAIASRProvider
        }));
      });

      it("should transcribe with audio enabled", async () => {
        // Initialize client with OpenAI provider
        client = new Client({
          openai: {
            apiKey: "openai-key"
          },
        });

        // Add the OpenAI provider to ASR providers list manually for testing
        client["asrProviders"].set("openai", mockOpenAIASRProvider);

        const audioBuffer = Buffer.from("test audio data");
        const request: TranscriptionRequest = {
          model: "openai:whisper-1",
          file: audioBuffer,
          language: "en",
          response_format: "verbose_json",
          temperature: 0,
          timestamps: true,
        };

        const mockResponse = {
          text: "Test transcription",
          language: "en",
          confidence: 0.95,
          words: [
            {
              text: "Test",
              start: 0.0,
              end: 0.5,
              confidence: 0.98,
            },
            {
              text: "transcription",
              start: 0.6,
              end: 1.2,
              confidence: 0.92,
            },
          ],
          segments: [
            {
              text: "Test transcription",
              start: 0.0,
              end: 1.2,
            },
          ],
        };

        mockOpenAIASRProvider.transcribe.mockResolvedValue(mockResponse);

        const result = await client.audio.transcriptions.create(request);

        expect(mockOpenAIASRProvider.transcribe).toHaveBeenCalledWith(
          { ...request, model: "whisper-1" },
          undefined
        );
        expect(result).toEqual(mockResponse);
      });

      it("should support different response formats", async () => {
        client = new Client({
          openai: {
            apiKey: "openai-key"
          },
        });

        // Add the OpenAI provider to ASR providers list
        client["asrProviders"].set("openai", mockOpenAIASRProvider);

        const request: TranscriptionRequest = {
          model: "openai:whisper-1",
          file: Buffer.from("test audio"),
          response_format: "text",
        };

        const mockResponse = {
          text: "Test transcription",
          language: "en",
          confidence: 1.0,
          words: [],
          segments: [],
        };

        mockOpenAIASRProvider.transcribe.mockResolvedValue(mockResponse);
        const result = await client.audio.transcriptions.create(request);

        expect(mockOpenAIASRProvider.transcribe).toHaveBeenCalledWith(
          { ...request, model: "whisper-1" },
          undefined
        );
        expect(result).toEqual(mockResponse);
      });

      it("should pass custom options to provider", async () => {
        client = new Client({
          openai: {
            apiKey: "openai-key"
          },
        });

        // Add the OpenAI provider to ASR providers list
        client["asrProviders"].set("openai", mockOpenAIASRProvider);

        const request: TranscriptionRequest = {
          model: "openai:whisper-1",
          file: Buffer.from("test audio"),
          language: "en",
        };

        const options = { timeout: 30000 };
        const mockResponse = {
          text: "Test transcription",
          language: "en",
          confidence: 0.9,
          words: [],
          segments: [],
        };

        mockOpenAIASRProvider.transcribe.mockResolvedValue(mockResponse);
        const result = await client.audio.transcriptions.create(
          request,
          options
        );

        expect(mockOpenAIASRProvider.transcribe).toHaveBeenCalledWith(
          { ...request, model: "whisper-1" },
          options
        );
        expect(result).toEqual(mockResponse);
      });
    });
  });

  describe("listProviders", () => {
    // All providers are now always registered (for env var support)
    const allChatProviders = ["openai", "anthropic", "mistral", "groq", "google"];

    it("should return all providers (always registered for env var support)", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        groq: { apiKey: "groq-key" },
      };

      const client = new Client(config);

      expect(client.listProviders()).toEqual(allChatProviders);
    });

    it("should return all providers even with empty config", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.listProviders()).toEqual(allChatProviders);
    });
  });

  describe("listASRProviders", () => {
    const allASRProviders = ["openai", "deepgram"];

    it("should return all ASR providers (always registered for env var support)", () => {
      const config: ProviderConfigs = {
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      expect(client.listASRProviders()).toEqual(allASRProviders);
    });

    it("should return all ASR providers even with empty config", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.listASRProviders()).toEqual(allASRProviders);
    });
  });

  describe("isProviderConfigured", () => {
    it("should return true for all known providers (always registered)", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        anthropic: { apiKey: "anthropic-key" },
      };

      const client = new Client(config);

      // All known providers return true (they're always registered)
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(true);
      expect(client.isProviderConfigured("mistral")).toBe(true);
      expect(client.isProviderConfigured("groq")).toBe(true);
      expect(client.isProviderConfigured("google")).toBe(true);
    });

    it("should return false for unknown providers", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
      };

      const client = new Client(config);

      expect(client.isProviderConfigured("unknown")).toBe(false);
      expect(client.isProviderConfigured("nonexistent")).toBe(false);
    });
  });

  describe("isASRProviderConfigured", () => {
    it("should return true for all known ASR providers (always registered)", () => {
      const config: ProviderConfigs = {
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
      expect(client.isASRProviderConfigured("openai")).toBe(true);
    });

    it("should return false for unknown ASR providers", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.isASRProviderConfigured("unknown")).toBe(false);
      expect(client.isASRProviderConfigured("nonexistent")).toBe(false);
    });
  });
});
