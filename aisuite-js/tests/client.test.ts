import { Client } from "../src/client";
import {
  ProviderConfigs,
  ChatCompletionRequest,
  TranscriptionRequest,
} from "../src/types";
import { ProviderNotConfiguredError } from "../src/core/errors";

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
jest.mock("../src/asr-providers/deepgram");
jest.mock("../src/asr-providers/openai");

describe("Client", () => {
  let mockOpenAIProvider: any;
  let mockAnthropicProvider: any;
  let mockMistralProvider: any;
  let mockGroqProvider: any;
  let mockDeepgramProvider: any;
  let mockOpenAIASRProvider: any;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Create mock provider instances
    mockOpenAIProvider = {
      chatCompletion: jest.fn(),
      streamChatCompletion: jest.fn(),
    };

    mockAnthropicProvider = {
      chatCompletion: jest.fn(),
      streamChatCompletion: jest.fn(),
    };

    mockMistralProvider = {
      chatCompletion: jest.fn(),
      streamChatCompletion: jest.fn(),
    };

    mockGroqProvider = {
      chatCompletion: jest.fn(),
      streamChatCompletion: jest.fn(),
    };

    mockDeepgramProvider = {
      transcribe: jest.fn(),
    };

    mockOpenAIASRProvider = {
      transcribe: jest.fn(),
    };

    // Mock the provider constructors
    const openaiModule = require("../src/providers/openai");
    const anthropicModule = require("../src/providers/anthropic");
    const mistralModule = require("../src/providers/mistral");
    const groqModule = require("../src/providers/groq");
    const deepgramModule = require("../src/asr-providers/deepgram");
    const opsnaiasrModule = require("../src/asr-providers/openai");

    openaiModule.OpenAIProvider.mockImplementation(() => mockOpenAIProvider);
    anthropicModule.AnthropicProvider.mockImplementation(
      () => mockAnthropicProvider
    );
    mistralModule.MistralProvider.mockImplementation(() => mockMistralProvider);
    groqModule.GroqProvider.mockImplementation(() => mockGroqProvider);
    deepgramModule.DeepgramASRProvider.mockImplementation(
      () => mockDeepgramProvider
    );
    opsnaiasrModule.OpenAIASRProvider.mockImplementation(
      () => mockOpenAIASRProvider
    );
  });

  describe("constructor", () => {
    it("should initialize providers based on config", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        anthropic: { apiKey: "anthropic-key" },
        mistral: { apiKey: "mistral-key" },
        groq: { apiKey: "groq-key" },
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      expect(client.listProviders()).toEqual([
        "openai",
        "anthropic",
        "mistral",
        "groq",
      ]);
      expect(client.listASRProviders()).toEqual(["deepgram"]);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(true);
      expect(client.isProviderConfigured("mistral")).toBe(true);
      expect(client.isProviderConfigured("groq")).toBe(true);
      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
    });

    it("should only initialize configured providers", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        groq: { apiKey: "groq-key" },
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      expect(client.listProviders()).toEqual(["openai", "groq"]);
      expect(client.listASRProviders()).toEqual(["deepgram"]);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(false);
      expect(client.isProviderConfigured("mistral")).toBe(false);
      expect(client.isProviderConfigured("groq")).toBe(true);
      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
      expect(client.isASRProviderConfigured("unknown")).toBe(false);
    });

    it("should handle empty config", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.listProviders()).toEqual([]);
      expect(client.listASRProviders()).toEqual([]);
      expect(client.isProviderConfigured("openai")).toBe(false);
      expect(client.isASRProviderConfigured("deepgram")).toBe(false);
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
          transcribe: jest.fn(),
        };

        const { OpenAIASRProvider } = require("../src/asr-providers/openai");
        OpenAIASRProvider.mockImplementation(() => mockOpenAIASRProvider);
      });

      it("should not be available without audio config", async () => {
        client = new Client({
          openai: { apiKey: "openai-key" },
        });

        const request: TranscriptionRequest = {
          model: "openai:whisper-1",
          file: Buffer.from("test audio"),
        };

        await expect(
          client.audio.transcriptions.create(request)
        ).rejects.toThrow(ProviderNotConfiguredError);
      });

      it("should transcribe with audio enabled", async () => {
        client = new Client({
          openai: {
            apiKey: "openai-key",
            audio: true,
          },
        });

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
            apiKey: "openai-key",
            audio: true,
          },
        });

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
            apiKey: "openai-key",
            audio: true,
          },
        });

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
    it("should return list of configured providers", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        groq: { apiKey: "groq-key" },
      };

      const client = new Client(config);

      expect(client.listProviders()).toEqual(["openai", "groq"]);
    });

    it("should return empty array when no providers configured", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.listProviders()).toEqual([]);
    });
  });

  describe("listASRProviders", () => {
    it("should return list of configured ASR providers", () => {
      const config: ProviderConfigs = {
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      expect(client.listASRProviders()).toEqual(["deepgram"]);
    });

    it("should return empty array when no ASR providers configured", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.listASRProviders()).toEqual([]);
    });
  });

  describe("isProviderConfigured", () => {
    it("should return true for configured providers", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        anthropic: { apiKey: "anthropic-key" },
      };

      const client = new Client(config);

      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(true);
    });

    it("should return false for unconfigured providers", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
      };

      const client = new Client(config);

      expect(client.isProviderConfigured("anthropic")).toBe(false);
      expect(client.isProviderConfigured("mistral")).toBe(false);
      expect(client.isProviderConfigured("groq")).toBe(false);
    });
  });

  describe("isASRProviderConfigured", () => {
    it("should return true for configured ASR providers", () => {
      const config: ProviderConfigs = {
        deepgram: { apiKey: "deepgram-key" },
      };

      const client = new Client(config);

      expect(client.isASRProviderConfigured("deepgram")).toBe(true);
    });

    it("should return false for unconfigured ASR providers", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.isASRProviderConfigured("deepgram")).toBe(false);
      expect(client.isASRProviderConfigured("unknown")).toBe(false);
    });
  });
});
