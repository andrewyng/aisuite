import { Client } from "../src/client";
import { ProviderConfigs, ChatCompletionRequest } from "../src/types";
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

describe("Client", () => {
  let mockOpenAIProvider: any;
  let mockAnthropicProvider: any;
  let mockMistralProvider: any;
  let mockGroqProvider: any;

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

    // Mock the provider constructors
    const openaiModule = require("../src/providers/openai");
    const anthropicModule = require("../src/providers/anthropic");
    const mistralModule = require("../src/providers/mistral");
    const groqModule = require("../src/providers/groq");

    openaiModule.OpenAIProvider.mockImplementation(() => mockOpenAIProvider);
    anthropicModule.AnthropicProvider.mockImplementation(
      () => mockAnthropicProvider
    );
    mistralModule.MistralProvider.mockImplementation(() => mockMistralProvider);
    groqModule.GroqProvider.mockImplementation(() => mockGroqProvider);
  });

  describe("constructor", () => {
    it("should initialize providers based on config", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        anthropic: { apiKey: "anthropic-key" },
        mistral: { apiKey: "mistral-key" },
        groq: { apiKey: "groq-key" },
      };

      const client = new Client(config);

      expect(client.listProviders()).toEqual([
        "openai",
        "anthropic",
        "mistral",
        "groq",
      ]);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(true);
      expect(client.isProviderConfigured("mistral")).toBe(true);
      expect(client.isProviderConfigured("groq")).toBe(true);
    });

    it("should only initialize configured providers", () => {
      const config: ProviderConfigs = {
        openai: { apiKey: "openai-key" },
        groq: { apiKey: "groq-key" },
      };

      const client = new Client(config);

      expect(client.listProviders()).toEqual(["openai", "groq"]);
      expect(client.isProviderConfigured("openai")).toBe(true);
      expect(client.isProviderConfigured("anthropic")).toBe(false);
      expect(client.isProviderConfigured("mistral")).toBe(false);
      expect(client.isProviderConfigured("groq")).toBe(true);
    });

    it("should handle empty config", () => {
      const config: ProviderConfigs = {};

      const client = new Client(config);

      expect(client.listProviders()).toEqual([]);
      expect(client.isProviderConfigured("openai")).toBe(false);
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
});
