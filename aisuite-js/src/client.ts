import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ProviderConfigs,
  RequestOptions,
  TranscriptionRequest,
  TranscriptionResult,
} from "./types";
import { Provider } from "./core/base-provider";
import { ASRProvider } from "./core/base-asr-provider";
import { parseModel } from "./core/model-parser";
import { ProviderNotConfiguredError } from "./core/errors";

// Type for provider factory functions
type ProviderFactory = () => Promise<Provider>;
type ASRProviderFactory = () => Promise<ASRProvider>;

export class Client {
  private chatProviders: Map<string, Provider> = new Map();
  private asrProviders: Map<string, ASRProvider> = new Map();
  private chatProviderFactories: Map<string, ProviderFactory> = new Map();
  private asrProviderFactories: Map<string, ASRProviderFactory> = new Map();
  private config: ProviderConfigs;

  constructor(config: ProviderConfigs = {}) {
    this.config = config;
    this.registerProviderFactories();
  }

  private registerProviderFactories(): void {
    // Register factories for ALL providers - they are lazily loaded when first used.
    // If config is provided, it's used; otherwise providers will read API keys from env vars.

    // OpenAI is both a chat provider and ASR provider
    this.chatProviderFactories.set("openai", async () => {
      const { OpenAIProvider } = await import("./providers/openai");
      return new OpenAIProvider(this.config.openai || {});
    });
    this.asrProviderFactories.set("openai", async () => {
      const { OpenAIProvider } = await import("./providers/openai");
      return new OpenAIProvider(this.config.openai || {});
    });

    this.chatProviderFactories.set("anthropic", async () => {
      const { AnthropicProvider } = await import("./providers/anthropic");
      return new AnthropicProvider(this.config.anthropic || {});
    });

    this.chatProviderFactories.set("mistral", async () => {
      const { MistralProvider } = await import("./providers/mistral");
      return new MistralProvider(this.config.mistral || {});
    });

    this.chatProviderFactories.set("groq", async () => {
      const { GroqProvider } = await import("./providers/groq");
      return new GroqProvider(this.config.groq || {});
    });

    this.chatProviderFactories.set("google", async () => {
      const { GoogleProvider } = await import("./providers/google");
      return new GoogleProvider(this.config.google || {});
    });

    this.asrProviderFactories.set("deepgram", async () => {
      const { DeepgramASRProvider } = await import("./asr-providers/deepgram");
      return new DeepgramASRProvider(this.config.deepgram || {});
    });
  }

  private async getChatProvider(provider: string): Promise<Provider> {
    // Return cached provider if available
    const cached = this.chatProviders.get(provider);
    if (cached) return cached;

    // Create provider lazily
    const factory = this.chatProviderFactories.get(provider);
    if (!factory) {
      throw new ProviderNotConfiguredError(
        provider,
        Array.from(this.chatProviderFactories.keys())
      );
    }

    const instance = await factory();
    this.chatProviders.set(provider, instance);
    return instance;
  }

  private async getASRProvider(provider: string): Promise<ASRProvider> {
    // Return cached provider if available
    const cached = this.asrProviders.get(provider);
    if (cached) return cached;

    // Create provider lazily
    const factory = this.asrProviderFactories.get(provider);
    if (!factory) {
      throw new ProviderNotConfiguredError(
        provider,
        Array.from(this.asrProviderFactories.keys())
      );
    }

    const instance = await factory();
    this.asrProviders.set(provider, instance);
    return instance;
  }

  public chat = {
    completions: {
      create: async (
        request: ChatCompletionRequest,
        options?: RequestOptions
      ): Promise<
        ChatCompletionResponse | AsyncIterable<ChatCompletionChunk>
      > => {
        const { provider, model } = parseModel(request.model);
        const providerInstance = await this.getChatProvider(provider);

        const requestWithParsedModel = {
          ...request,
          model, // Just the model name without provider prefix
        };

        if (request.stream) {
          return providerInstance.streamChatCompletion(
            requestWithParsedModel,
            options
          );
        } else {
          return providerInstance.chatCompletion(
            requestWithParsedModel,
            options
          );
        }
      },
    },
  };

  public audio = {
    transcriptions: {
      create: async (
        request: TranscriptionRequest,
        options?: RequestOptions
      ): Promise<TranscriptionResult> => {
        const { provider, model } = parseModel(request.model);
        const providerInstance = await this.getASRProvider(provider);

        const requestWithParsedModel = {
          ...request,
          model, // Just the model name without provider prefix
        };

        return providerInstance.transcribe(requestWithParsedModel, options);
      },
    },
  };

  public listProviders(): string[] {
    return Array.from(this.chatProviderFactories.keys());
  }

  public listASRProviders(): string[] {
    return Array.from(this.asrProviderFactories.keys());
  }

  public isProviderConfigured(provider: string): boolean {
    return this.chatProviderFactories.has(provider);
  }

  public isASRProviderConfigured(provider: string): boolean {
    return this.asrProviderFactories.has(provider);
  }
}
