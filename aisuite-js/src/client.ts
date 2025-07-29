import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ProviderConfigs,
  RequestOptions,
} from "./types";
import { BaseProvider } from "./core/base-provider";
import { parseModel } from "./core/model-parser";
import { ProviderNotConfiguredError } from "./core/errors";
import { OpenAIProvider } from "./providers/openai";
import { AnthropicProvider } from "./providers/anthropic";
import { MistralProvider } from "./providers/mistral";
import { GroqProvider } from "./providers/groq";

export class Client {
  private providers: Map<string, BaseProvider> = new Map();

  constructor(config: ProviderConfigs) {
    this.initializeProviders(config);
  }

  private initializeProviders(config: ProviderConfigs): void {
    if (config.openai) {
      this.providers.set("openai", new OpenAIProvider(config.openai));
    }

    if (config.anthropic) {
      this.providers.set("anthropic", new AnthropicProvider(config.anthropic));
    }

    if (config.mistral) {
      this.providers.set("mistral", new MistralProvider(config.mistral));
    }

    if (config.groq) {
      this.providers.set("groq", new GroqProvider(config.groq));
    }
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
        const providerInstance = this.providers.get(provider);

        if (!providerInstance) {
          throw new ProviderNotConfiguredError(
            provider,
            Array.from(this.providers.keys())
          );
        }

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

  public listProviders(): string[] {
    return Array.from(this.providers.keys());
  }

  public isProviderConfigured(provider: string): boolean {
    return this.providers.has(provider);
  }
}
