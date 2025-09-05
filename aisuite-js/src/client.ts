import {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ProviderConfigs,
  RequestOptions,
  TranscriptionRequest,
  TranscriptionResult,
} from "./types";
import { BaseProvider } from "./core/base-provider";
import { BaseASRProvider } from "./core/base-asr-provider";
import { parseModel } from "./core/model-parser";
import { ProviderNotConfiguredError } from "./core/errors";
import { OpenAIProvider } from "./providers/openai";
import { AnthropicProvider } from "./providers/anthropic";
import { MistralProvider } from "./providers/mistral";
import { GroqProvider } from "./providers/groq";
import { DeepgramASRProvider } from "./asr-providers/deepgram";
import { OpenAIASRProvider } from "./asr-providers/openai";

export class Client {
  private chatProviders: Map<string, BaseProvider> = new Map();
  private asrProviders: Map<string, BaseASRProvider> = new Map();

  constructor(config: ProviderConfigs) {
    this.initializeProviders(config);
  }

  private initializeProviders(config: ProviderConfigs): void {
    if (config.openai) {
      if (config.openai.audio) {
        this.asrProviders.set("openai", new OpenAIASRProvider(config.openai));
      } else {
        this.chatProviders.set("openai", new OpenAIProvider(config.openai));
      }
    }

    if (config.anthropic) {
      this.chatProviders.set(
        "anthropic",
        new AnthropicProvider(config.anthropic)
      );
    }

    if (config.mistral) {
      this.chatProviders.set("mistral", new MistralProvider(config.mistral));
    }

    if (config.groq) {
      this.chatProviders.set("groq", new GroqProvider(config.groq));
    }

    if (config.deepgram) {
      this.asrProviders.set(
        "deepgram",
        new DeepgramASRProvider(config.deepgram)
      );
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
        const providerInstance = this.chatProviders.get(provider);

        if (!providerInstance) {
          throw new ProviderNotConfiguredError(
            provider,
            Array.from(this.chatProviders.keys())
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

  public audio = {
    transcriptions: {
      create: async (
        request: TranscriptionRequest,
        options?: RequestOptions
      ): Promise<TranscriptionResult> => {
        const { provider, model } = parseModel(request.model);
        const providerInstance = this.asrProviders.get(provider);

        if (!providerInstance) {
          throw new ProviderNotConfiguredError(
            provider,
            Array.from(this.asrProviders.keys())
          );
        }

        const requestWithParsedModel = {
          ...request,
          model, // Just the model name without provider prefix
        };

        return providerInstance.transcribe(requestWithParsedModel, options);
      },
    },
  };

  public listProviders(): string[] {
    return Array.from(this.chatProviders.keys());
  }

  public listASRProviders(): string[] {
    return Array.from(this.asrProviders.keys());
  }

  public isProviderConfigured(provider: string): boolean {
    return this.chatProviders.has(provider);
  }

  public isASRProviderConfigured(provider: string): boolean {
    return this.asrProviders.has(provider);
  }
}
