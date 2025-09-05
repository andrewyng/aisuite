export { Client } from "./client";
export * from "./types";
export * from "./core/errors";
export { parseModel } from "./core/model-parser";

// Re-export providers for advanced usage
export {
  OpenAIProvider,
  AnthropicProvider,
  GroqProvider,
  MistralProvider,
} from "./providers";

export {
  DeepgramASRProvider,
  OpenAIASRProvider,
} from "./asr-providers";
