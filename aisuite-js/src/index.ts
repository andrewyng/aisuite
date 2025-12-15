export { Client } from "./client";
export * from "./types";
export * from "./core/errors";
export { parseModel } from "./core/model-parser";

// NOTE: Providers are NOT re-exported here to avoid side-effect conflicts
// between SDK libraries (e.g., @mistralai/mistralai conflicts with @google/genai streaming).
// If you need to use providers directly, import them individually:
//   import { GoogleProvider } from 'aisuite/providers/google';
//   import { OpenAIProvider } from 'aisuite/providers/openai';
// etc.
