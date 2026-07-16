import { LLMConfig } from '../types/chat';

export const configuredLLMs: LLMConfig[] = [
  {
    name: "OpenAI GPT-4o",
    provider: "openai",
    model: "gpt-4o"
  },
  {
    name: "OpenAI GPT-4o Mini",
    provider: "openai",
    model: "gpt-4o-mini"
  },
  {
    name: "Anthropic Claude 3.5 Sonnet",
    provider: "anthropic",
    model: "claude-3-5-sonnet-20241022"
  },
  {
    name: "Anthropic Claude 3 Haiku",
    provider: "anthropic",
    model: "claude-3-haiku-20240307"
  },
  {
    name: "Groq Llama 3.3-70b Versatile",
    provider: "groq",
    model: "llama-3.3-70b-versatile"
  },
  {
    name: "Groq Mixtral 24B",
    provider: "groq",
    model: "mistral-saba-24b"
  },
  {
    name: "Groq Gemma 2 9B",
    provider: "groq",
    model: "gemma2-9b-it"
  },
  {
    name: "Mistral Medium",
    provider: "mistral",
    model: "mistral-medium"
  },
  {
    name: "Mistral Large",
    provider: "mistral",
    model: "mistral-large-latest"
  }
];

export const getLLMConfigByName = (name: string): LLMConfig | undefined => {
  return configuredLLMs.find(llm => llm.name === name);
};

export const getLLMConfigByProviderAndModel = (provider: string, model: string): LLMConfig | undefined => {
  return configuredLLMs.find(llm => llm.provider === provider && llm.model === model);
}; 