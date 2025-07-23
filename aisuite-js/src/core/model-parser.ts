import { InvalidModelFormatError } from './errors';

export interface ParsedModel {
  provider: string;
  model: string;
}

export function parseModel(model: string): ParsedModel {
  if (!model || typeof model !== 'string') {
    throw new InvalidModelFormatError(model);
  }

  const [provider, ...modelParts] = model.split(':');
  
  if (!provider || modelParts.length === 0) {
    throw new InvalidModelFormatError(model);
  }
  
  return {
    provider,
    model: modelParts.join(':') // Handle cases like "openai:gpt-4:vision"
  };
}