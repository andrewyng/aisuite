export class AISuiteError extends Error {
  constructor(
    message: string,
    public provider: string,
    public code?: string,
    public statusCode?: number
  ) {
    super(message);
    this.name = "AISuiteError";
  }
}

export class ProviderNotConfiguredError extends AISuiteError {
  constructor(provider: string, availableProviders: string[]) {
    super(
      `Provider '${provider}' not configured. Available: ${availableProviders.join(
        ", "
      )}`,
      provider,
      "PROVIDER_NOT_CONFIGURED"
    );
  }
}

export class InvalidModelFormatError extends AISuiteError {
  constructor(model: string) {
    super(
      `Invalid model format: ${model}. Expected "provider:model"`,
      "unknown",
      "INVALID_MODEL_FORMAT"
    );
  }
}

export class ToolCallError extends AISuiteError {
  constructor(message: string, provider: string) {
    super(message, provider, "TOOL_CALL_ERROR");
  }
}

export class AudioProcessingError extends AISuiteError {
  constructor(message: string, provider: string) {
    super(message, provider, "AUDIO_PROCESSING_ERROR");
    this.name = "AudioProcessingError";
  }
}

export class UnsupportedParameterError extends AISuiteError {
  constructor(parameter: string, provider: string) {
    super(
      `Parameter '${parameter}' is not supported by provider '${provider}'`,
      provider,
      'UNSUPPORTED_PARAMETER'
    );
    this.name = 'UnsupportedParameterError';
  }
}
