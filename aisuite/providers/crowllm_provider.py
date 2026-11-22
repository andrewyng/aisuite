"""CrowLLM provider for aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


# pylint: disable=too-few-public-methods
class CrowllmProvider(Provider):
    """Provider for CrowLLM, an OpenAI-compatible multi-model gateway."""

    def __init__(self, **config):
        """
        Initialize the CrowLLM provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CROWLLM_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "CrowLLM API key is missing. Please provide it in the config or "
                "set the CROWLLM_API_KEY environment variable."
            )
        config.setdefault("base_url", "https://crowllm.com/v1")

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        # CrowLLM serves the OpenAI Chat Completions format as-is.
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the gateway
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
