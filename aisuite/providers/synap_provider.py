"""Synap provider for the aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


# pylint: disable=too-few-public-methods
class SynapProvider(Provider):
    """Provider for Synap (Linkrra's OpenAI-compatible inference pool)."""

    def __init__(self, **config):
        """
        Initialize the Synap provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        config.setdefault("api_key", os.getenv("SYNAP_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Synap API key is missing. Please provide it in the config or "
                "set the SYNAP_API_KEY environment variable."
            )
        config["base_url"] = "https://pool.linkrra.com/v1"

        self.client = openai.OpenAI(**config)
        # Synap's response format is the same as OpenAI's.
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
