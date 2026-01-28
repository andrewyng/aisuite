"""Minimax provider for the aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class MinimaxProvider(Provider):
    """Provider for Minimax using OpenAI-compatible API."""

    def __init__(self, **config):
        """
        Initialize the Minimax provider with the given configuration.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MINIMAX_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Minimax API key is missing. Please provide it in the config or "
                "set the MINIMAX_API_KEY environment variable."
            )

        config.setdefault(
            "base_url", os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        )

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion using Minimax's OpenAI-compatible API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
