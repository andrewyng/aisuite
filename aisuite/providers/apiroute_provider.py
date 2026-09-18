"""API Route provider for aisuite."""

import os

import openai

from aisuite.provider import LLMError, Provider
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class ApirouteProvider(Provider):
    """Provider for API Route's OpenAI-compatible API."""

    def __init__(self, **config):
        """Initialize API Route with an API key and optional base URL override."""
        config.setdefault("api_key", os.getenv("API_ROUTE_API_KEY"))
        config.setdefault(
            "base_url",
            os.getenv("API_ROUTE_BASE_URL", "https://global.api-route.com/v1"),
        )

        if not config["api_key"]:
            raise ValueError(
                "API Route API key is missing. Please provide it in the config or "
                "set the API_ROUTE_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()
        super().__init__()

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion through API Route."""
        try:
            transformed_messages = self.transformer.convert_request(messages)
            return self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
