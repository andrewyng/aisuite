"""ModelsLab provider for aisuite.

ModelsLab provides uncensored chat via an OpenAI-compatible endpoint.
API docs: https://docs.modelslab.com
API key: https://modelslab.com/account/api-key
"""

import os

import openai

from aisuite.provider import LLMError, Provider


class ModelsLabProvider(Provider):
    """Provider for ModelsLab's OpenAI-compatible chat API."""

    BASE_URL = "https://modelslab.com/api/uncensored-chat/v1"

    def __init__(self, **config):
        """Initialize the ModelsLab provider.

        Args:
            **config: Configuration kwargs. Accepts `api_key` and any other
                kwargs supported by `openai.OpenAI`. If `api_key` is not
                provided, reads from the ``MODELSLAB_API_KEY`` environment
                variable.
        """
        config.setdefault("api_key", os.getenv("MODELSLAB_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "ModelsLab API key is missing. Please provide it in the config "
                "or set the MODELSLAB_API_KEY environment variable."
            )
        config["base_url"] = self.BASE_URL
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion using ModelsLab's API.

        Args:
            model: The model ID (e.g. "meta-llama/Meta-Llama-3-8B-Instruct").
            messages: List of message dicts in OpenAI format.
            **kwargs: Additional kwargs forwarded to the API.

        Returns:
            The API response object.

        Raises:
            LLMError: If the API call fails.
        """
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"ModelsLab API error: {exc}") from exc
