import os
import openai

from aisuite.provider import Provider, LLMError


class VolcanoProvider(Provider):
    """Provider for Volcano Engine's OpenAI-compatible API."""

    def __init__(self, **config):
        """
        Initialize the Volcano Engine provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        config.setdefault(
            "api_key",
            os.getenv("VOLCANO_API_KEY") or os.getenv("VOLCENGINE_API_KEY"),
        )
        config.setdefault("base_url", "https://ark.cn-beijing.volces.com/api/v3")

        if not config["api_key"]:
            raise ValueError(
                "Volcano API key is missing. Please provide it in the config or set "
                "the VOLCANO_API_KEY or VOLCENGINE_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
