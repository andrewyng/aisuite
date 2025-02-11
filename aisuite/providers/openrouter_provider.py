import os
import openai
from aisuite.provider import Provider


class OpenrouterProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenRouter provider with the given configuration.
        Pass the entire configuration dictionary to the OpenRouter client constructor.
        """

        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENROUTER_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenRouter API key is missing. Please provide it in the config or set the OPENROUTER_API_KEY environment variable."
            )

        # Set the base URL for the OpenRouter API
        config["base_url"] = "https://openrouter.ai/api/v1"

        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
