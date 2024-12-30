import os
from aisuite.provider import Provider
from openai import OpenAI


class SambanovaProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the SambaNova provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("SAMBANOVA_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Sambanova API key is missing. Please provide it in the config or set the SAMBANOVA_API_KEY environment variable."
            )

        config["base_url"] = "https://api.sambanova.ai/v1/"
        # Pass the entire config to the OpenAI client constructor
        self.client = OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Sambanova will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Sambanova API
        )
