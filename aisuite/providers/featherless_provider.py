import os
import openai
from aisuite.provider import Provider


class FeatherlessProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Featherless provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("FEATHERLESS_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Featherless API key is missing. Please provide it in the config or set the FEATHERLESS_API_KEY environment variable."
            )
        config["base_url"] = "https://api.featherless.ai/v1"
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to Featherless using the official client.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Featherless API
        )

        return response
