import os
from aisuite.provider import Provider
from openai import Client

BASE_URL = "https://api.inference.crusoecloud.com/v1"


class CrusoeProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Crusoe provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CRUSOE_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Crusoe API key is missing. Please provide it in the config or set the CRUSOE_API_KEY environment variable. You can get your API key from the Crusoe Cloud Console at https://console.crusoecloud.com/"
            )

        config["base_url"] = BASE_URL
        # Pass the entire config to the OpenAI client constructor
        self.client = Client(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Crusoe API
        )
