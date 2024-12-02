import os

from cerebras.cloud.sdk import Cerebras
from aisuite.provider import Provider


class CerebrasProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Cerebras provider with the given configuration.
        Pass the entire configuration dictionary to the Cerebras client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CEREBRAS_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                " API key is missing. Please provide it in the config or set the CEREBRAS_API_KEY environment variable."
            )
        self.client = Cerebras(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Cerebras API
        )
