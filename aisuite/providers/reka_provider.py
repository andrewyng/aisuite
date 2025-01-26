from reka.client import Reka
import os
from aisuite.provider import Provider, LLMError


class RekaProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Reka provider with the given configuration.
        Pass the entire configuration dictionary to the Reka client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("REKA_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Reka API key is missing. Please provide it in the config or set the REKA_API_KEY environment variable."
            )

        # Pass the entire config to the Reka client constructor
        self.client = Reka(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Reka will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        return self.client.chat.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Reka API
        )
