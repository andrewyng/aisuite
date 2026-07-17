import openai
import os
from aisuite.provider import Provider, LLMError


class InceptionProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Inception provider with the given configuration.
        Pass the entire configuration dictionary to the Inception client constructor using openai.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("INCEPTION_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Inception API key is missing. Please provide it in the config or set the INCEPTION_API_KEY environment variable."
            )
        config["base_url"] = "https://api.inceptionlabs.ai/v1"

        # Pass the entire config to the Inception client constructor using openai
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Inception will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the Inception API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
