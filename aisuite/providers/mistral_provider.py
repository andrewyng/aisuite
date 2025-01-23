import os

from mistralai import Mistral

from aisuite.provider import Provider


class MistralProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Mistral provider with the given configuration.
        Pass the entire configuration dictionary to the Mistral client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                " API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            )
        self.client = Mistral(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.complete(model=model, messages=messages, **kwargs)

    async def chat_completions_create_async(self, model, messages, **kwargs):
        return await self.client.chat.complete_async(model=model, messages=messages, **kwargs)