import os

from zhipuai import ZhipuAI
from aisuite.provider import Provider


class ZhipuaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the ZhipuAI provider with the given configuration.
        Pass the entire configuration dictionary to the ZhipuAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("ZHIPUAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "API key is missing. Please provide it in the config or set the ZHIPUAI_API_KEY environment variable."
            )
        self.client = ZhipuAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the ZhipuAI API
        )
