import os
import openai

from aisuite.provider import Provider, LLMError


class TongyiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Tongyi provider with the given configuration.
        Pass the entire configuration dictionary to the Tongyi client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("TONGYI_API_KEY"))
        config["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not config["api_key"]:
            raise ValueError(
                "Tongyi API key is missing. Please provide it in the config or set the TONGYI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the Tongyi API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
