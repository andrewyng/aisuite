import openai
import os
from aisuite.provider import Provider


class VLLMProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the vLLM provider with the given configuration.
        """
        config.setdefault("api_key", os.getenv("VLLM_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "vLLM API key is missing. Please provide it in the config or set the VLLM_API_KEY environment variable."
            )
        config.setdefault("base_url", os.getenv("VLLM_API_BASE_URL"))
        if not config["base_url"]:
            raise ValueError(
                "vLLM API base URL is missing. Please provide it in the config or set the VLLM_API_BASE_URL environment variable."
            )
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
