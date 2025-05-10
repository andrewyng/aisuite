import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter

class LMStudioProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the LM Studio provider.  Accepts the same config keys
        as OpenAI (api_key, organization, etc.), plus:
          - base_url: defaults to http://localhost:1234/v1
        """
        # 1. Ensure API key (LM Studio doesn’t enforce it, but OpenAI SDK requires non-empty)
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "API key is missing. Set OPENAI_API_KEY or pass `api_key=` in config."
            )

        # 2. Default base_url to localhost LM Studio server
        config.setdefault("base_url", "http://localhost:1234/v1")

        # 3. Initialize the OpenAI-compatible client pointed at LM Studio
        self.client = openai.OpenAI(**config)

        # 4. Use same message transformer as OpenAI
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Proxy to LM Studio’s /chat/completions endpoint.
        """
        try:
            transformed = self.transformer.convert_request(messages)
            return self.client.chat.completions.create(
                model=model,
                messages=transformed,
                **kwargs
            )
        except Exception as e:
            raise LLMError(f"LMStudio chat error: {e}")

    def completions_create(self, model, prompt, **kwargs):
        """
        Proxy to LM Studio’s /completions endpoint.
        """
        try:
            return self.client.completions.create(
                model=model,
                prompt=prompt,
                **kwargs
            )
        except Exception as e:
            raise LLMError(f"LMStudio completion error: {e}")

    def embeddings_create(self, model, input, **kwargs):
        """
        Proxy to LM Studio’s /embeddings endpoint.
        """
        try:
            return self.client.embeddings.create(
                model=model,
                input=input,
                **kwargs
            )
        except Exception as e:
            raise LLMError(f"LMStudio embedding error: {e}")
