"""LiteLLM provider for aisuite — access 300+ LLM providers through a single interface."""

import os
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class LitellmProvider(Provider):
    """Provider that routes completions through LiteLLM.

    Users can pass any LiteLLM-supported model string as the model name
    (e.g. ``anthropic/claude-sonnet-4-6``, ``vertex_ai/gemini-2.5-pro``,
    ``bedrock/anthropic.claude-3-haiku``).

    When running against a self-hosted LiteLLM proxy, set ``base_url``
    in the provider config and supply the proxy master key via
    ``api_key`` or the ``LITELLM_API_KEY`` environment variable.
    """

    def __init__(self, **config):
        self.api_key = config.pop("api_key", None) or os.getenv("LITELLM_API_KEY")
        self.base_url = config.pop("base_url", None) or os.getenv("LITELLM_BASE_URL")
        self.drop_params = config.pop("drop_params", True)
        self.extra_config = config
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "The litellm package is required for the LiteLLM provider. "
                "Install it with: pip install 'aisuite[litellm]'"
            )

        try:
            transformed_messages = self.transformer.convert_request(messages)

            call_kwargs = {
                "model": model,
                "messages": transformed_messages,
                "drop_params": self.drop_params,
                **self.extra_config,
                **kwargs,
            }
            if self.api_key:
                call_kwargs["api_key"] = self.api_key
            if self.base_url:
                call_kwargs["api_base"] = self.base_url

            response = litellm.completion(**call_kwargs)
            return self.transformer.convert_response(response.model_dump())
        except ImportError:
            raise
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e

    async def achat_completions_create(self, model, messages, **kwargs):
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "The litellm package is required for the LiteLLM provider. "
                "Install it with: pip install 'aisuite[litellm]'"
            )

        try:
            transformed_messages = self.transformer.convert_request(messages)

            call_kwargs = {
                "model": model,
                "messages": transformed_messages,
                "drop_params": self.drop_params,
                **self.extra_config,
                **kwargs,
            }
            if self.api_key:
                call_kwargs["api_key"] = self.api_key
            if self.base_url:
                call_kwargs["api_base"] = self.base_url

            response = await litellm.acompletion(**call_kwargs)
            return self.transformer.convert_response(response.model_dump())
        except ImportError:
            raise
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
