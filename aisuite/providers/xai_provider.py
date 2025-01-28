import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class XaiMessageConverter(OpenAICompliantMessageConverter):
    """
    xAI-specific message converter if needed
    """

    pass


class XaiProvider(Provider):
    """
    xAI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the xAI provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("XAI_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "xAI API key is missing. Please provide it in the config or set the XAI_API_KEY environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)
        self.transformer = XaiMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the xAI chat completions endpoint using httpx.
        """
        # Transform messages using converter
        transformed_messages = self.transformer.convert_request(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": transformed_messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            # Make the request to xAI endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return self.transformer.convert_response(response.json())
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"xAI request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
