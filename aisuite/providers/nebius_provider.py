import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class NebiusProvider(Provider):
    """
    Nebius AI Studio Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.studio.nebius.ai/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the Nebius AI Studio provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("NEBIUS_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Nebius AI Studio API key is missing. Please provide it in the config or set the NEBIUS_API_KEY environment variable. You can get your API key at https://studio.nebius.ai/settings/api-keys"
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Nebius AI Studio chat completions endpoint using httpx.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            # Make the request to the Nebius AI Studio endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Nebius AI Studio request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["choices"][0][
            "message"
        ]["content"]
        return normalized_response
