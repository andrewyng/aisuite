import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class FireworksProvider(Provider):
    """
    Fireworks AI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the Fireworks provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("FIREWORKS_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Fireworks API key is missing. Please provide it in the config or set the FIREWORKS_API_KEY environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Fireworks AI chat completions endpoint using httpx.
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
            # Make the request to Fireworks AI endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Fireworks AI request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        return self._normalize_response(response.json())

    async def chat_completions_create_async(self, model, messages, **kwargs):
        """
        Makes an async request to the Fireworks AI chat completions endpoint.
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

        async with httpx.AsyncClient() as client:
            try:
                # Make the async request to Fireworks AI endpoint
                response = await client.post(
                    self.BASE_URL, json=data, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as http_err:
                raise LLMError(f"Fireworks AI request failed: {http_err}")
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
