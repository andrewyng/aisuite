import httpx
import os
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse

class BaiduProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the provider with the given configuration.
        The token is fetched from the config or environment variables.
        """
        # Ensure API key is provided either in config or via environment variable
        self.token = config.get("token") or os.getenv("BAIDU_TOKEN")
        if not self.token:
            raise ValueError(
                "Baidu token is missing. Please provide it in the config or set the BAIDU_TOKEN environment variable."
            )
        self.timeout = config.get("timeout", 200)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Inference API endpoint using httpx.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }
        url = "https://qianfan.baidubce.com/v2/chat/completions"
        try:
            # Make the request to Baidu Qianfan endpoint.
            response = httpx.post(url, json=data, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Baidu qianfan request failed: {http_err}")
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
