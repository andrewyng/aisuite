import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls instead of using SDK.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, **config):
        """
        Initialize the Ollama provider with the given configuration.
        """
        self.url = config.get("api_url") or os.getenv(
            "OLLAMA_API_URL", "http://localhost:11434"
        )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the chat completions endpoint using httpx.
        """
        kwargs["stream"] = False
        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError:  # Handle connection errors
            raise LLMError(f"Connection failed: {self._CONNECT_ERROR_MESSAGE}")
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Ollama request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the API response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["message"][
            "content"
        ]
        return normalized_response
