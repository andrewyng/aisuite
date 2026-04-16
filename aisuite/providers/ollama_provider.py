import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


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

        # Initialize the message converter to sanitize aisuite Message objects into JSON
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the chat completions endpoint using httpx.
        """
        kwargs["stream"] = False

        # Convert internal aisuite Message objects to standard dictionaries
        transformed_messages = self.transformer.convert_request(messages)

        data = {
            "model": model,
            "messages": transformed_messages,
            # Pass any additional arguments to the API (like tools, temperature)
            **kwargs,
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
        message_data = response_data.get("message", {})

        # Map the standard text content (defaults to empty string if missing)
        normalized_response.choices[0].message.content = message_data.get("content", "")

        # Map the tool calls if the LLM generated them
        if "tool_calls" in message_data and message_data["tool_calls"]:
            normalized_response.choices[0].message.tool_calls = message_data[
                "tool_calls"
            ]

        return normalized_response
