import os

import httpx
import openai

from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


def _parse_base_url(config) -> str:
    base_url = (
        config.get("base_url")
        or config.get("api_url")
        or os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1")
    )

    # strip last / if any and add v1 if not present
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls instead of using SDK.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, **config):
        """
        Initialize the Ollama provider with the given configuration.
        """
        # just for backward compatibility
        if "api_url" in config:
            config["base_url"] = config.pop("api_url")

        config["base_url"] = _parse_base_url(config)
        config["api_key"] = "ollama"  # required but ignored by ollama server

        # Optionally set a custom timeout (default to 30s)
        config["timeout"] = config.get("timeout", 30)

        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the chat completions endpoint using openai client.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return self.transformer.convert_response(response.model_dump())
        except httpx.ConnectError:  # Handle connection errors
            raise LLMError(f"Connection failed: {self._CONNECT_ERROR_MESSAGE}")
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Ollama request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
