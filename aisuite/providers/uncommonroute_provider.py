import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class UncommonrouteProvider(Provider):
    """
    UncommonRoute provider - an intelligent LLM cost optimizer.

    UncommonRoute runs as a local proxy that analyzes each request and
    routes it to the most cost-effective model capable of handling the task.

    Setup:
        pipx install uncommon-route
        uncommon-route init
        uncommon-route serve

    Configuration:
        Set UNCOMMON_ROUTE_BASE_URL (default: http://localhost:8403/v1)
        Or pass base_url in provider config.

    Usage:
        client = ai.Client()
        response = client.chat.completions.create(
            model="uncommonroute:auto",
            messages=[{"role": "user", "content": "Hello!"}],
        )

    Routing modes (passed as the model name):
        - auto: balanced quality-per-dollar (recommended)
        - fast: cost-first, routes to cheapest capable model
        - best: quality-first, uses strongest model available

    Learn more: https://github.com/CommonstackAI/UncommonRoute
    """

    BASE_URL = "http://localhost:8403/v1"

    def __init__(self, **config):
        self.base_url = config.get(
            "base_url",
            os.getenv("UNCOMMON_ROUTE_BASE_URL", self.BASE_URL),
        )
        self.timeout = config.get("timeout", 60)
        self.converter = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        model = self._resolve_model(model)
        converted_messages = self.converter.convert_request(messages)

        data = {
            "model": model,
            "messages": converted_messages,
            **kwargs,
        }

        url = self.base_url.rstrip("/") + "/chat/completions"

        try:
            response = httpx.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
        except httpx.ConnectError:
            raise LLMError(
                "Could not connect to UncommonRoute. "
                "Make sure it is running: uncommon-route serve"
            )
        except httpx.HTTPStatusError as e:
            raise LLMError(f"UncommonRoute request failed: {e}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        return self.converter.convert_response(response.json())

    def _resolve_model(self, model):
        mode_map = {
            "auto": "uncommon-route/auto",
            "fast": "uncommon-route/fast",
            "best": "uncommon-route/best",
        }
        return mode_map.get(model, model)
