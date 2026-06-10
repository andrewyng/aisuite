import os

from aisuite.providers.openai_provider import OpenaiProvider


class LmstudioProvider(OpenaiProvider):
    """
    LM Studio provider that talks to LM Studio's OpenAI-compatible API under /v1.

    LM Studio already exposes an OpenAI-compatible server, so reusing the OpenAI
    SDK means tool calls, tool-result messages, and finish_reason flow through
    unchanged instead of being dropped during normalization.

    The host defaults to http://localhost:1234 and can be overridden via the
    ``api_url`` config key or the ``LMSTUDIO_API_URL`` environment variable. Any
    other config (e.g. ``timeout``) is forwarded to the OpenAI client.
    See https://lmstudio.ai/docs/api
    """

    _DEFAULT_HOST = "http://localhost:1234"

    def __init__(self, **config):
        host = config.pop("api_url", None) or os.getenv(
            "LMSTUDIO_API_URL", self._DEFAULT_HOST
        )
        config["base_url"] = host.rstrip("/") + "/v1"
        # LM Studio ignores the API key, but the OpenAI SDK requires one.
        config.setdefault("api_key", "lmstudio")
        super().__init__(**config)
