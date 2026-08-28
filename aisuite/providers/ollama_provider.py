import os

from aisuite.providers.openai_provider import OpenaiProvider


def _parse_base_url(config) -> str:
    """Resolve the Ollama host — config ``base_url``/``api_url``, then the
    ``OLLAMA_API_URL`` env var — and normalize it to the OpenAI-compatible
    ``/v1`` endpoint (idempotently, so a host already ending in /v1 is fine)."""
    base_url = (
        config.pop("base_url", None)
        or config.pop("api_url", None)
        or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    )
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


class OllamaProvider(OpenaiProvider):
    """
    Ollama provider that talks to Ollama's OpenAI-compatible API under /v1.

    Using the /v1 endpoint (instead of Ollama's native /api/chat) means tool
    calls, tool-result messages, and finish_reason flow through the OpenAI SDK
    unchanged, so tool calling works the same as it does for OpenAI.

    The host defaults to http://localhost:11434 and can be overridden via the
    ``base_url`` or ``api_url`` config keys or the ``OLLAMA_API_URL``
    environment variable. The API key comes from an explicit ``api_key`` config
    entry, then the ``OLLAMA_API_KEY`` environment variable (used by Ollama
    Cloud), falling back to a dummy value that local servers ignore. Any other
    config (e.g. ``timeout``) is forwarded to the OpenAI client. See
    https://github.com/ollama/ollama/blob/main/docs/openai.md
    """

    def __init__(self, **config):
        config["base_url"] = _parse_base_url(config)
        # Ollama's local server ignores the API key but the OpenAI SDK requires
        # one; Ollama Cloud (https://ollama.com) requires a real key, which is
        # read from OLLAMA_API_KEY like every other provider's env var. An
        # explicit ``api_key`` in config always wins.
        config.setdefault("api_key", os.getenv("OLLAMA_API_KEY") or "ollama")
        # Local generation can be slow to first token; keep a sane default.
        config.setdefault("timeout", 30)
        super().__init__(**config)
