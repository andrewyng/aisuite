import os

from aisuite.providers.openai_provider import OpenaiProvider


def _parse_base_url(config) -> str:
    """Resolve and normalize the Atlas Cloud OpenAI-compatible LLM endpoint."""
    base_url = (
        config.pop("base_url", None)
        or config.pop("api_url", None)
        or os.getenv("ATLASCLOUD_API_BASE", "https://api.atlascloud.ai/v1")
    )
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


class AtlascloudProvider(OpenaiProvider):
    """
    Atlas Cloud provider for its OpenAI-compatible LLM API.

    Atlas Cloud exposes chat completions through the standard OpenAI-compatible
    /v1 API. Reusing the OpenAI provider keeps tool calls, tool-result
    messages, streaming, and finish_reason behavior aligned with the rest of
    aisuite's OpenAI-compatible providers.
    """

    def __init__(self, **config):
        config["base_url"] = _parse_base_url(config)
        config.setdefault("api_key", os.getenv("ATLASCLOUD_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Atlas Cloud API key is missing. Please provide it in the config "
                "or set the ATLASCLOUD_API_KEY environment variable."
            )
        super().__init__(**config)
