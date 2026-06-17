import os

from aisuite.provider import LLMError
from aisuite.providers.openai_provider import OpenaiProvider

try:  # pragma: no cover - import guard exercised via tests by patching the symbol
    from foundry_local import FoundryLocalManager
except ImportError:
    FoundryLocalManager = None


def _normalize_endpoint(base_url: str) -> str:
    """Normalize a Foundry Local host to its OpenAI-compatible ``/v1`` endpoint
    (idempotently, so a host already ending in /v1 is left untouched)."""
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


class FoundryProvider(OpenaiProvider):
    """
    Foundry Local provider for Microsoft's `Foundry Local
    <https://learn.microsoft.com/azure/ai-foundry/foundry-local/>`_ on-device runtime.

    Foundry Local exposes an OpenAI-compatible API, so reusing the OpenAI SDK
    means tool calls, tool-result messages, and finish_reason flow through
    unchanged. There are two ways to use it:

    * Managed (default): the ``foundry-local-sdk`` package
      (``FoundryLocalManager``) starts the local service on demand, downloads
      and loads the requested model, and resolves the model alias to the
      concrete model id. Install it with ``pip install foundry-local-sdk``.
      Because Foundry Local picks a dynamic port, the manager is the easiest
      way to discover the endpoint.
    * Explicit endpoint: set ``api_url``/``base_url`` (or the
      ``FOUNDRY_LOCAL_API_URL`` environment variable) to an already-running
      Foundry Local OpenAI-compatible endpoint. In this mode the SDK is not
      required and the model string is passed through unchanged.
    """

    _ENV_API_URL = "FOUNDRY_LOCAL_API_URL"

    def __init__(self, **config):
        base_url = (
            config.pop("base_url", None)
            or config.pop("api_url", None)
            or os.getenv(self._ENV_API_URL)
        )
        self._manager = None
        self._model_ids = {}

        if base_url:
            # Talk directly to an already-running Foundry Local endpoint.
            config["base_url"] = _normalize_endpoint(base_url)
            # Foundry Local ignores the API key, but the OpenAI SDK requires one.
            config.setdefault("api_key", "foundry")
            self._managed = False
            super().__init__(**config)
        else:
            # Defer client creation until the first request, when the model
            # alias is known and FoundryLocalManager can start the service and
            # load the model.
            self._managed = True
            self._config = config
            self.audio = None

    def _ensure_managed_model(self, model):
        """Bootstrap the Foundry Local service for ``model`` (starting it and
        loading the model on first use) and return the concrete model id the
        OpenAI-compatible endpoint expects."""
        if FoundryLocalManager is None:
            raise LLMError(
                "Foundry Local SDK is not installed. Install it with "
                "`pip install foundry-local-sdk`, or set api_url/base_url "
                "(or the FOUNDRY_LOCAL_API_URL environment variable) to point "
                "at a running Foundry Local endpoint."
            )

        if self._manager is None:
            # Starts the service if needed and downloads/loads the model.
            self._manager = FoundryLocalManager(model)
            config = dict(self._config)
            config["base_url"] = self._manager.endpoint
            config.setdefault("api_key", self._manager.api_key or "foundry")
            super().__init__(**config)
        elif model not in self._model_ids:
            # A different alias on an already-running service: make sure it is
            # downloaded and loaded before use.
            self._manager.download_model(model)
            self._manager.load_model(model)

        if model not in self._model_ids:
            info = self._manager.get_model_info(model)
            self._model_ids[model] = info.id if info is not None else model
        return self._model_ids[model]

    def chat_completions_create(self, model, messages, **kwargs):
        if self._managed:
            model = self._ensure_managed_model(model)
        return super().chat_completions_create(model, messages, **kwargs)
