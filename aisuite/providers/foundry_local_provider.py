import importlib
import os

from aisuite.provider import LLMError
from aisuite.providers.openai_provider import OpenaiProvider

_APP_NAME = "aisuite"


def _normalize_endpoint(base_url: str) -> str:
    """Normalize a Foundry Local host to its OpenAI-compatible ``/v1`` endpoint
    (idempotently, so a host already ending in /v1 is left untouched)."""
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def _import_new_sdk():
    """Return ``(Configuration, FoundryLocalManager)`` from the 1.x
    ``foundry_local_sdk`` package, or ``None`` if it is not installed."""
    try:
        module = importlib.import_module("foundry_local_sdk")
        return module.Configuration, module.FoundryLocalManager
    except ImportError:
        return None


def _import_legacy_sdk():
    """Return ``FoundryLocalManager`` from the 0.x ``foundry_local`` package, or
    ``None`` if it is not installed."""
    try:
        module = importlib.import_module("foundry_local")
        return module.FoundryLocalManager
    except ImportError:
        return None


class FoundryLocalProvider(OpenaiProvider):
    """
    Foundry Local provider for Microsoft's `Foundry Local
    <https://learn.microsoft.com/azure/ai-foundry/foundry-local/>`_ on-device runtime.

    Foundry Local exposes an OpenAI-compatible API, so reusing the OpenAI SDK
    means tool calls, tool-result messages, and finish_reason flow through
    unchanged. There are two ways to use it:

    * Managed (default): the Foundry Local Python SDK starts the local service,
      downloads and loads the requested model, and resolves the model alias to
      the concrete model id served by the OpenAI-compatible endpoint. Both the
      current ``foundry-local-sdk`` (1.x, imported as ``foundry_local_sdk``) and
      the legacy 0.x package (imported as ``foundry_local``) are supported.
      Install the variant that matches your hardware, e.g.
      ``pip install foundry-local-sdk``. Because Foundry Local picks a dynamic
      port, the SDK is the easiest way to discover the endpoint.
    * Explicit endpoint: set ``api_url``/``base_url`` (or the
      ``FOUNDRY_LOCAL_API_URL`` environment variable) to an already-running
      Foundry Local OpenAI-compatible endpoint. In this mode the SDK is not
      required. A friendly model alias is resolved to the concrete served id by
      querying the endpoint's ``/v1/models`` (an already-concrete id is passed
      through unchanged), so the same ``foundry_local:<alias>`` string works in
      both modes.
    """

    _ENV_API_URL = "FOUNDRY_LOCAL_API_URL"

    def __init__(self, **config):
        base_url = (
            config.pop("base_url", None)
            or config.pop("api_url", None)
            or os.getenv(self._ENV_API_URL)
        )
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
            # alias is known and the SDK can start the service and load it.
            self._managed = True
            self._config = config
            self._manager = None
            self._backend = None
            self._bootstrapped = False
            self.audio = None

    def chat_completions_create(self, model, messages, **kwargs):
        if self._managed:
            model = self._ensure_managed_model(model)
        else:
            model = self._resolve_explicit_model(model)
        return super().chat_completions_create(model, messages, **kwargs)

    def _resolve_explicit_model(self, model):
        """In explicit-endpoint mode, resolve a friendly alias (e.g.
        ``phi-3.5-mini``) to the concrete model id the endpoint serves, so the
        same ``foundry_local:<alias>`` string works in both managed and explicit
        modes. Resolution queries the endpoint's ``/v1/models`` and matches, in
        order, an exact id, a ``parent`` alias, then an id prefixed by the alias.
        Falls back to ``model`` unchanged when the lookup fails or there is no
        confident match (so an already-concrete id keeps working)."""
        if model in self._model_ids:
            return self._model_ids[model]

        try:
            served = list(self.client.models.list())
        except Exception:
            # Endpoint unreachable or doesn't implement /v1/models: don't block
            # the request, just forward the model string unchanged.
            served = None

        resolved = self._match_served_model(model, served) if served else model
        self._model_ids[model] = resolved
        return resolved

    @staticmethod
    def _served_parent(served_model):
        """Return the ``parent`` alias a served model was derived from, if the
        endpoint advertises one (it may be a model attribute or an extra field)."""
        parent = getattr(served_model, "parent", None)
        if parent is None:
            extra = getattr(served_model, "model_extra", None)
            if extra:
                parent = extra.get("parent")
        return parent

    def _match_served_model(self, alias, served):
        ids = [m.id for m in served]
        # 1. Already a concrete served id.
        if alias in ids:
            return alias
        # 2. A model the endpoint reports as derived from this alias.
        by_parent = [m.id for m in served if self._served_parent(m) == alias]
        if len(by_parent) == 1:
            return by_parent[0]
        if len(by_parent) > 1:
            raise LLMError(self._ambiguous_alias_message(alias, by_parent))
        # 3. A served id whose alias-boundary prefix matches (e.g.
        #    "phi-3.5-mini" -> "phi-3.5-mini-instruct-generic-cpu:4").
        by_prefix = [model_id for model_id in ids if model_id.startswith(alias + "-")]
        if len(by_prefix) == 1:
            return by_prefix[0]
        if len(by_prefix) > 1:
            raise LLMError(self._ambiguous_alias_message(alias, by_prefix))
        # 4. No confident match: forward unchanged and let the endpoint decide.
        return alias

    @staticmethod
    def _ambiguous_alias_message(alias, candidates):
        return (
            f"Foundry Local alias '{alias}' matches multiple models served by the "
            f"endpoint: {', '.join(sorted(candidates))}. Pass one of these concrete "
            "model ids instead."
        )

    def _ensure_managed_model(self, alias):
        """Bootstrap the Foundry Local service for ``alias`` on first use and
        return the concrete model id the OpenAI-compatible endpoint expects."""
        if not self._bootstrapped:
            self._bootstrap(alias)
        elif alias not in self._model_ids:
            self._load_additional_model(alias)
        return self._model_ids[alias]

    def _bootstrap(self, alias):
        new_sdk = _import_new_sdk()
        if new_sdk is not None:
            self._backend = "new"
            self._bootstrap_new(new_sdk, alias)
        else:
            legacy_manager = _import_legacy_sdk()
            if legacy_manager is None:
                raise LLMError(
                    "Foundry Local SDK is not installed. Install it with "
                    "`pip install foundry-local-sdk`, or set api_url/base_url "
                    "(or the FOUNDRY_LOCAL_API_URL environment variable) to point "
                    "at a running Foundry Local endpoint."
                )
            self._backend = "legacy"
            self._bootstrap_legacy(legacy_manager, alias)
        self._bootstrapped = True

    def _bootstrap_new(self, new_sdk, alias):
        """Bootstrap using the 1.x ``foundry_local_sdk`` singleton API: start the
        OpenAI-compatible web service and point the OpenAI client at it."""
        configuration_cls, manager_cls = new_sdk
        if manager_cls.instance is None:
            manager_cls.initialize(configuration_cls(app_name=_APP_NAME))
        self._manager = manager_cls.instance
        # Make the execution providers available before loading any model.
        self._manager.download_and_register_eps()
        self._model_ids[alias] = self._load_new_model(alias)
        if getattr(self._manager, "urls", None) is None:
            self._manager.start_web_service()

        config = dict(self._config)
        config["base_url"] = _normalize_endpoint(self._manager.urls[0])
        config.setdefault("api_key", "foundry")
        super().__init__(**config)

    def _load_new_model(self, alias):
        """Download and load ``alias`` via the 1.x catalog and return its id."""
        imodel = self._manager.catalog.get_model(alias)
        if imodel is None:
            raise LLMError(
                f"Foundry Local model '{alias}' was not found in the catalog."
            )
        imodel.download()
        imodel.load()
        return imodel.id

    def _bootstrap_legacy(self, manager_cls, alias):
        """Bootstrap using the legacy 0.x ``foundry_local`` API, where the
        manager starts the service and loads the model on construction."""
        self._manager = manager_cls(alias)
        info = self._manager.get_model_info(alias)
        self._model_ids[alias] = info.id if info is not None else alias

        config = dict(self._config)
        config["base_url"] = _normalize_endpoint(self._manager.endpoint)
        config.setdefault("api_key", self._manager.api_key or "foundry")
        super().__init__(**config)

    def _load_additional_model(self, alias):
        """Download and load a second alias on the already-running service."""
        if self._backend == "new":
            self._model_ids[alias] = self._load_new_model(alias)
        else:
            self._manager.download_model(alias)
            self._manager.load_model(alias)
            info = self._manager.get_model_info(alias)
            self._model_ids[alias] = info.id if info is not None else alias


# The provider factory derives the class name from the provider key
# ("foundry_local" -> "Foundry_localProvider" via str.capitalize), so keep that
# name available as an alias of the product-named class.
Foundry_localProvider = FoundryLocalProvider
