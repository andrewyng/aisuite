from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aisuite.provider import LLMError
from aisuite.providers import foundry_local_provider
from aisuite.providers.foundry_local_provider import FoundryLocalProvider

# --- Fakes for the legacy 0.x `foundry_local` SDK -------------------------


class FakeLegacyManager:
    """Stand-in for foundry_local.FoundryLocalManager (0.x API)."""

    instances = []

    def __init__(self, alias):
        self.alias = alias
        self.endpoint = "http://localhost:5273/v1"
        self.api_key = "foundry-key"
        self.downloaded = []
        self.loaded = []
        FakeLegacyManager.instances.append(self)

    def download_model(self, alias):
        self.downloaded.append(alias)

    def load_model(self, alias):
        self.loaded.append(alias)

    def get_model_info(self, alias):
        return SimpleNamespace(id=f"{alias}-cpu:1")


# --- Fakes for the current 1.x `foundry_local_sdk` SDK --------------------


class FakeIModel:
    def __init__(self, alias):
        self.alias = alias
        self.id = f"{alias}-generic-cpu:1"
        self.downloaded = False
        self.loaded = False

    def download(self, *args, **kwargs):
        self.downloaded = True

    def load(self, *args, **kwargs):
        self.loaded = True


class FakeCatalog:
    def __init__(self):
        self.models = {}

    def get_model(self, alias):
        return self.models.setdefault(alias, FakeIModel(alias))


class FakeConfiguration:
    def __init__(self, app_name=None, **kwargs):
        self.app_name = app_name


class FakeNewManager:
    """Stand-in for foundry_local_sdk.FoundryLocalManager (1.x singleton API)."""

    instance = None

    def __init__(self, config):
        self.config = config
        self.catalog = FakeCatalog()
        self.urls = None
        self.eps_registered = False
        FakeNewManager.instance = self

    @classmethod
    def initialize(cls, config):
        cls(config)

    def download_and_register_eps(self, *args, **kwargs):
        self.eps_registered = True

    def start_web_service(self):
        self.urls = ["http://127.0.0.1:5273"]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("FOUNDRY_LOCAL_API_URL", raising=False)
    FakeLegacyManager.instances = []
    FakeNewManager.instance = None
    yield


def _use_legacy_sdk(monkeypatch):
    monkeypatch.setattr(foundry_local_provider, "_import_new_sdk", lambda: None)
    monkeypatch.setattr(
        foundry_local_provider, "_import_legacy_sdk", lambda: FakeLegacyManager
    )


def _use_new_sdk(monkeypatch):
    monkeypatch.setattr(
        foundry_local_provider,
        "_import_new_sdk",
        lambda: (FakeConfiguration, FakeNewManager),
    )
    monkeypatch.setattr(foundry_local_provider, "_import_legacy_sdk", lambda: None)


def _no_sdk(monkeypatch):
    monkeypatch.setattr(foundry_local_provider, "_import_new_sdk", lambda: None)
    monkeypatch.setattr(foundry_local_provider, "_import_legacy_sdk", lambda: None)


_MOCK_RESPONSE = SimpleNamespace(
    choices=[
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(role="assistant", content="hello", tool_calls=None),
        )
    ]
)


# --- Explicit-endpoint mode (no SDK required) -----------------------------


def test_explicit_endpoint_points_at_openai_compatible_v1():
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    assert "localhost:1234/v1" in str(provider.client.base_url)
    assert provider.client.api_key == "foundry"


def test_explicit_endpoint_normalises_v1_idempotently():
    for kwargs in (
        {"base_url": "http://localhost:5273"},
        {"base_url": "http://localhost:5273/"},
        {"base_url": "http://localhost:5273/v1"},
        {"api_url": "http://localhost:5273/v1"},
    ):
        provider = FoundryLocalProvider(**kwargs)
        assert str(provider.client.base_url) == "http://localhost:5273/v1/"


def test_explicit_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_LOCAL_API_URL", "http://remote-host:9999")
    provider = FoundryLocalProvider()
    assert "remote-host:9999/v1" in str(provider.client.base_url)


def test_explicit_endpoint_passes_model_through():
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    messages = [{"role": "user", "content": "Hi"}]
    with patch.object(provider.client.models, "list", return_value=[]), patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        response = provider.chat_completions_create(
            model="phi-3.5-mini", messages=messages
        )
    assert response.choices[0].message.content == "hello"
    # The endpoint advertises no models, so the alias is forwarded unchanged.
    assert mock_create.call_args.kwargs["model"] == "phi-3.5-mini"


def _served(model_id, parent=None):
    extra = {"parent": parent} if parent is not None else {}
    return SimpleNamespace(id=model_id, parent=parent, model_extra=extra)


def test_explicit_endpoint_resolves_alias_by_prefix():
    # CLI-served endpoints expose ids like "...-cpu:4" with no parent field,
    # so the alias must be matched as an id prefix.
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    served = [_served("phi-3.5-mini-instruct-generic-cpu:4")]
    with patch.object(
        provider.client.models, "list", return_value=served
    ), patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        provider.chat_completions_create(
            model="phi-3.5-mini", messages=[{"role": "user", "content": "Hi"}]
        )
    assert (
        mock_create.call_args.kwargs["model"] == "phi-3.5-mini-instruct-generic-cpu:4"
    )


def test_explicit_endpoint_resolves_alias_by_parent():
    # SDK-started endpoints advertise a parent alias for each served model.
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    served = [_served("phi-3.5-mini-instruct-generic-gpu", parent="phi-3.5-mini")]
    with patch.object(
        provider.client.models, "list", return_value=served
    ), patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        provider.chat_completions_create(
            model="phi-3.5-mini", messages=[{"role": "user", "content": "Hi"}]
        )
    assert mock_create.call_args.kwargs["model"] == "phi-3.5-mini-instruct-generic-gpu"


def test_explicit_endpoint_concrete_id_passes_through():
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    served = [_served("phi-3.5-mini-instruct-generic-cpu:4")]
    with patch.object(
        provider.client.models, "list", return_value=served
    ), patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        provider.chat_completions_create(
            model="phi-3.5-mini-instruct-generic-cpu:4",
            messages=[{"role": "user", "content": "Hi"}],
        )
    assert (
        mock_create.call_args.kwargs["model"] == "phi-3.5-mini-instruct-generic-cpu:4"
    )


def test_explicit_endpoint_ambiguous_alias_raises():
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    served = [
        _served("phi-3.5-mini-instruct-generic-cpu:4"),
        _served("phi-3.5-mini-instruct-generic-gpu:4"),
    ]
    with patch.object(provider.client.models, "list", return_value=served):
        with pytest.raises(LLMError, match="matches multiple models"):
            provider.chat_completions_create(
                model="phi-3.5-mini", messages=[{"role": "user", "content": "Hi"}]
            )


def test_explicit_endpoint_unreachable_models_list_falls_back():
    provider = FoundryLocalProvider(api_url="http://localhost:1234")
    with patch.object(
        provider.client.models, "list", side_effect=RuntimeError("connection error")
    ), patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        provider.chat_completions_create(
            model="phi-3.5-mini", messages=[{"role": "user", "content": "Hi"}]
        )
    assert mock_create.call_args.kwargs["model"] == "phi-3.5-mini"


# --- Managed mode: legacy 0.x SDK -----------------------------------------


def test_managed_legacy_bootstraps_and_resolves_model_id(monkeypatch):
    _use_legacy_sdk(monkeypatch)
    provider = FoundryLocalProvider()
    messages = [{"role": "user", "content": "Hi"}]

    model_id = provider._ensure_managed_model("phi-3.5-mini")
    assert model_id == "phi-3.5-mini-cpu:1"
    assert "localhost:5273/v1" in str(provider.client.base_url)
    assert provider.client.api_key == "foundry-key"

    with patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        provider.chat_completions_create(model="phi-3.5-mini", messages=messages)
    # The resolved model id (not the alias) is sent to the endpoint.
    assert mock_create.call_args.kwargs["model"] == "phi-3.5-mini-cpu:1"
    # Only one manager/service is created across requests.
    assert len(FakeLegacyManager.instances) == 1


def test_managed_legacy_loads_additional_alias(monkeypatch):
    _use_legacy_sdk(monkeypatch)
    provider = FoundryLocalProvider()

    provider._ensure_managed_model("phi-3.5-mini")
    manager = provider._manager
    # First alias is loaded by the manager constructor, not by an extra call.
    assert manager.loaded == []

    second_id = provider._ensure_managed_model("qwen2.5-0.5b")
    assert second_id == "qwen2.5-0.5b-cpu:1"
    # A second alias is explicitly downloaded and loaded on the same service.
    assert manager.downloaded == ["qwen2.5-0.5b"]
    assert manager.loaded == ["qwen2.5-0.5b"]
    assert len(FakeLegacyManager.instances) == 1


# --- Managed mode: current 1.x SDK ----------------------------------------


def test_managed_new_bootstraps_via_web_service(monkeypatch):
    _use_new_sdk(monkeypatch)
    provider = FoundryLocalProvider()
    messages = [{"role": "user", "content": "Hi"}]

    model_id = provider._ensure_managed_model("qwen2.5-0.5b")
    assert model_id == "qwen2.5-0.5b-generic-cpu:1"
    # The OpenAI client targets the web service's /v1 endpoint.
    assert "127.0.0.1:5273/v1" in str(provider.client.base_url)
    manager = provider._manager
    assert manager.eps_registered is True
    assert manager.urls == ["http://127.0.0.1:5273"]
    # The model was downloaded and loaded via the catalog.
    imodel = manager.catalog.models["qwen2.5-0.5b"]
    assert imodel.downloaded and imodel.loaded

    with patch.object(
        provider.client.chat.completions, "create", return_value=_MOCK_RESPONSE
    ) as mock_create:
        provider.chat_completions_create(model="qwen2.5-0.5b", messages=messages)
    assert mock_create.call_args.kwargs["model"] == "qwen2.5-0.5b-generic-cpu:1"


def test_managed_new_reuses_singleton_and_loads_additional_alias(monkeypatch):
    _use_new_sdk(monkeypatch)
    provider = FoundryLocalProvider()

    first = provider._ensure_managed_model("qwen2.5-0.5b")
    manager = provider._manager
    second = provider._ensure_managed_model("phi-3.5-mini")
    assert first == "qwen2.5-0.5b-generic-cpu:1"
    assert second == "phi-3.5-mini-generic-cpu:1"
    # The singleton is reused (web service started only once).
    assert provider._manager is manager
    assert manager.catalog.models["phi-3.5-mini"].loaded is True


# --- Missing SDK ----------------------------------------------------------


def test_managed_mode_without_sdk_raises(monkeypatch):
    _no_sdk(monkeypatch)
    provider = FoundryLocalProvider()
    with pytest.raises(LLMError, match="Foundry Local SDK is not installed"):
        provider.chat_completions_create(
            model="phi-3.5-mini", messages=[{"role": "user", "content": "Hi"}]
        )
