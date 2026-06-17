from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aisuite.provider import LLMError
from aisuite.providers import foundry_provider
from aisuite.providers.foundry_provider import FoundryProvider


class FakeManager:
    """Stand-in for foundry_local.FoundryLocalManager used in tests."""

    instances = []

    def __init__(self, alias):
        self.alias = alias
        self.endpoint = "http://localhost:5273/v1"
        self.api_key = "foundry-key"
        self.downloaded = []
        self.loaded = []
        FakeManager.instances.append(self)

    def download_model(self, alias):
        self.downloaded.append(alias)

    def load_model(self, alias):
        self.loaded.append(alias)

    def get_model_info(self, alias):
        return SimpleNamespace(id=f"{alias}-cpu:1")


@pytest.fixture(autouse=True)
def _clear_manager_instances(monkeypatch):
    monkeypatch.delenv("FOUNDRY_LOCAL_API_URL", raising=False)
    FakeManager.instances = []
    yield


def test_explicit_endpoint_points_at_openai_compatible_v1(monkeypatch):
    provider = FoundryProvider(api_url="http://localhost:1234")
    assert "localhost:1234/v1" in str(provider.client.base_url)
    assert provider.client.api_key == "foundry"


def test_explicit_endpoint_normalises_v1_idempotently(monkeypatch):
    for kwargs in (
        {"base_url": "http://localhost:5273"},
        {"base_url": "http://localhost:5273/"},
        {"base_url": "http://localhost:5273/v1"},
        {"api_url": "http://localhost:5273/v1"},
    ):
        provider = FoundryProvider(**kwargs)
        assert str(provider.client.base_url) == "http://localhost:5273/v1/"


def test_explicit_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_LOCAL_API_URL", "http://remote-host:9999")
    provider = FoundryProvider()
    assert "remote-host:9999/v1" in str(provider.client.base_url)


def test_explicit_endpoint_passes_model_through(monkeypatch):
    provider = FoundryProvider(api_url="http://localhost:1234")
    messages = [{"role": "user", "content": "Hi"}]
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant", content="hello", tool_calls=None
                ),
            )
        ]
    )
    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            model="phi-3.5-mini", messages=messages
        )
    assert response.choices[0].message.content == "hello"
    # No SDK involved, so the alias is forwarded unchanged.
    assert mock_create.call_args.kwargs["model"] == "phi-3.5-mini"


def test_managed_mode_bootstraps_and_resolves_model_id(monkeypatch):
    monkeypatch.setattr(foundry_provider, "FoundryLocalManager", FakeManager)
    provider = FoundryProvider()
    messages = [{"role": "user", "content": "Hi"}]
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant", content="hello", tool_calls=None
                ),
            )
        ]
    )

    # Trigger bootstrap, then patch the (now-created) client's create method.
    model_id = provider._ensure_managed_model("phi-3.5-mini")
    assert model_id == "phi-3.5-mini-cpu:1"
    assert "localhost:5273/v1" in str(provider.client.base_url)
    assert provider.client.api_key == "foundry-key"

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            model="phi-3.5-mini", messages=messages
        )
    assert response.choices[0].message.content == "hello"
    # The resolved model id (not the alias) is sent to the endpoint.
    assert mock_create.call_args.kwargs["model"] == "phi-3.5-mini-cpu:1"
    # Only one manager/service is created across requests.
    assert len(FakeManager.instances) == 1


def test_managed_mode_loads_additional_alias(monkeypatch):
    monkeypatch.setattr(foundry_provider, "FoundryLocalManager", FakeManager)
    provider = FoundryProvider()

    provider._ensure_managed_model("phi-3.5-mini")
    manager = provider._manager
    # First alias is loaded by the manager constructor, not by an extra call.
    assert manager.loaded == []

    second_id = provider._ensure_managed_model("qwen2.5-0.5b")
    assert second_id == "qwen2.5-0.5b-cpu:1"
    # A second alias is explicitly downloaded and loaded on the same service.
    assert manager.downloaded == ["qwen2.5-0.5b"]
    assert manager.loaded == ["qwen2.5-0.5b"]
    assert len(FakeManager.instances) == 1


def test_managed_mode_without_sdk_raises(monkeypatch):
    monkeypatch.setattr(foundry_provider, "FoundryLocalManager", None)
    provider = FoundryProvider()
    with pytest.raises(LLMError, match="Foundry Local SDK is not installed"):
        provider.chat_completions_create(
            model="phi-3.5-mini", messages=[{"role": "user", "content": "Hi"}]
        )
