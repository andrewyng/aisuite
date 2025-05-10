import os
import pytest
from unittest.mock import patch, MagicMock

import openai
from aisuite.provider import LLMError
from aisuite.providers.lmstudio_provider import LMStudioProvider

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Ensure tests run with a known OPENAI_API_KEY and no side effects."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-env-key")
    yield

def test_init_defaults_to_localhost_and_env_key(monkeypatch):
    # Act
    prov = LMStudioProvider()

    # Assert client was configured correctly
    assert isinstance(prov.client, openai.OpenAI)
    assert prov.client.api_key == "test-env-key"
    assert prov.client.base_url == "http://localhost:1234/v1/"

def test_init_with_explicit_config_overrides_env(monkeypatch):
    prov = LMStudioProvider(api_key="explicit-key", base_url="http://localhost:1234/v1/")
    assert prov.client.api_key == "explicit-key"
    assert prov.client.base_url == "http://localhost:1234/v1/"

def test_init_missing_key_raises(monkeypatch):
    # Remove env var to simulate missing key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc:
        LMStudioProvider()
    assert "API key is missing" in str(exc.value)

@pytest.mark.parametrize("method, client_path, call_kwargs", [
    (
        "chat_completions_create",
        ("chat", "completions", "create"),
        {"model": "m1", "messages": [{"role":"user","content":"hi"}], "temperature":0.5}
    ),
    (
        "completions_create",
        ("completions", "create"),
        {"model": "m2", "prompt": "hello", "max_tokens":10}
    ),
    (
        "embeddings_create",
        ("embeddings", "create"),
        {"model": "m3", "input": ["foo", "bar"]}
    ),
])
def test_successful_proxy(monkeypatch, method, client_path, call_kwargs):
    """
    Ensure that each provider method calls through to the corresponding
    OpenAI client method with correct arguments and returns its result.
    """
    prov = LMStudioProvider()
    # Create a fake response
    fake_resp = MagicMock()
    # Drill into prov.client to patch the target method
    target = prov.client
    for attr in client_path[:-1]:
        target = getattr(target, attr)
    monkeypatch.setattr(target, client_path[-1], lambda **kw: fake_resp)

    # For chat, also patch transformer
    if method == "chat_completions_create":
        monkeypatch.setattr(prov.transformer, "convert_request", lambda msgs: msgs)

    # Call the provider method
    resp = getattr(prov, method)(**call_kwargs)
    assert resp is fake_resp

@pytest.mark.parametrize("method, client_path, call_kwargs", [
    (
        "chat_completions_create",
        ("chat", "completions", "create"),
        {"model": "m1", "messages": [{"role":"user","content":"err"}]}
    ),
    (
        "completions_create",
        ("completions", "create"),
        {"model": "m2", "prompt": "err"}
    ),
    (
        "embeddings_create",
        ("embeddings", "create"),
        {"model": "m3", "input": ["err"]}
    ),
])
def test_error_path_raises_llmerror(monkeypatch, method, client_path, call_kwargs):
    """
    If the underlying OpenAI client raises, the provider should catch
    and re-raise as LLMError.
    """
    prov = LMStudioProvider()

    # Drill into prov.client to patch the target method to raise
    target = prov.client
    for attr in client_path[:-1]:
        target = getattr(target, attr)
    def blow_up(**kw):
        raise RuntimeError("underlying failure")
    monkeypatch.setattr(target, client_path[-1], blow_up)

    if method == "chat_completions_create":
        monkeypatch.setattr(prov.transformer, "convert_request", lambda msgs: msgs)

    with pytest.raises(LLMError) as exc:
        getattr(prov, method)(**call_kwargs)
    assert "LMStudio" in str(exc.value)
