"""Tests for the Featherless provider."""

from unittest.mock import MagicMock

import pytest

from aisuite.providers.featherless_provider import FeatherlessProvider


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-api-key")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is missing"):
        FeatherlessProvider()


def test_completion_passes_through():
    provider = FeatherlessProvider()
    response = MagicMock()
    provider.client.chat.completions.create = MagicMock(return_value=response)

    result = provider.chat_completions_create(
        "featherless-model", [{"role": "user", "content": "hi"}], temperature=0.2
    )

    assert result is response
    call = provider.client.chat.completions.create.call_args
    assert call.kwargs["model"] == "featherless-model"
    assert call.kwargs["temperature"] == 0.2
