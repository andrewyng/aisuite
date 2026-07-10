"""Tests for the LiteLLM provider."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-api-key")


@pytest.fixture(autouse=True)
def stub_litellm(monkeypatch):
    """Stub the litellm module so tests run without the real package."""
    fake = types.ModuleType("litellm")
    fake.completion = MagicMock(name="litellm.completion")
    fake.acompletion = MagicMock(name="litellm.acompletion")
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def test_litellm_provider():
    """Test that the provider initializes and dispatches chat completions."""
    from aisuite.providers.litellm_provider import LitellmProvider

    message_history = [{"role": "user", "content": "Hello!"}]
    selected_model = "anthropic/claude-sonnet-4-6"
    response_text = "mocked-text-response-from-model"

    provider = LitellmProvider()

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": response_text, "role": "assistant"}}]
    }

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        response = provider.chat_completions_create(
            model=selected_model,
            messages=message_history,
            temperature=0.7,
        )

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == selected_model
        assert call_kwargs.kwargs["drop_params"] is True
        assert call_kwargs.kwargs["temperature"] == 0.7
        assert response.choices[0].message.content == response_text


def test_litellm_provider_with_usage():
    """Test that usage data is correctly parsed when present."""
    from aisuite.providers.litellm_provider import LitellmProvider

    provider = LitellmProvider()

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "response", "role": "assistant"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    with patch("litellm.completion", return_value=mock_response):
        response = provider.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


def test_litellm_provider_forwards_api_key():
    """Test that api_key is forwarded when set."""
    from aisuite.providers.litellm_provider import LitellmProvider

    provider = LitellmProvider(api_key="sk-custom-key")

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "ok", "role": "assistant"}}]
    }

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        provider.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-custom-key"


def test_litellm_provider_forwards_base_url():
    """Test that base_url is forwarded as api_base when set."""
    from aisuite.providers.litellm_provider import LitellmProvider

    provider = LitellmProvider(base_url="http://localhost:4000")

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "ok", "role": "assistant"}}]
    }

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        provider.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["api_base"] == "http://localhost:4000"


def test_litellm_provider_omits_api_key_when_empty():
    """Test that api_key is omitted when not set, letting litellm read provider env vars."""
    from aisuite.providers.litellm_provider import LitellmProvider

    provider = LitellmProvider(api_key=None)
    provider.api_key = None

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "ok", "role": "assistant"}}]
    }

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        provider.chat_completions_create(
            model="anthropic/claude-haiku",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert "api_key" not in call_kwargs


def test_litellm_provider_drop_params_override():
    """Test that drop_params can be overridden to False."""
    from aisuite.providers.litellm_provider import LitellmProvider

    provider = LitellmProvider(drop_params=False)

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "ok", "role": "assistant"}}]
    }

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        provider.chat_completions_create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["drop_params"] is False
