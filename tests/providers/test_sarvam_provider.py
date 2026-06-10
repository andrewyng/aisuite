from unittest.mock import MagicMock, patch

import pytest

from aisuite.framework import ChatCompletionResponse
from aisuite.providers.sarvam_provider import SarvamProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-api-key")


def test_sarvam_provider():
    """Test that the provider is initialized and chat completions are requested."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "sarvam-m"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = SarvamProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {"message": {"content": response_text_content, "role": "assistant"}}
        ]
    }

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=mock_response,
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_once_with(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content == response_text_content
        assert response.usage is None


def test_sarvam_provider_with_usage():
    """Tests that usage data is correctly parsed when present in the response."""

    message_history = [{"role": "user", "content": "Hello!"}]
    selected_model = "sarvam-m"
    response_text_content = "mocked-text-response-from-model"

    provider = SarvamProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {"message": {"content": response_text_content, "role": "assistant"}}
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=mock_response,
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
        )

        mock_create.assert_called_once_with(
            messages=message_history,
            model=selected_model,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


def test_sarvam_provider_missing_api_key(monkeypatch):
    """Tests that a missing API key raises a ValueError."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Sarvam API key is missing"):
        SarvamProvider()


def test_sarvam_provider_uses_correct_auth_header():
    """Tests that the provider sets the Sarvam-specific auth header on the client."""
    provider = SarvamProvider()

    assert provider.client.default_headers.get("api-subscription-key") == "test-api-key"


def test_sarvam_provider_api_key_from_config(monkeypatch):
    """Tests that the API key can be passed via config instead of env var."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    provider = SarvamProvider(api_key="config-api-key")

    assert provider.api_key == "config-api-key"
    assert (
        provider.client.default_headers.get("api-subscription-key") == "config-api-key"
    )
