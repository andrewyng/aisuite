"""Tests for the Cheaper Inference provider."""

from unittest.mock import MagicMock, patch
import pytest

from aisuite.providers.cheaperinference_provider import CheaperinferenceProvider
from aisuite.framework.chat_completion_response import ChatCompletionResponse


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set the Cheaper Inference API key environment variable for tests."""
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "test-api-key")


def test_cheaperinference_provider():
    """Test that the provider is initialized and chat completions are requested."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "gpt-5.4-mini"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = CheaperinferenceProvider()
    assert provider.client.base_url.host == "api.cheaperinference.com"

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {"message": {"content": response_text_content, "role": "assistant"}}
        ],
        "model": selected_model,
        "created": 12345,
        "id": "chatcmpl-mockid",
    }

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
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


def test_cheaperinference_provider_with_usage():
    """Tests that usage data is correctly parsed when present in the response."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "gpt-5.4-mini"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = CheaperinferenceProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {"message": {"content": response_text_content, "role": "assistant"}}
        ],
        "model": selected_model,
        "created": 12345,
        "id": "chatcmpl-mockid",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
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
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30
