import pytest
from unittest.mock import patch, MagicMock

from aisuite.providers.mistral_provider import MistralProvider, MistralAsyncProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")


def test_mistral_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = MistralProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": response_text_content}}]
    }

    with patch.object(
        provider.client.chat, "complete", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_with(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert response.choices[0].message.content == response_text_content


@pytest.mark.asyncio
async def test_mistral_provider_async():
    """High-level test that the provider handles async chat completions successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = MistralAsyncProvider()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": response_text_content}}]
    }

    with patch.object(
        provider.client.chat, "complete_async", return_value=mock_response
    ) as mock_create:
        response = await provider.chat_completions_create_async(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_with(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert response.choices[0].message.content == response_text_content
