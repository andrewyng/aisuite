from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.cohere_provider import CohereProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("CO_API_KEY", "test-api-key")


def test_cohere_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = CohereProvider()
    mock_response = MagicMock()
    mock_response.message = MagicMock()
    mock_response.message.content = [MagicMock()]
    mock_response.message.content[0].text = response_text_content

    with patch.object(
        provider.client,
        "chat",
        return_value=mock_response,
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


def test_cohere_finish_reason_max_tokens():
    """A MAX_TOKENS Cohere response normalizes to OpenAI 'length', not 'stop'."""
    provider = CohereProvider()
    mock_response = MagicMock()
    mock_response.finish_reason = "MAX_TOKENS"
    mock_response.message = MagicMock()
    mock_response.message.content = [MagicMock()]
    mock_response.message.content[0].text = "truncated output"

    with patch.object(provider.client, "chat", return_value=mock_response):
        response = provider.chat_completions_create(
            messages=[{"role": "user", "content": "hi"}],
            model="our-favorite-model",
        )

    assert response.choices[0].finish_reason == "length"
