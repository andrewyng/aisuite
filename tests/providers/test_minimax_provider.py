"""Tests for the Minimax provider."""

import os
from unittest.mock import MagicMock, patch
import pytest

from aisuite.providers.minimax_provider import MinimaxProvider
from aisuite.framework.chat_completion_response import ChatCompletionResponse


@pytest.fixture
def mock_api_key(monkeypatch):
    """Fixture to set a mock Minimax API key for unit tests."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-api-key")


def test_minimax_provider(mock_api_key):
    """Test that the provider is initialized and chat completions are requested."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "MiniMax-Text-01"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = MinimaxProvider()
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


def test_minimax_provider_with_usage(mock_api_key):
    """Tests that usage data is correctly parsed when present in the response."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "MiniMax-Text-01"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = MinimaxProvider()
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

        mock_create.assert_called_once()

        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content == response_text_content
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20


def test_minimax_provider_with_system_message(mock_api_key):
    """Tests that system messages are correctly passed to the API."""

    message_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    selected_model = "MiniMax-M2.1"
    response_text_content = "Hello, user!"

    provider = MinimaxProvider()
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
        )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        # OpenAI-compatible API passes system message in the messages array
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "You are a helpful assistant."
        assert call_kwargs["messages"][1]["role"] == "user"

        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content == response_text_content


def test_minimax_provider_initialization(mock_api_key):
    """Test that Minimax provider initializes correctly."""
    provider = MinimaxProvider()
    assert provider is not None
    assert hasattr(provider, "client")
    assert hasattr(provider, "transformer")


# Integration tests - require real API key
@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_API_KEY") == "test-api-key",
    reason="MINIMAX_API_KEY not set or is test key",
)
class TestMinimaxIntegration:
    """Integration tests that call the real Minimax API."""

    def test_real_chat_completion(self):
        """Test a real chat completion with the Minimax API."""
        provider = MinimaxProvider()

        messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]

        response = provider.chat_completions_create(
            model="MiniMax-M2.1",
            messages=messages,
            max_tokens=50,
        )

        assert response is not None
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_real_chat_completion_with_system_message(self):
        """Test chat completion with system message."""
        provider = MinimaxProvider()

        messages = [
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "Say hello."},
        ]

        response = provider.chat_completions_create(
            model="MiniMax-M2.1",
            messages=messages,
            max_tokens=100,
        )

        assert response is not None
        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content is not None
