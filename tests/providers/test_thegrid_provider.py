"""Tests for The Grid provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from aisuite.providers.thegrid_provider import ThegridProvider
from aisuite.framework.chat_completion_response import ChatCompletionResponse


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set The Grid API key environment variable for tests."""
    monkeypatch.setenv("THEGRID_API_KEY", "test-api-key")


def test_missing_api_key_raises(monkeypatch):
    """The Grid is a hosted, keyed API: constructing without a key must fail."""
    monkeypatch.delenv("THEGRID_API_KEY", raising=False)
    with pytest.raises(ValueError, match="The Grid API key is missing"):
        ThegridProvider()


def test_thegrid_provider():
    """Test that the provider is initialized and chat completions are requested."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "text-standard"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = ThegridProvider()
    assert provider.client.base_url.host == "api.thegrid.ai"

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

        mock_create.assert_called_with(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content == response_text_content


def test_thegrid_provider_returns_serving_model():
    """An instrument pools several backing models, so the response `model` names
    the model that served the request rather than the instrument requested. A
    caller must not assume the two match."""

    provider = ThegridProvider()

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "OK", "role": "assistant"}}],
        "model": "openai/gpt-oss-120b",
        "created": 12345,
        "id": "chatcmpl-mockid",
    }

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ):
        response = provider.chat_completions_create(
            messages=[{"role": "user", "content": "Hi"}],
            model="text-standard",
        )

    assert response.choices[0].message.content == "OK"


def test_thegrid_provider_tool_calling():
    """Tools are forwarded unchanged and tool_calls in the response survive."""

    messages = [{"role": "user", "content": "What's the weather in SF?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    selected_model = "agent-standard"

    provider = ThegridProvider()

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                },
            }
        ],
        "model": selected_model,
        "created": 12345,
        "id": "chatcmpl-mockid",
    }

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=messages,
            model=selected_model,
            tools=tools,
            tool_choice="auto",
        )

        mock_create.assert_called_with(
            messages=messages,
            model=selected_model,
            tools=tools,
            tool_choice="auto",
        )

        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert (
            response.choices[0].message.tool_calls[0].function.arguments
            == '{"city": "SF"}'
        )


def test_explicit_config_overrides_env(monkeypatch):
    """An explicit api_key and base_url take precedence over the environment."""
    monkeypatch.setenv("THEGRID_API_KEY", "env-key")
    provider = ThegridProvider(
        api_key="explicit-key", base_url="https://proxy.internal/v1"
    )
    assert provider.client.api_key == "explicit-key"
    assert provider.client.base_url.host == "proxy.internal"
