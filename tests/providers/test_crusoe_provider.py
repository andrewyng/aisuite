from types import SimpleNamespace

import pytest
from unittest.mock import patch, MagicMock

from aisuite.providers.crusoe_provider import CrusoeProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("CRUSOE_API_KEY", "test-api-key")


def test_crusoe_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = CrusoeProvider()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = response_text_content

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

        mock_create.assert_called_with(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        assert response.choices[0].message.content == response_text_content


def test_crusoe_provider_tool_calling():
    """tools are forwarded unchanged and tool_calls in the response survive."""

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
    selected_model = "our-favorite-model"

    provider = CrusoeProvider()

    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "SF"}'),
    )
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant", content=None, tool_calls=[tool_call]
                ),
            )
        ]
    )

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=mock_response,
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

        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert (
            response.choices[0].message.tool_calls[0].function.arguments
            == '{"city": "SF"}'
        )
