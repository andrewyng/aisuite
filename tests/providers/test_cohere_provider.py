from unittest.mock import MagicMock, patch

import pytest
from cohere import V2ChatResponse

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


@pytest.mark.parametrize("serialized", [False, True])
def test_replay_tool_plan(serialized):
    provider = CohereProvider()
    plan = "Look up the forecast."
    reply = V2ChatResponse(
        id="reply",
        finish_reason="TOOL_CALL",
        message={
            "role": "assistant",
            "tool_plan": plan,
            "tool_calls": [
                {
                    "id": "forecast-1",
                    "type": "function",
                    "function": {"name": "forecast", "arguments": "{}"},
                }
            ],
        },
        usage={"tokens": {"input_tokens": 5, "output_tokens": 3}},
    )
    with patch.object(provider.client, "chat", return_value=reply) as chat:
        history = [{"role": "user", "content": "What's the weather?"}]
        assistant = (
            provider.chat_completions_create("command", history).choices[0].message
        )
        assert assistant.content == plan
        history.extend(
            [
                assistant.model_dump() if serialized else assistant,
                {"role": "tool", "tool_call_id": "forecast-1", "content": "sunny"},
            ]
        )
        provider.chat_completions_create("command", history)
    recalled = chat.call_args.kwargs["messages"][1]
    assert recalled["tool_plan"] == plan
    assert recalled["tool_calls"][0]["id"] == "forecast-1"
    assert "content" not in recalled


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
