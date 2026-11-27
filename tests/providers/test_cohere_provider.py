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


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "answer"},
            ],
            "answer",
        ),
        (
            [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}],
            "one two",
        ),
        ([], None),
        ([{"type": "text", "text": ""}], ""),
    ],
)
def test_cohere_response_content(content, expected):
    provider = CohereProvider()
    reply = V2ChatResponse(
        id="reply",
        finish_reason="COMPLETE",
        message={"role": "assistant", "content": content},
        usage={"tokens": {"input_tokens": 5, "output_tokens": 3}},
    )
    with patch.object(provider.client, "chat", return_value=reply):
        response = provider.chat_completions_create(
            model="command", messages=[{"role": "user", "content": "Hello"}]
        )
    assert response.choices[0].message.content == expected
    assert response.choices[0].finish_reason == "stop"
