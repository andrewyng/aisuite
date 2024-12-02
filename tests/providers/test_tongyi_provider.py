from unittest.mock import MagicMock, patch
import pytest
import dashscope
from aisuite.providers.tongyi_provider import TongyiProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-api-key")


def test_tongyi_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "qwen-plus"
    chosen_temperature = 0
    response_text_content = "mocked-text-response-from-model"

    provider = TongyiProvider()
    mock_response = MagicMock()
    mock_response = {
        "output": {"choices": [{"message": {"content": response_text_content}}]}
    }

    with patch.object(
        dashscope.Generation, "call", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_with(
            api_key=provider.api_key,
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
            result_format="message",
        )

        assert response.choices[0].message.content == response_text_content
