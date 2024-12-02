from unittest.mock import MagicMock, patch
import pytest
from aisuite.providers.baidu_provider import BaiduProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("QIANFAN_ACCESS_KEY", "test-access-key")
    monkeypatch.setenv("QIANFAN_SECRET_KEY", "test-secret-key")


def test_baidu_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "ERNIE-3.5-8K"
    chosen_temperature = 0
    response_text_content = "mocked-text-response-from-model"

    provider = BaiduProvider()
    mock_response = MagicMock()
    mock_response = {"body": {"result": response_text_content}}

    with patch.object(provider.client, "do", return_value=mock_response) as mock_create:
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
