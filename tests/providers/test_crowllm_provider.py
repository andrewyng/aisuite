"""Tests for the CrowLLM provider."""

from unittest.mock import MagicMock, patch
import pytest

from aisuite.providers.crowllm_provider import CrowllmProvider
from aisuite.framework.chat_completion_response import ChatCompletionResponse


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set the CrowLLM API key environment variable for tests."""
    monkeypatch.setenv("CROWLLM_API_KEY", "test-api-key")


def test_crowllm_provider():
    """Test that the provider is initialized and chat completions are requested."""

    message_history = [{"role": "user", "content": "Hello!"}]
    selected_model = "glm-5.3"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = CrowllmProvider()
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


def test_crowllm_requires_api_key(monkeypatch):
    """A missing key must fail at construction time with a clear message."""
    monkeypatch.delenv("CROWLLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CrowLLM API key is missing"):
        CrowllmProvider()
