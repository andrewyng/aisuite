import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.ollama_provider import OllamaProvider


@pytest.fixture(autouse=True)
def set_api_url_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OLLAMA_API_URL", "http://localhost:11434")


def test_completion():
    """Test that completions request successfully."""

    user_greeting = "Howdy!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "best-model-ever"
    chosen_temperature = 0.77
    response_text_content = "mocked-text-response-from-ollama-model"

    ollama = OllamaProvider()
    mock_response = {"message": {"content": response_text_content}}

    with patch(
        "httpx.post",
        return_value=MagicMock(status_code=200, json=lambda: mock_response),
    ) as mock_post:
        response = ollama.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={
                "model": selected_model,
                "messages": message_history,
                "stream": False,
                "temperature": chosen_temperature,
            },
            timeout=30,
        )

        assert response.choices[0].message.content == response_text_content
