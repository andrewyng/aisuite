import os
import pytest

from unittest.mock import patch, MagicMock

from aisuite.providers.centml_provider import CentmlProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("CENTML_API_KEY", "test-api-key")


def test_centml_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    headers = {
        "Authorization": f"Bearer {os.getenv('CENTML_API_KEY')}",
        "Content-Type": "application/json",
    }

    provider = CentmlProvider()

    # Create a dictionary that matches the expected JSON response structure
    mock_json_response = {"choices": [{"message": {"content": response_text_content}}]}

    with patch(
        "httpx.post",
        return_value=MagicMock(status_code=200, json=lambda: mock_json_response),
    ) as mock_post:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        mock_post.assert_called_once_with(
            provider.BASE_URL,
            json={
                "model": selected_model,
                "messages": message_history,
                "temperature": chosen_temperature,
            },
            timeout=30,
            headers=headers,
        )

        assert response.choices[0].message.content == response_text_content
