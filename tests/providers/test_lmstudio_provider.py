from dataclasses import dataclass
from unittest.mock import patch

from aisuite.providers.lmstudio_provider import LmstudioProvider


@dataclass
class MockResponse:
    content: str


def test_lmstudio_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    config = {
        "temperature": 0.6,
        "maxTokens": 5000,
    }
    response_text_content = "mocked-text-response-from-model"

    provider = LmstudioProvider()
    mock_response = MockResponse(response_text_content)

    with patch.object(provider, "_chat", return_value=mock_response) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            config=config,
        )

        mock_create.assert_called_with(
            model=selected_model, messages=message_history, config=config
        )

        assert response.choices[0].message.content == response_text_content
