import pytest
from unittest.mock import MagicMock, patch
from aisuite.providers.openai_provider import OpenaiProvider


@pytest.fixture(autouse=True)
def set_api_key_env(monkeypatch):
    """Fixture to set the OPENAI_API_KEY environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_openai_provider_chat_completions_create():
    """
    Test: Validate that OpenaiProvider.chat_completions_create correctly calls
    client.chat.completions.create via the provider instance.
    """
    message_history = [{"role": "user", "content": "Hello"}]
    model = "openai-gpt-4o"
    temperature = 0.7

    expected_response = {"choices": [{"message": {"content": "Hello back!"}}]}

    provider = OpenaiProvider()

    fake_client = MagicMock()
    provider.client = fake_client
    fake_chat = MagicMock()
    fake_client.chat = fake_chat
    fake_chat.completions = MagicMock()
    fake_chat.completions.create.return_value = expected_response

    with patch.object(
        fake_chat.completions, "create", return_value=expected_response
    ) as mock_create:
        response = provider.chat_completions_create(
            model=model,
            messages=message_history,
            temperature=temperature,
        )

        mock_create.assert_called_once_with(
            model=model,
            messages=provider.transformer.convert_request(message_history),
            temperature=temperature,
        )
        assert response == expected_response
