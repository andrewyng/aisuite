from unittest.mock import MagicMock, patch

from aisuite.providers.volcano_provider import VolcanoProvider


def test_volcano_provider_initializes_and_sends_chat_request(monkeypatch):
    monkeypatch.setenv("VOLCANO_API_KEY", "test-api-key")

    provider = VolcanoProvider()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "mocked-text-response-from-volcano"

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=mock_response,
    ) as mock_create:
        response = provider.chat_completions_create(
            model="deepseek-r1-250120",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7,
        )

        mock_create.assert_called_with(
            model="deepseek-r1-250120",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7,
        )
        assert response.choices[0].message.content == "mocked-text-response-from-volcano"
