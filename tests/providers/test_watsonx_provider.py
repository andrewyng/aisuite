from unittest.mock import MagicMock, patch

import pytest

try:
    from ibm_watsonx_ai.metanames import GenChatParamsMetaNames as GenChatParams
except Exception as e:
    pytest.skip(f"Skipping test due to import error: {e}", allow_module_level=True)

from aisuite.providers.watsonx_provider import WatsonxProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("WATSONX_SERVICE_URL", "https://watsonx-service-url.com")
    monkeypatch.setenv("WATSONX_API_KEY", "test-api-key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")


@pytest.mark.skip(reason="Skipping due to version compatibility issue on python 3.11")
def test_watsonx_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.7
    response_text_content = "mocked-text-response-from-model"

    provider = WatsonxProvider()
    mock_response = {"choices": [{"message": {"content": response_text_content}}]}

    with patch(
        "aisuite.providers.watsonx_provider.ModelInference"
    ) as mock_model_inference:
        mock_model = MagicMock()
        mock_model_inference.return_value = mock_model
        mock_model.chat.return_value = mock_response

        response = provider.chat_completions_create(
            messages=message_history,
            model=selected_model,
            temperature=chosen_temperature,
        )

        # Assert that ModelInference was called with correct arguments.
        mock_model_inference.assert_called_once()
        args, kwargs = mock_model_inference.call_args
        assert kwargs["model_id"] == selected_model
        assert kwargs["project_id"] == provider.project_id

        # Assert that the credentials have the correct API key and service URL.
        credentials = kwargs["credentials"]
        assert credentials.api_key == provider.api_key
        assert credentials.url == provider.service_url

        # Assert that chat was called with correct history and params
        mock_model.chat.assert_called_once_with(
            messages=message_history,
            params={GenChatParams.TEMPERATURE: chosen_temperature},
        )

        assert response.choices[0].message.content == response_text_content
