import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.featherless_provider import FeatherlessProvider

@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-api-key")
def test_featherless_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""
    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"
    provider = FeatherlessProvider()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = response_text_content
    with patch.object(
        provider.client.chat.completions,
        "create",
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
def test_missing_api_key_raises_value_error(monkeypatch):
    """
    Test that initializing FeatherlessProvider without an API key raises a ValueError.
    """
    # Delete the FEATHERLESS_API_KEY to simulate a missing API key.
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    
    with pytest.raises(ValueError, match="Featherless API key is missing"):
        FeatherlessProvider()
def test_explicit_api_key_overrides_env(monkeypatch):
    """
    Test that providing an explicit API key in the configuration takes precedence over the environment variable.
    Also verifies that additional configuration parameters are correctly passed to the OpenAI client.
    """
    # Set the environment variable to a different API key.
    monkeypatch.setenv("FEATHERLESS_API_KEY", "env-api-key")
    explicit_key = "explicit-api-key"
    extra_param = "extra-value"
    
    # Patch openai.OpenAI to intercept its instantiation.
    with patch("aisuite.providers.featherless_provider.openai.OpenAI") as mock_openai_constructor:
        fake_instance = MagicMock()
        mock_openai_constructor.return_value = fake_instance
        
        # Create the provider with an explicit API key and an extra parameter.
        provider = FeatherlessProvider(api_key=explicit_key, extra_param=extra_param)
        
        # Verify openai.OpenAI was called once.
        mock_openai_constructor.assert_called_once()
        # Retrieve the arguments passed to openai.OpenAI.
        _, called_kwargs = mock_openai_constructor.call_args
        
        # Check that the explicit API key was used instead of the env variable.
        assert called_kwargs["api_key"] == explicit_key
        # Check the extra parameter is passed correctly.
        assert called_kwargs["extra_param"] == extra_param
        # Ensure the base_url is set correctly.
        assert called_kwargs["base_url"] == "https://api.featherless.ai/v1"