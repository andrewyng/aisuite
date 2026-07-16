"""Tests for OpenRouter provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.openrouter_provider import OpenrouterProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-api-key")
    monkeypatch.setenv("OR_SITE_URL", "https://my-test-site.com")
    monkeypatch.setenv("OR_APP_NAME", "TestApp")


@pytest.fixture
def openrouter_provider():
    """Create an OpenRouter provider instance for testing."""
    return OpenrouterProvider()


class TestOpenrouterProvider:
    """Test suite for OpenRouter provider initialization."""

    def test_provider_initialization(self, openrouter_provider):
        """Test that OpenRouter provider initializes correctly."""
        assert openrouter_provider is not None
        assert hasattr(openrouter_provider, "client")
        assert hasattr(openrouter_provider, "transformer")
        # Ensure the base URL is properly overridden for OpenRouter
        assert (
            str(openrouter_provider.client.base_url) == "https://openrouter.ai/api/v1/"
        )

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenRouter API key is missing"):
            OpenrouterProvider()


class TestOpenrouterChatCompletions:
    """Test suite for OpenRouter chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, openrouter_provider
    ):
        """Test successful chat completion request."""
        # Setup mock client and response
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Inject the mock client into our provider
        openrouter_provider.client = mock_client_instance

        # Mock the message converter
        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        # Execute the method
        result = openrouter_provider.chat_completions_create(
            model="openrouter:openai/gpt-4o",
            messages=original_messages,
            temperature=0.7,
        )

        # Assertions
        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="openrouter:openai/gpt-4o",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, openrouter_provider
    ):
        """Test error handling for API failures."""
        # Setup mock client to throw an exception
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        # Inject the mock client
        openrouter_provider.client = mock_client_instance

        # Execute and assert the custom LLMError is raised
        with pytest.raises(LLMError, match="An error occurred: API Error"):
            openrouter_provider.chat_completions_create(
                model="openrouter:openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
