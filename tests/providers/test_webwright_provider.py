"""Tests for WebWright provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.webwright_provider import WebwrightProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("WEBWRIGHT_API_KEY", "test-webwright-api-key")


@pytest.fixture
def webwright_provider():
    """Create a WebWright provider instance for testing."""
    return WebwrightProvider()


class TestWebwrightProvider:
    """Test suite for WebWright provider initialization."""

    def test_provider_initialization(self, webwright_provider):
        """Test that WebWright provider initializes correctly."""
        assert webwright_provider is not None
        assert hasattr(webwright_provider, "client")
        assert hasattr(webwright_provider, "transformer")
        # Ensure the base URL is properly set for WebWright
        assert str(webwright_provider.client.base_url) == "https://webwright.ai/v1/"

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("WEBWRIGHT_API_KEY", raising=False)
        with pytest.raises(ValueError, match="WebWright API key is missing"):
            WebwrightProvider()


class TestWebwrightChatCompletions:
    """Test suite for WebWright chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, webwright_provider
    ):
        """Test successful chat completion request."""
        # Setup mock client and response
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Inject the mock client into our provider
        webwright_provider.client = mock_client_instance

        # Mock the message converter
        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        # Execute the method
        result = webwright_provider.chat_completions_create(
            model="wren-r1",
            messages=original_messages,
            temperature=0.7,
        )

        # Assertions
        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="wren-r1",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, webwright_provider
    ):
        """Test error handling for API failures."""
        # Setup mock client to throw an exception
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        # Inject the mock client
        webwright_provider.client = mock_client_instance

        # Execute and assert the custom LLMError is raised
        with pytest.raises(LLMError, match="An error occurred: API Error"):
            webwright_provider.chat_completions_create(
                model="wren-r1",
                messages=[{"role": "user", "content": "Hello"}],
            )
