"""Tests for Requesty provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.requesty_provider import RequestyProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("REQUESTY_API_KEY", "test-requesty-api-key")
    monkeypatch.setenv("REQUESTY_SITE_URL", "https://my-test-site.com")
    monkeypatch.setenv("REQUESTY_APP_NAME", "TestApp")


@pytest.fixture
def requesty_provider():
    """Create a Requesty provider instance for testing."""
    return RequestyProvider()


class TestRequestyProvider:
    """Test suite for Requesty provider initialization."""

    def test_provider_initialization(self, requesty_provider):
        """Test that Requesty provider initializes correctly."""
        assert requesty_provider is not None
        assert hasattr(requesty_provider, "client")
        assert hasattr(requesty_provider, "transformer")
        # Ensure the base URL is properly overridden for Requesty
        assert (
            str(requesty_provider.client.base_url) == "https://router.requesty.ai/v1/"
        )

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Requesty API key is missing"):
            RequestyProvider()


class TestRequestyChatCompletions:
    """Test suite for Requesty chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, requesty_provider
    ):
        """Test successful chat completion request."""
        # Setup mock client and response
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Inject the mock client into our provider
        requesty_provider.client = mock_client_instance

        # Mock the message converter
        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        # Execute the method
        result = requesty_provider.chat_completions_create(
            model="requesty:openai/gpt-4o-mini",
            messages=original_messages,
            temperature=0.7,
        )

        # Assertions
        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="requesty:openai/gpt-4o-mini",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, requesty_provider
    ):
        """Test error handling for API failures."""
        # Setup mock client to throw an exception
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        # Inject the mock client
        requesty_provider.client = mock_client_instance

        # Execute and assert the custom LLMError is raised
        with pytest.raises(LLMError, match="An error occurred: API Error"):
            requesty_provider.chat_completions_create(
                model="requesty:openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )
