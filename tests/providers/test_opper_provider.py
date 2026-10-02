"""Tests for Opper provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.opper_provider import OpperProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OPPER_API_KEY", "test-opper-api-key")


@pytest.fixture
def opper_provider():
    """Create an Opper provider instance for testing."""
    return OpperProvider()


class TestOpperProvider:
    """Test suite for Opper provider initialization."""

    def test_provider_initialization(self, opper_provider):
        """Test that Opper provider initializes correctly."""
        assert opper_provider is not None
        assert hasattr(opper_provider, "client")
        assert hasattr(opper_provider, "transformer")
        # Ensure the base URL is properly overridden for Opper
        assert str(opper_provider.client.base_url) == "https://api.opper.ai/v3/compat/"

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("OPPER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Opper API key is missing"):
            OpperProvider()


class TestOpperChatCompletions:
    """Test suite for Opper chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, opper_provider
    ):
        """Test successful chat completion request."""
        # Setup mock client and response
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Inject the mock client into our provider
        opper_provider.client = mock_client_instance

        # Mock the message converter
        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        # Execute the method
        result = opper_provider.chat_completions_create(
            model="opper:gpt-5.4-mini",
            messages=original_messages,
            temperature=0.7,
        )

        # Assertions
        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="opper:gpt-5.4-mini",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, opper_provider
    ):
        """Test error handling for API failures."""
        # Setup mock client to throw an exception
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        # Inject the mock client
        opper_provider.client = mock_client_instance

        # Execute and assert the custom LLMError is raised
        with pytest.raises(LLMError, match="An error occurred: API Error"):
            opper_provider.chat_completions_create(
                model="opper:gpt-5.4-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )
