"""Tests for Pinstripes provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.pinstripes_provider import PinestripesProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("PINSTRIPES_API_KEY", "test-pinstripes-api-key")


@pytest.fixture
def pinstripes_provider():
    """Create a Pinstripes provider instance for testing."""
    return PinestripesProvider()


class TestPinestripesProvider:
    """Test suite for Pinstripes provider initialization."""

    def test_provider_initialization(self, pinstripes_provider):
        """Test that Pinstripes provider initializes correctly."""
        assert pinstripes_provider is not None
        assert hasattr(pinstripes_provider, "client")
        assert hasattr(pinstripes_provider, "transformer")
        assert str(pinstripes_provider.client.base_url) == "https://pinstripes.io/v1/"

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("PINSTRIPES_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Pinstripes API key is missing"):
            PinestripesProvider()


class TestPinestripesChatCompletions:
    """Test suite for Pinstripes chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, pinstripes_provider
    ):
        """Test successful chat completion request."""
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Inject the mock client into our provider
        pinstripes_provider.client = mock_client_instance

        # Mock the message converter
        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        result = pinstripes_provider.chat_completions_create(
            model="ps/deepseek-v4-flash",
            messages=original_messages,
            temperature=0.7,
        )

        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="ps/deepseek-v4-flash",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, pinstripes_provider
    ):
        """Test error handling for API failures."""
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        # Inject the mock client
        pinstripes_provider.client = mock_client_instance

        with pytest.raises(LLMError, match="An error occurred: API Error"):
            pinstripes_provider.chat_completions_create(
                model="ps/deepseek-v4-flash",
                messages=[{"role": "user", "content": "Hello"}],
            )
