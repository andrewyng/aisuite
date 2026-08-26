"""Tests for Synthorai provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.synthorai_provider import SynthoraiProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("SYNTHORAI_API_KEY", "test-synthorai-api-key")


@pytest.fixture
def synthorai_provider():
    """Create a Synthorai provider instance for testing."""
    return SynthoraiProvider()


class TestSynthoraiProvider:
    """Test suite for Synthorai provider initialization."""

    def test_provider_initialization(self, synthorai_provider):
        """Test that Synthorai provider initializes correctly."""
        assert synthorai_provider is not None
        assert hasattr(synthorai_provider, "client")
        assert hasattr(synthorai_provider, "transformer")
        # Ensure the base URL is properly overridden for Synthorai
        assert str(synthorai_provider.client.base_url) == "https://synthorai.io/v1/"

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("SYNTHORAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Synthorai API key is missing"):
            SynthoraiProvider()


class TestSynthoraiChatCompletions:
    """Test suite for Synthorai chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, synthorai_provider
    ):
        """Test successful chat completion request."""
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        synthorai_provider.client = mock_client_instance

        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        result = synthorai_provider.chat_completions_create(
            model="synthorai:claude-opus-5",
            messages=original_messages,
            temperature=0.7,
        )

        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="synthorai:claude-opus-5",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, synthorai_provider
    ):
        """Test error handling for API failures."""
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        synthorai_provider.client = mock_client_instance

        with pytest.raises(LLMError, match="An error occurred: API Error"):
            synthorai_provider.chat_completions_create(
                model="synthorai:claude-opus-5",
                messages=[{"role": "user", "content": "Hello"}],
            )
