"""Tests for API Route provider functionality."""

from unittest.mock import MagicMock, patch

import pytest

from aisuite.provider import LLMError
from aisuite.providers.api_route_provider import ApirouteProvider
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Set API Route environment variables for tests."""
    monkeypatch.setenv("API_ROUTE_API_KEY", "test-api-route-key")


@pytest.fixture
def api_route_provider():
    """Create an API Route provider instance for testing."""
    return ApirouteProvider()


class TestApiRouteProvider:
    """Test API Route provider initialization."""

    def test_provider_initialization(self, api_route_provider):
        """API Route initializes with the expected default base URL."""
        assert api_route_provider is not None
        assert hasattr(api_route_provider, "client")
        assert hasattr(api_route_provider, "transformer")
        assert (
            str(api_route_provider.client.base_url)
            == "https://global.api-route.com/v1/"
        )

    def test_provider_missing_api_key(self, monkeypatch):
        """Initialization fails when the API key is missing."""
        monkeypatch.delenv("API_ROUTE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API Route API key is missing"):
            ApirouteProvider()


class TestApiRouteChatCompletions:
    """Test API Route chat completions."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, api_route_provider
    ):
        """A successful request is forwarded to the OpenAI-compatible endpoint."""
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        api_route_provider.client = mock_client_instance

        converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = converted_messages
        original_messages = [{"role": "user", "content": "Hello"}]

        result = api_route_provider.chat_completions_create(
            model="claude-sonnet-4-6",
            messages=original_messages,
            temperature=0.7,
        )

        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="claude-sonnet-4-6",
            messages=converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, api_route_provider
    ):
        """API failures are surfaced as aisuite LLMError exceptions."""
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        api_route_provider.client = mock_client_instance

        with pytest.raises(LLMError, match="An error occurred: API Error"):
            api_route_provider.chat_completions_create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "Hello"}],
            )
