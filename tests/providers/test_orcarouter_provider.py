"""Tests for OrcaRouter provider functionality."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aisuite.providers.orcarouter_provider import OrcarouterProvider
from aisuite.provider import LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("ORCAROUTER_API_KEY", "test-orcarouter-api-key")


@pytest.fixture
def orcarouter_provider():
    """Create an OrcaRouter provider instance for testing."""
    return OrcarouterProvider()


class TestOrcarouterProvider:
    """Test suite for OrcaRouter provider initialization."""

    def test_provider_initialization(self, orcarouter_provider):
        """Test that OrcaRouter provider initializes correctly."""
        assert orcarouter_provider is not None
        assert hasattr(orcarouter_provider, "client")
        assert hasattr(orcarouter_provider, "aclient")
        assert hasattr(orcarouter_provider, "transformer")
        # Ensure the base URL is properly overridden for OrcaRouter
        assert (
            str(orcarouter_provider.client.base_url) == "https://api.orcarouter.ai/v1/"
        )

    def test_provider_missing_api_key(self, monkeypatch):
        """Test initialization fails when API key is missing."""
        monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OrcaRouter API key is missing"):
            OrcarouterProvider()

    def test_optional_attribution_headers(self, monkeypatch):
        """Attribution headers are sent when the optional env vars are set."""
        monkeypatch.setenv("ORCAROUTER_SITE_URL", "https://example.com")
        monkeypatch.setenv("ORCAROUTER_APP_NAME", "my-app")

        provider = OrcarouterProvider()

        assert provider.client.default_headers["HTTP-Referer"] == "https://example.com"
        assert provider.client.default_headers["X-Title"] == "my-app"

    def test_explicit_headers_win_over_env(self, monkeypatch):
        """A caller-supplied header is not overwritten by the env var."""
        monkeypatch.setenv("ORCAROUTER_SITE_URL", "https://example.com")

        provider = OrcarouterProvider(
            default_headers={"HTTP-Referer": "https://explicit.example"}
        )

        assert (
            provider.client.default_headers["HTTP-Referer"]
            == "https://explicit.example"
        )


class TestOrcarouterChatCompletions:
    """Test suite for OrcaRouter chat completions functionality."""

    @patch("openai.OpenAI")
    @patch.object(OpenAICompliantMessageConverter, "convert_request")
    def test_chat_completions_create_success(
        self, mock_convert, mock_openai_class, orcarouter_provider
    ):
        """Test successful chat completion request."""
        # Setup mock client and response
        mock_client_instance = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        # Inject the mock client into our provider
        orcarouter_provider.client = mock_client_instance

        # Mock the message converter
        mock_converted_messages = [{"role": "user", "content": "Transformed"}]
        mock_convert.return_value = mock_converted_messages

        original_messages = [{"role": "user", "content": "Hello"}]

        # Execute the method
        result = orcarouter_provider.chat_completions_create(
            model="openai/gpt-5.5",
            messages=original_messages,
            temperature=0.7,
        )

        # Assertions
        mock_convert.assert_called_once_with(original_messages)
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="openai/gpt-5.5",
            messages=mock_converted_messages,
            temperature=0.7,
        )
        assert result == mock_response

    @patch("openai.OpenAI")
    def test_chat_completions_create_error_handling(
        self, mock_openai_class, orcarouter_provider
    ):
        """Test error handling for API failures."""
        # Setup mock client to throw an exception
        mock_client_instance = mock_openai_class.return_value
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        # Inject the mock client
        orcarouter_provider.client = mock_client_instance

        # Execute and assert the custom LLMError is raised
        with pytest.raises(LLMError, match="An error occurred: API Error"):
            orcarouter_provider.chat_completions_create(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_chat_completions_create_stream_passes_chunks_through(
        self, orcarouter_provider
    ):
        """Streaming yields the OpenAI-shaped chunks unchanged, with stream=True set."""
        chunks = [SimpleNamespace(id="1"), SimpleNamespace(id="2")]
        orcarouter_provider.client.chat.completions.create = MagicMock(
            return_value=iter(chunks)
        )

        received = list(
            orcarouter_provider.chat_completions_create_stream(
                model="openai/gpt-5.5",
                messages=[{"role": "user", "content": "Hello"}],
            )
        )

        assert received == chunks
        _, kwargs = orcarouter_provider.client.chat.completions.create.call_args
        assert kwargs["stream"] is True
        assert kwargs["model"] == "openai/gpt-5.5"

    @pytest.mark.asyncio
    async def test_native_async_chat_completions_create(self, orcarouter_provider):
        """The provider overrides the async path with a native AsyncOpenAI call."""
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="async hi"))]
        )
        orcarouter_provider.aclient.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        response = await orcarouter_provider.achat_completions_create(
            "openai/gpt-5.5", [{"role": "user", "content": "hi"}]
        )

        assert response.choices[0].message.content == "async hi"
        orcarouter_provider.aclient.chat.completions.create.assert_awaited_once()
