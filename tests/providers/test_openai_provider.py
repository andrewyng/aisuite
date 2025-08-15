"""Tests for OpenAI provider functionality."""

import io
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aisuite.providers.openai_provider import OpenaiProvider
from aisuite.provider import ASRError
from aisuite.framework.message import TranscriptionResult, Segment, Word


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


@pytest.fixture
def openai_provider():
    """Create an OpenAI provider instance for testing."""
    return OpenaiProvider()


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response for ASR."""
    mock_response = MagicMock()
    mock_response.text = "Hello, this is a test transcription."
    mock_response.task = "transcribe"
    mock_response.language = "en"
    mock_response.duration = 10.5
    mock_response.segments = None
    mock_response.words = None
    return mock_response


class TestOpenAIProvider:
    """Test suite for OpenAI provider functionality."""

    def test_provider_initialization(self, openai_provider):
        """Test that OpenAI provider initializes correctly."""
        assert openai_provider is not None
        assert hasattr(openai_provider, "client")


class TestOpenAIASR:
    """Test suite for OpenAI ASR functionality."""

    def test_audio_transcriptions_create_success(
        self, openai_provider, mock_openai_response
    ):
        """Test successful audio transcription."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ):
            result = openai_provider.audio_transcriptions_create(
                model="whisper-1", file="test_audio.mp3"
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."
            assert result.task == "transcribe"
            assert result.language == "en"

    def test_audio_transcriptions_create_with_file_object(
        self, openai_provider, mock_openai_response
    ):
        """Test audio transcription with file-like object."""
        audio_data = io.BytesIO(b"fake audio data")

        with patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ):
            result = openai_provider.audio_transcriptions_create(
                model="whisper-1", file=audio_data
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."

    def test_audio_transcriptions_create_with_kwargs(
        self, openai_provider, mock_openai_response
    ):
        """Test audio transcription with additional parameters."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            return_value=mock_openai_response,
        ) as mock_create:
            openai_provider.audio_transcriptions_create(
                model="whisper-1",
                file="test_audio.mp3",
                language="en",
                temperature=0.5,
            )

            # Verify the API was called with the correct parameters
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["language"] == "en"
            assert call_kwargs["temperature"] == 0.5

    def test_audio_transcriptions_create_error_handling(self, openai_provider):
        """Test error handling for API failures."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            openai_provider.client.audio.transcriptions,
            "create",
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(ASRError, match="OpenAI transcription error: API Error"):
                openai_provider.audio_transcriptions_create(
                    model="whisper-1", file="test_audio.mp3"
                )

    def test_parse_openai_response_with_segments_and_words(self, openai_provider):
        """Test parsing OpenAI response with segments and words."""
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.task = "transcribe"
        mock_response.language = "en"
        mock_response.duration = 5.0

        # Mock segment
        mock_segment = MagicMock()
        mock_segment.id = 0
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "Hello world"
        mock_response.segments = [mock_segment]

        # Mock words
        mock_word = MagicMock()
        mock_word.word = "Hello"
        mock_word.start = 0.0
        mock_word.end = 0.5
        mock_response.words = [mock_word]

        result = openai_provider._parse_openai_response(mock_response)

        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert len(result.words) == 1
        assert isinstance(result.segments[0], Segment)
        assert isinstance(result.words[0], Word)

    def test_parse_openai_response_missing_attributes(self, openai_provider):
        """Test parsing response with missing optional attributes."""
        mock_response = MagicMock()
        mock_response.text = "Test"
        # Set missing attributes to raise AttributeError
        del mock_response.task
        del mock_response.language
        del mock_response.duration
        mock_response.segments = None
        mock_response.words = None

        result = openai_provider._parse_openai_response(mock_response)

        assert result.text == "Test"
        assert result.task is None
        assert result.language is None
        assert result.duration is None
