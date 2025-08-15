"""Tests for Deepgram provider functionality."""

import io
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from aisuite.providers.deepgram_provider import DeepgramProvider
from aisuite.provider import ASRError
from aisuite.framework.message import TranscriptionResult


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-api-key")


@pytest.fixture
def deepgram_provider():
    """Create a Deepgram provider instance for testing."""
    return DeepgramProvider()


@pytest.fixture
def mock_deepgram_response():
    """Create a mock Deepgram API response for ASR."""
    return {
        "metadata": {
            "request_id": "test-request-id",
            "duration": 10.5,
            "channels": 1,
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello, this is a test transcription from Deepgram.",
                            "confidence": 0.95,
                            "words": [
                                {
                                    "word": "hello",
                                    "start": 0.0,
                                    "end": 0.5,
                                    "confidence": 0.98,
                                    "speaker": 0,
                                }
                            ],
                        }
                    ]
                }
            ],
            "language": "en-US",
        },
    }


class TestDeepgramProvider:
    """Test suite for Deepgram provider functionality."""

    def test_provider_initialization(self, deepgram_provider):
        """Test that Deepgram provider initializes correctly."""
        assert deepgram_provider is not None
        assert hasattr(deepgram_provider, "api_key")
        assert deepgram_provider.api_key == "test-api-key"

    def test_chat_completions_create_not_implemented(self, deepgram_provider):
        """Test that chat completions are not supported."""
        with pytest.raises(
            NotImplementedError,
            match="Deepgram provider only supports audio transcription",
        ):
            deepgram_provider.chat_completions_create("model", [])

    def test_audio_transcriptions_create_success(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test successful audio transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_deepgram_response

        with patch("builtins.open", mock_open(read_data=b"fake audio data")), patch(
            "requests.post", return_value=mock_response
        ) as mock_post:
            result = deepgram_provider.audio_transcriptions_create(
                model="nova-2", file="test_audio.mp3"
            )

            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == deepgram_provider.base_url
            assert call_args[1]["headers"]["Authorization"] == "Token test-api-key"
            assert call_args[1]["params"]["model"] == "nova-2"

            # Verify result
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription from Deepgram."
            assert result.language == "en-US"
            assert result.confidence == 0.95

    def test_audio_transcriptions_create_with_file_object(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test audio transcription with file-like object."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_deepgram_response

        audio_data = io.BytesIO(b"fake audio data")

        with patch("requests.post", return_value=mock_response):
            result = deepgram_provider.audio_transcriptions_create(
                model="nova-2", file=audio_data
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription from Deepgram."

    def test_audio_transcriptions_create_with_kwargs(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test audio transcription with additional parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_deepgram_response

        with patch("builtins.open", mock_open(read_data=b"fake audio data")), patch(
            "requests.post", return_value=mock_response
        ) as mock_post:
            deepgram_provider.audio_transcriptions_create(
                model="nova-2",
                file="test_audio.mp3",
                diarize=True,
                punctuate=True,
                language="en-US",
            )

            # Verify parameters were passed
            call_params = mock_post.call_args[1]["params"]
            assert call_params["diarize"] is True
            assert call_params["punctuate"] is True
            assert call_params["language"] == "en-US"

    def test_audio_transcriptions_create_api_error(self, deepgram_provider):
        """Test handling of Deepgram API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch("builtins.open", mock_open(read_data=b"fake audio data")), patch(
            "requests.post", return_value=mock_response
        ):
            with pytest.raises(ASRError, match="Deepgram API error: 400 - Bad Request"):
                deepgram_provider.audio_transcriptions_create(
                    model="nova-2", file="test_audio.mp3"
                )

    def test_audio_transcriptions_create_request_exception(self, deepgram_provider):
        """Test handling of request exceptions."""
        with patch("builtins.open", mock_open(read_data=b"fake audio data")), patch(
            "requests.post", side_effect=requests.RequestException("Network error")
        ):
            with pytest.raises(
                ASRError, match="Deepgram API request error: Network error"
            ):
                deepgram_provider.audio_transcriptions_create(
                    model="nova-2", file="test_audio.mp3"
                )

    def test_parse_deepgram_response_complete(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test parsing complete Deepgram response."""
        result = deepgram_provider._parse_deepgram_response(mock_deepgram_response)

        # Basic fields
        assert result.text == "Hello, this is a test transcription from Deepgram."
        assert result.language == "en-US"
        assert result.confidence == 0.95

        # Words
        assert len(result.words) == 1
        word = result.words[0]
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.98

        # Alternatives and Channels
        assert len(result.alternatives) == 1
        assert len(result.channels) == 1

    def test_parse_deepgram_response_empty_channels(self, deepgram_provider):
        """Test parsing response with empty channels."""
        empty_response = {"results": {"channels": []}}

        result = deepgram_provider._parse_deepgram_response(empty_response)
        assert result.text == ""
        assert result.language is None
