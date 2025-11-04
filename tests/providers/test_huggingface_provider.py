"""Tests for Hugging Face provider ASR functionality."""

import io
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aisuite.providers.huggingface_provider import HuggingfaceProvider
from aisuite.provider import ASRError
from aisuite.framework.message import TranscriptionResult


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")


@pytest.fixture
def huggingface_provider():
    """Create a Hugging Face provider instance for testing."""
    return HuggingfaceProvider()


@pytest.fixture
def mock_huggingface_response():
    """Create a mock Hugging Face API response for ASR."""
    return {
        "text": "Hello, this is a test transcription from Hugging Face.",
        "chunks": [
            {"text": " Hello", "timestamp": [0.0, 0.5]},
            {"text": ",", "timestamp": [0.5, 0.6]},
            {"text": " this", "timestamp": [0.6, 0.9]},
            {"text": " is", "timestamp": [0.9, 1.1]},
            {"text": " a", "timestamp": [1.1, 1.2]},
            {"text": " test", "timestamp": [1.2, 1.5]},
        ],
    }


@pytest.fixture
def mock_huggingface_response_text_only():
    """Create a mock Hugging Face API response with text only (no chunks)."""
    return {"text": "Simple transcription without timestamps."}


class TestHuggingFaceProvider:
    """Test suite for Hugging Face provider functionality."""

    def test_provider_initialization(self, huggingface_provider):
        """Test that Hugging Face provider initializes correctly."""
        assert huggingface_provider is not None
        assert hasattr(huggingface_provider, "token")
        assert huggingface_provider.token == "test-hf-token"
        assert hasattr(huggingface_provider, "audio")
        assert hasattr(huggingface_provider.audio, "transcriptions")

    def test_audio_transcriptions_create_success(
        self, huggingface_provider, mock_huggingface_response
    ):
        """Test successful audio transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_huggingface_response

        with patch("builtins.open", mock_open(read_data=b"fake audio data")), patch(
            "requests.post", return_value=mock_response
        ) as mock_post:
            result = huggingface_provider.audio.transcriptions.create(
                model="huggingface:openai/whisper-large-v3", file="test_audio.wav"
            )

            # Verify the request
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            # URL is first positional argument
            assert "api-inference.huggingface.co" in call_args.args[0]
            assert "openai/whisper-large-v3" in call_args.args[0]
            assert (
                call_args.kwargs["headers"]["Authorization"] == "Bearer test-hf-token"
            )
            assert call_args.kwargs["headers"]["Content-Type"] == "audio/wav"

            # Verify the result
            assert isinstance(result, TranscriptionResult)
            assert (
                result.text == "Hello, this is a test transcription from Hugging Face."
            )
            assert len(result.words) == 6
            assert result.words[0].word == " Hello"
            assert result.words[0].start == 0.0
            assert result.words[0].end == 0.5

    def test_audio_transcriptions_with_file_object(
        self, huggingface_provider, mock_huggingface_response
    ):
        """Test audio transcription with file-like object."""
        audio_data = io.BytesIO(b"fake audio data")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_huggingface_response

        with patch("requests.post", return_value=mock_response):
            result = huggingface_provider.audio.transcriptions.create(
                model="huggingface:openai/whisper-large-v3", file=audio_data
            )

            assert isinstance(result, TranscriptionResult)
            assert (
                result.text == "Hello, this is a test transcription from Hugging Face."
            )

    def test_audio_transcriptions_content_type_detection(self, huggingface_provider):
        """Test content type detection for different audio formats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test"}

        test_cases = [
            ("audio.wav", "audio/wav"),
            ("audio.mp3", "audio/mpeg"),  # HF API requires audio/mpeg for MP3
            ("audio.flac", "audio/flac"),
            ("audio.unknown", "audio/wav"),  # Default to wav
        ]

        for filename, expected_content_type in test_cases:
            with patch("builtins.open", mock_open(read_data=b"audio")), patch(
                "requests.post", return_value=mock_response
            ) as mock_post:
                huggingface_provider.audio.transcriptions.create(
                    model="huggingface:test-model", file=filename
                )

                call_args = mock_post.call_args
                assert (
                    call_args.kwargs["headers"]["Content-Type"] == expected_content_type
                )

    def test_audio_transcriptions_retry_503(self, huggingface_provider):
        """Test retry logic for 503 model loading error."""
        import requests

        # First response: 503 error
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503

        # Create HTTP error with response attribute
        http_error = requests.exceptions.HTTPError("Model loading")
        http_error.response = mock_response_503
        mock_response_503.raise_for_status.side_effect = http_error

        # Second response: Success
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"text": "Success after retry"}

        responses = [mock_response_503, mock_response_success]

        with patch("builtins.open", mock_open(read_data=b"audio")), patch(
            "requests.post", side_effect=responses
        ) as mock_post:
            result = huggingface_provider.audio.transcriptions.create(
                model="huggingface:test-model", file="test.wav"
            )

            # Verify retry happened
            assert mock_post.call_count == 2

            # Verify second call had x-wait-for-model header
            second_call_headers = mock_post.call_args_list[1].kwargs["headers"]
            assert second_call_headers.get("x-wait-for-model") == "true"

            assert result.text == "Success after retry"

    def test_audio_transcriptions_error_handling(self, huggingface_provider):
        """Test error handling for API failures."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch("builtins.open", mock_open(read_data=b"audio")), patch(
            "requests.post", return_value=mock_response
        ):
            with pytest.raises(ASRError, match="Hugging Face transcription error"):
                huggingface_provider.audio.transcriptions.create(
                    model="huggingface:test-model", file="test.wav"
                )

    def test_parse_response_standard_format(
        self, huggingface_provider, mock_huggingface_response
    ):
        """Test parsing response with text and chunks."""
        result = huggingface_provider.audio.transcriptions._parse_huggingface_response(
            mock_huggingface_response, "test-model"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription from Hugging Face."
        assert len(result.words) == 6
        assert result.words[0].word == " Hello"
        assert result.words[0].start == 0.0
        assert result.words[0].end == 0.5

    def test_parse_response_text_only(
        self, huggingface_provider, mock_huggingface_response_text_only
    ):
        """Test parsing response with text only (no chunks)."""
        result = huggingface_provider.audio.transcriptions._parse_huggingface_response(
            mock_huggingface_response_text_only, "test-model"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Simple transcription without timestamps."
        assert result.words is None

    def test_parse_response_string_format(self, huggingface_provider):
        """Test parsing response that is a plain string."""
        result = huggingface_provider.audio.transcriptions._parse_huggingface_response(
            "Plain string transcription", "test-model"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Plain string transcription"
        assert result.words is None

    def test_model_id_extraction(self, huggingface_provider):
        """Test that model ID is correctly extracted from model string."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "test"}

        with patch("builtins.open", mock_open(read_data=b"audio")), patch(
            "requests.post", return_value=mock_response
        ) as mock_post:
            huggingface_provider.audio.transcriptions.create(
                model="huggingface:openai/whisper-large-v3", file="test.wav"
            )

            # Verify URL contains correct model ID (URL is first positional arg)
            call_args = mock_post.call_args
            assert "openai/whisper-large-v3" in call_args.args[0]
