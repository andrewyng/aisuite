"""Tests for Deepgram provider functionality."""

import io
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aisuite.providers.deepgram_provider import DeepgramProvider
from aisuite.provider import ASRError
from aisuite.framework.message import (
    TranscriptionResult,
    TranscriptionOptions,
    StreamingTranscriptionChunk,
)


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
        assert hasattr(deepgram_provider, "audio")
        assert hasattr(deepgram_provider.audio, "transcriptions")

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
        mock_sdk_response = MagicMock()
        mock_sdk_response.to_dict.return_value = mock_deepgram_response
        mock_sdk_response.model_dump.return_value = mock_deepgram_response

        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            deepgram_provider.client.listen.v1.media,
            "transcribe_file",
            return_value=mock_sdk_response,
        ):
            result = deepgram_provider.audio.transcriptions.create(
                model="deepgram:nova-2", file="test_audio.mp3"
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription from Deepgram."
            assert result.language == "en-US"
            assert result.confidence == 0.95

    def test_audio_transcriptions_create_with_file_object(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test audio transcription with file-like object."""
        audio_data = io.BytesIO(b"fake audio data")

        mock_response = MagicMock()
        mock_response.to_dict.return_value = mock_deepgram_response
        mock_response.model_dump.return_value = mock_deepgram_response

        with patch.object(
            deepgram_provider.client.listen.v1.media,
            "transcribe_file",
            return_value=mock_response,
        ):
            result = deepgram_provider.audio.transcriptions.create(
                model="deepgram:nova-2", file=audio_data
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription from Deepgram."

    def test_audio_transcriptions_create_with_options(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test audio transcription with TranscriptionOptions."""
        options = TranscriptionOptions(
            language="en",
            enable_speaker_diarization=True,
            enable_automatic_punctuation=True,
        )

        mock_response = MagicMock()
        mock_response.to_dict.return_value = mock_deepgram_response
        mock_response.model_dump.return_value = mock_deepgram_response

        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            deepgram_provider.client.listen.v1.media,
            "transcribe_file",
            return_value=mock_response,
        ) as mock_transcribe:
            result = deepgram_provider.audio.transcriptions.create(
                model="deepgram:nova-2", file="test_audio.mp3", options=options
            )

            mock_transcribe.assert_called_once()
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription from Deepgram."

    def test_audio_transcriptions_create_error_handling(self, deepgram_provider):
        """Test error handling for API failures."""
        with patch(
            "builtins.open", mock_open(read_data=b"fake audio data")
        ), patch.object(
            deepgram_provider.client.listen.v1.media,
            "transcribe_file",
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(
                ASRError, match="Deepgram transcription error: API Error"
            ):
                deepgram_provider.audio.transcriptions.create(
                    model="deepgram:nova-2", file="test_audio.mp3"
                )

    @pytest.mark.asyncio
    async def test_audio_transcriptions_create_stream_output(self, deepgram_provider):
        """Test streaming audio transcription with single connection and chunking."""
        import numpy as np

        # Mock audio file data (simulate 16kHz mono audio)
        audio_samples = 48000  # 3 seconds of 16kHz audio
        mock_audio_data = np.zeros(audio_samples, dtype=np.float32)

        # Create async context manager mock for v5 API
        mock_connection = MagicMock()
        mock_connection.send = MagicMock()
        mock_connection.on = MagicMock()

        async def mock_connect(*args, **kwargs):
            # Return an async context manager
            class MockAsyncContextManager:
                async def __aenter__(self):
                    return mock_connection

                async def __aexit__(self, *args):
                    pass

            return MockAsyncContextManager()

        with patch(
            "soundfile.read", return_value=(mock_audio_data, 16000)
        ), patch.object(
            deepgram_provider.client.listen.v1,
            "connect",
            side_effect=mock_connect,
        ):
            result = deepgram_provider.audio.transcriptions.create_stream_output(
                model="deepgram:nova-2",
                file="test_audio.mp3",
                chunk_size_minutes=1.0,  # Test with smaller chunks
            )

            # Test that it returns an async generator
            assert hasattr(result, "__aiter__")

    def test_parse_deepgram_response_complete(
        self, deepgram_provider, mock_deepgram_response
    ):
        """Test parsing complete Deepgram response."""
        result = deepgram_provider.audio.transcriptions._parse_deepgram_response(
            mock_deepgram_response
        )

        assert result.text == "Hello, this is a test transcription from Deepgram."
        assert result.language == "en-US"
        assert result.confidence == 0.95

        assert len(result.words) == 1
        word = result.words[0]
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.98

    def test_parse_deepgram_response_empty_channels(self, deepgram_provider):
        """Test parsing response with empty channels."""
        empty_response = {"results": {"channels": []}}

        result = deepgram_provider.audio.transcriptions._parse_deepgram_response(
            empty_response
        )
        assert result.text == ""
        assert result.language is None
