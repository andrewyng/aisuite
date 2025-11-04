"""Tests for base Provider class and ASRError."""

import pytest
from unittest.mock import MagicMock
from typing import Union, BinaryIO, Optional, AsyncGenerator

from aisuite.provider import Provider, ASRError, Audio
from aisuite.framework.message import (
    TranscriptionResult,
    TranscriptionOptions,
    StreamingTranscriptionChunk,
)


class MockProvider(Provider):
    """Mock provider for testing (no audio support)."""

    def chat_completions_create(self, model, messages):
        return MagicMock()


class MockTranscription(Audio.Transcription):
    """Mock transcription implementation."""

    def create(
        self,
        model: str,
        file: Union[str, BinaryIO],
        options: Optional[TranscriptionOptions] = None,
        **kwargs,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text="Mock transcription result", language="en", confidence=0.9
        )


class MockAudio(Audio):
    """Mock audio implementation."""

    def __init__(self):
        super().__init__()
        self.transcriptions = MockTranscription()


class MockASRProvider(Provider):
    """Mock provider that implements ASR."""

    def __init__(self):
        super().__init__()
        self.audio = MockAudio()

    def chat_completions_create(self, model, messages):
        return MagicMock()


class TestProvider:
    """Test suite for base Provider class."""

    def test_provider_is_abstract(self):
        """Test that Provider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Provider()

    def test_provider_without_audio_support(self):
        """Test that provider without audio support has None audio attribute."""
        provider = MockProvider()
        assert provider.audio is None

    def test_provider_asr_implementation_works(self):
        """Test that providers can successfully implement ASR."""
        provider = MockASRProvider()

        assert provider.audio is not None
        assert hasattr(provider.audio, "transcriptions")

        result = provider.audio.transcriptions.create("model", "file.mp3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Mock transcription result"
        assert result.language == "en"
        assert result.confidence == 0.9

    def test_transcription_base_class_not_implemented(self):
        """Test that base Transcription class raises NotImplementedError."""
        transcription = Audio.Transcription()

        with pytest.raises(NotImplementedError, match="Transcription not supported"):
            transcription.create("model", "file.mp3")

    def test_audio_base_class_initialization(self):
        """Test that base Audio class initializes correctly."""
        audio = Audio()
        assert audio.transcriptions is None


class TestASRError:
    """Test suite for ASRError exception."""

    def test_asr_error_creation_and_inheritance(self):
        """Test ASRError creation and inheritance."""
        error = ASRError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, ASRError)
        assert isinstance(error, Exception)

    def test_asr_error_raising_and_catching(self):
        """Test raising and catching ASRError."""
        with pytest.raises(ASRError, match="Specific ASR error"):
            raise ASRError("Specific ASR error")

        # Test that it can be caught as Exception too
        with pytest.raises(Exception):
            raise ASRError("Generic catch test")

    def test_asr_error_chaining(self):
        """Test ASRError exception chaining."""
        original_error = ValueError("Original error")

        with pytest.raises(ASRError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise ASRError("Wrapped error") from e

        assert exc_info.value.__cause__ == original_error
