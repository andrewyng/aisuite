"""Tests for base Provider class and ASRError."""

import pytest
from unittest.mock import MagicMock

from aisuite.provider import Provider, ASRError
from aisuite.framework.message import TranscriptionResult


class MockProvider(Provider):
    """Mock provider for testing."""

    def chat_completions_create(self, model, messages):
        return MagicMock()


class MockASRProvider(Provider):
    """Mock provider that implements ASR."""

    def chat_completions_create(self, model, messages):
        return MagicMock()

    def audio_transcriptions_create(self, model, file, **kwargs):
        return TranscriptionResult(
            text="Mock transcription result", language="en", confidence=0.9
        )


class TestProvider:
    """Test suite for base Provider class."""

    def test_provider_is_abstract(self):
        """Test that Provider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Provider()

    def test_provider_audio_transcriptions_create_default_implementation(self):
        """Test that default ASR implementation raises NotImplementedError."""
        provider = MockProvider()

        with pytest.raises(
            NotImplementedError,
            match="Provider MockProvider does not support audio transcription",
        ):
            provider.audio_transcriptions_create("whisper-1", "audio.mp3")

    def test_provider_asr_implementation_works(self):
        """Test that providers can successfully implement ASR."""
        provider = MockASRProvider()

        result = provider.audio_transcriptions_create("model", "file.mp3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Mock transcription result"
        assert result.language == "en"
        assert result.confidence == 0.9


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
