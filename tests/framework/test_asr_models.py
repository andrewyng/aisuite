"""Tests for ASR framework data models."""

import pytest
from pydantic import ValidationError

from aisuite.framework.message import (
    Word,
    Segment,
    Alternative,
    Channel,
    TranscriptionResult,
)


class TestWord:
    """Test suite for Word model."""

    def test_word_creation_and_validation(self):
        """Test Word creation with all fields and validation."""
        # Test basic creation
        word = Word(word="hello", start=0.0, end=0.5)
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence is None

        # Test with all fields
        word_full = Word(
            word="hello",
            start=0.0,
            end=0.5,
            confidence=0.95,
            speaker=1,
            punctuated_word="Hello,",
        )
        assert word_full.confidence == 0.95
        assert word_full.speaker == 1

        # Test validation
        with pytest.raises(ValidationError):
            Word()  # Missing required fields


class TestSegment:
    """Test suite for Segment model."""

    def test_segment_creation_and_validation(self):
        """Test Segment creation and validation."""
        # Test basic creation
        segment = Segment(id=0, seek=0, start=0.0, end=5.0, text="Hello world")
        assert segment.id == 0
        assert segment.text == "Hello world"

        # Test validation
        with pytest.raises(ValidationError):
            Segment()  # Missing required fields


class TestAlternative:
    """Test suite for Alternative model."""

    def test_alternative_creation_and_validation(self):
        """Test Alternative creation and validation."""
        # Test creation
        alt = Alternative(transcript="Hello world", confidence=0.9)
        assert alt.transcript == "Hello world"
        assert alt.confidence == 0.9

        # Test validation
        with pytest.raises(ValidationError):
            Alternative()  # Missing required transcript


class TestChannel:
    """Test suite for Channel model."""

    def test_channel_creation_and_validation(self):
        """Test Channel creation and validation."""
        # Test creation
        alternatives = [Alternative(transcript="Test transcript")]
        channel = Channel(alternatives=alternatives)
        assert len(channel.alternatives) == 1

        # Test validation
        with pytest.raises(ValidationError):
            Channel()  # Missing required alternatives


class TestTranscriptionResult:
    """Test suite for TranscriptionResult model."""

    def test_transcription_result_basic(self):
        """Test basic TranscriptionResult creation and validation."""
        # Test basic creation
        result = TranscriptionResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.language is None

        # Test validation
        with pytest.raises(ValidationError):
            TranscriptionResult()  # Missing required text field

    def test_transcription_result_openai_style(self):
        """Test TranscriptionResult with OpenAI-style fields."""
        segments = [Segment(id=0, seek=0, start=0.0, end=2.5, text="Hello world")]
        words = [Word(word="hello", start=0.0, end=0.5)]

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            task="transcribe",
            segments=segments,
            words=words,
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert len(result.segments) == 1
        assert len(result.words) == 1

    def test_transcription_result_deepgram_style(self):
        """Test TranscriptionResult with Deepgram-style fields."""
        alternatives = [Alternative(transcript="Hello world", confidence=0.9)]
        channels = [Channel(alternatives=alternatives)]

        result = TranscriptionResult(
            text="Hello world",
            language="en-US",
            channels=channels,
            alternatives=alternatives,
            topics=[{"topic": "greeting"}],
        )

        assert result.text == "Hello world"
        assert len(result.channels) == 1
        assert len(result.alternatives) == 1
        assert result.topics is not None
