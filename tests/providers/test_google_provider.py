"""Tests for Google provider functionality (both chat and ASR)."""

import io
import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from aisuite.providers.google_provider import GoogleProvider
from aisuite.provider import ASRError
from aisuite.framework.message import (
    TranscriptionResult,
    TranscriptionOptions,
    StreamingTranscriptionChunk,
)
from vertexai.generative_models import Content, Part


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "path-to-service-account-json")
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "vertex-project-id")
    monkeypatch.setenv("GOOGLE_REGION", "us-central1")


@pytest.fixture
def mock_google_speech_response():
    """Create a mock Google Speech-to-Text API response."""
    mock_response = MagicMock()
    mock_result = MagicMock()
    mock_alternative = MagicMock()

    mock_alternative.transcript = "Hello, this is a test transcription."
    mock_alternative.confidence = 0.95

    # Mock words
    mock_word = MagicMock()
    mock_word.word = "Hello"
    mock_word.start_time.total_seconds.return_value = 0.0
    mock_word.end_time.total_seconds.return_value = 0.5
    mock_word.confidence = 0.98
    mock_alternative.words = [mock_word]

    mock_result.alternatives = [mock_alternative]
    mock_response.results = [mock_result]

    return mock_response


def test_missing_env_vars():
    """Test that an error is raised if required environment variables are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            GoogleProvider()
        assert "Missing one or more required Google environment variables" in str(
            exc_info.value
        )


def test_vertex_interface():
    """High-level test that the interface is initialized and chat completions are requested successfully."""

    # Test case 1: Regular text response
    def test_text_response():
        user_greeting = "Hello!"
        message_history = [{"role": "user", "content": user_greeting}]
        selected_model = "our-favorite-model"
        response_text_content = "mocked-text-response-from-model"

        interface = GoogleProvider()
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = response_text_content
        # Ensure function_call attribute doesn't exist
        del mock_response.candidates[0].content.parts[0].function_call

        with patch(
            "aisuite.providers.google_provider.GenerativeModel"
        ) as mock_generative_model:
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model
            mock_chat = MagicMock()
            mock_model.start_chat.return_value = mock_chat
            mock_chat.send_message.return_value = mock_response

            response = interface.chat_completions_create(
                messages=message_history,
                model=selected_model,
                temperature=0.7,
            )

            # Assert the response is in the correct format
            assert response.choices[0].message.content == response_text_content
            assert response.choices[0].finish_reason == "stop"

    # Test case 2: Function call response
    def test_function_call():
        user_greeting = "What's the weather?"
        message_history = [{"role": "user", "content": user_greeting}]
        selected_model = "our-favorite-model"

        interface = GoogleProvider()
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]

        # Mock the function call response
        function_call_mock = MagicMock()
        function_call_mock.name = "get_weather"
        function_call_mock.args = {"location": "San Francisco"}
        mock_response.candidates[0].content.parts[0].function_call = function_call_mock
        mock_response.candidates[0].content.parts[0].text = None

        with patch(
            "aisuite.providers.google_provider.GenerativeModel"
        ) as mock_generative_model:
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model
            mock_chat = MagicMock()
            mock_model.start_chat.return_value = mock_chat
            mock_chat.send_message.return_value = mock_response

            response = interface.chat_completions_create(
                messages=message_history,
                model=selected_model,
                temperature=0.7,
            )

            # Assert the response contains the function call
            assert response.choices[0].message.content is None
            assert response.choices[0].message.tool_calls[0].type == "function"
            assert (
                response.choices[0].message.tool_calls[0].function.name == "get_weather"
            )
            assert json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            ) == {"location": "San Francisco"}
            assert response.choices[0].finish_reason == "tool_calls"

    # Run both test cases
    test_text_response()
    test_function_call()


def test_convert_openai_to_vertex_ai():
    """Test the message conversion from OpenAI format to Vertex AI format."""
    interface = GoogleProvider()
    message = {"role": "user", "content": "Hello!"}

    # Use the transformer to convert the message
    result = interface.transformer.convert_request([message])

    # Verify the conversion result
    assert len(result) == 1
    assert isinstance(result[0], Content)
    assert result[0].role == "user"
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], Part)
    assert result[0].parts[0].text == "Hello!"


def test_role_conversions():
    """Test that different message roles are converted correctly."""
    interface = GoogleProvider()

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]

    result = interface.transformer.convert_request(messages)

    # System and user messages should both be converted to "user" role in Vertex AI
    assert len(result) == 3
    assert result[0].role == "user"  # system converted to user
    assert result[0].parts[0].text == "System message"

    assert result[1].role == "user"
    assert result[1].parts[0].text == "User message"

    assert result[2].role == "model"  # assistant converted to model
    assert result[2].parts[0].text == "Assistant message"


class TestGoogleProvider:
    """Test suite for Google provider functionality."""

    def test_provider_initialization(self):
        """Test that Google provider initializes correctly."""
        provider = GoogleProvider()
        assert provider is not None
        assert hasattr(provider, "audio")
        assert hasattr(provider.audio, "transcriptions")


class TestGoogleASR:
    """Test suite for Google ASR functionality."""

    def test_audio_transcriptions_create_success(self, mock_google_speech_response):
        """Test successful audio transcription."""
        google_provider = GoogleProvider()
        mock_client = MagicMock()
        mock_client.recognize.return_value = mock_google_speech_response
        google_provider._speech_client = mock_client

        with patch("builtins.open", mock_open(read_data=b"fake audio data")):
            result = google_provider.audio.transcriptions.create(
                model="google:latest_long", file="test_audio.wav"
            )

            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."
            assert result.confidence == 0.95
            assert result.task == "transcribe"

    def test_audio_transcriptions_create_with_file_object(
        self, mock_google_speech_response
    ):
        """Test audio transcription with file-like object."""
        google_provider = GoogleProvider()
        mock_client = MagicMock()
        mock_client.recognize.return_value = mock_google_speech_response
        google_provider._speech_client = mock_client

        audio_data = io.BytesIO(b"fake audio data")

        result = google_provider.audio.transcriptions.create(
            model="google:latest_long", file=audio_data
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription."

    def test_audio_transcriptions_create_with_options(
        self, mock_google_speech_response
    ):
        """Test audio transcription with TranscriptionOptions."""
        google_provider = GoogleProvider()
        mock_client = MagicMock()
        mock_client.recognize.return_value = mock_google_speech_response
        google_provider._speech_client = mock_client

        options = TranscriptionOptions(
            language="en",
            enable_automatic_punctuation=True,
            include_word_timestamps=True,
        )

        with patch("builtins.open", mock_open(read_data=b"fake audio data")):
            result = google_provider.audio.transcriptions.create(
                model="google:latest_long", file="test_audio.wav", options=options
            )

            mock_client.recognize.assert_called_once()
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello, this is a test transcription."

    def test_audio_transcriptions_create_error_handling(self):
        """Test handling of Google Speech API errors."""
        google_provider = GoogleProvider()
        mock_client = MagicMock()
        mock_client.recognize.side_effect = Exception("API Error")
        google_provider._speech_client = mock_client

        with patch("builtins.open", mock_open(read_data=b"fake audio data")):
            with pytest.raises(
                ASRError, match="Google Speech-to-Text error: API Error"
            ):
                google_provider.audio.transcriptions.create(
                    model="google:latest_long", file="test_audio.wav"
                )

    @pytest.mark.asyncio
    async def test_audio_transcriptions_create_stream_output(
        self, mock_google_speech_response
    ):
        """Test streaming audio transcription with fixed config parameter."""
        google_provider = GoogleProvider()
        mock_client = MagicMock()

        # Mock streaming response
        mock_streaming_response = MagicMock()
        mock_streaming_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "Hello streaming"
        mock_streaming_result.alternatives = [mock_alternative]
        mock_streaming_result.is_final = True
        mock_streaming_response.results = [mock_streaming_result]

        mock_client.streaming_recognize.return_value = [mock_streaming_response]
        google_provider._speech_client = mock_client

        with patch("builtins.open", mock_open(read_data=b"fake audio data")):
            result = google_provider.audio.transcriptions.create_stream_output(
                model="google:latest_long", file="test_audio.wav"
            )

            assert hasattr(result, "__aiter__")

            # Test that streaming_recognize is called with both config and requests
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
                break  # Just test one chunk

            # Verify the method was called with correct parameters
            mock_client.streaming_recognize.assert_called_once()
            call_args = mock_client.streaming_recognize.call_args
            assert "config" in call_args.kwargs
            assert "requests" in call_args.kwargs

    def test_parse_google_response_complete(self, mock_google_speech_response):
        """Test parsing complete Google response."""
        google_provider = GoogleProvider()
        result = google_provider.audio.transcriptions._parse_google_response(
            mock_google_speech_response
        )

        assert result.text == "Hello, this is a test transcription."
        assert result.confidence == 0.95
        assert result.task == "transcribe"

        assert len(result.words) == 1
        word = result.words[0]
        assert word.word == "Hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.98

        assert len(result.alternatives) == 1

    def test_parse_google_response_empty_results(self):
        """Test parsing response with empty results."""
        google_provider = GoogleProvider()
        mock_response = MagicMock()
        mock_response.results = []

        result = google_provider.audio.transcriptions._parse_google_response(
            mock_response
        )
        assert result.text == ""
        assert result.language is None
