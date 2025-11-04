"""Component tests for ASR parameter pass-through to provider SDKs."""

import io
from unittest.mock import MagicMock, mock_open, patch
import pytest

from aisuite.providers.openai_provider import OpenaiProvider
from aisuite.providers.deepgram_provider import DeepgramProvider
from aisuite.providers.google_provider import GoogleProvider
from aisuite.framework.message import TranscriptionResult


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Fixture to set environment variables for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram-key")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "test-creds.json")
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "test-project")
    monkeypatch.setenv("GOOGLE_REGION", "us-central1")


class TestOpenAIParameterPassthrough:
    """Test that parameters correctly reach OpenAI SDK."""

    def test_language_param_passthrough(self):
        """Test language parameter reaches OpenAI SDK."""
        provider = OpenaiProvider()
        mock_response = MagicMock()
        mock_response.text = "Test transcription"
        mock_response.language = "en"
        mock_response.segments = None

        with patch("builtins.open", mock_open(read_data=b"audio")), patch.object(
            provider.client.audio.transcriptions, "create", return_value=mock_response
        ) as mock_create:

            provider.audio.transcriptions.create(
                model="whisper-1", file="test.mp3", language="en"
            )

            # Verify language was passed to SDK
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert "language" in call_kwargs
            assert call_kwargs["language"] == "en"

    def test_temperature_param_passthrough(self):
        """Test temperature parameter reaches OpenAI SDK."""
        provider = OpenaiProvider()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.language = "en"
        mock_response.segments = None

        with patch("builtins.open", mock_open(read_data=b"audio")), patch.object(
            provider.client.audio.transcriptions, "create", return_value=mock_response
        ) as mock_create:

            provider.audio.transcriptions.create(
                model="whisper-1", file="test.mp3", temperature=0.7
            )

            call_kwargs = mock_create.call_args.kwargs
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.7

    def test_response_format_param_passthrough(self):
        """Test response_format parameter reaches OpenAI SDK."""
        provider = OpenaiProvider()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.language = "en"
        mock_response.segments = None

        with patch("builtins.open", mock_open(read_data=b"audio")), patch.object(
            provider.client.audio.transcriptions, "create", return_value=mock_response
        ) as mock_create:

            provider.audio.transcriptions.create(
                model="whisper-1", file="test.mp3", response_format="verbose_json"
            )

            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" in call_kwargs
            assert call_kwargs["response_format"] == "verbose_json"

    def test_multiple_params_passthrough(self):
        """Test multiple parameters reach OpenAI SDK together."""
        provider = OpenaiProvider()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.language = "en"
        mock_response.segments = None

        with patch("builtins.open", mock_open(read_data=b"audio")), patch.object(
            provider.client.audio.transcriptions, "create", return_value=mock_response
        ) as mock_create:

            provider.audio.transcriptions.create(
                model="whisper-1",
                file="test.mp3",
                language="en",
                temperature=0.5,
                response_format="json",
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["language"] == "en"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["response_format"] == "json"

    def test_file_object_with_params(self):
        """Test that file-like object works with parameters."""
        provider = OpenaiProvider()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_response.language = "en"
        mock_response.segments = None

        audio_data = io.BytesIO(b"fake audio data")

        with patch.object(
            provider.client.audio.transcriptions, "create", return_value=mock_response
        ) as mock_create:

            provider.audio.transcriptions.create(
                model="whisper-1", file=audio_data, language="en"
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["file"] == audio_data
            assert call_kwargs["language"] == "en"


class TestGoogleParameterPassthrough:
    """Test that parameters correctly reach Google Speech SDK."""

    @patch("aisuite.providers.google_provider.vertexai.init")
    def test_language_code_param_passthrough(self, mock_vertexai_init):
        """Test language_code parameter reaches Google SDK."""
        provider = GoogleProvider()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "Test"
        mock_alternative.confidence = 0.95
        mock_alternative.words = []
        mock_result.alternatives = [mock_alternative]
        mock_response.results = [mock_result]

        provider._speech_client = MagicMock()
        provider._speech_client.recognize.return_value = mock_response

        with patch("builtins.open", mock_open(read_data=b"audio")):
            provider.audio.transcriptions.create(
                model="latest_long", file="test.wav", language_code="en-US"
            )

            # Verify language_code was in the config passed to SDK
            provider._speech_client.recognize.assert_called_once()
            call_kwargs = provider._speech_client.recognize.call_args.kwargs
            assert "config" in call_kwargs
            config = call_kwargs["config"]
            assert config.language_code == "en-US"

    @patch("aisuite.providers.google_provider.vertexai.init")
    def test_enable_automatic_punctuation_passthrough(self, mock_vertexai_init):
        """Test enable_automatic_punctuation parameter reaches Google SDK."""
        provider = GoogleProvider()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "Test."
        mock_alternative.confidence = 0.95
        mock_alternative.words = []
        mock_result.alternatives = [mock_alternative]
        mock_response.results = [mock_result]

        provider._speech_client = MagicMock()
        provider._speech_client.recognize.return_value = mock_response

        with patch("builtins.open", mock_open(read_data=b"audio")):
            provider.audio.transcriptions.create(
                model="latest_long",
                file="test.wav",
                language_code="en-US",
                enable_automatic_punctuation=True,
            )

            call_kwargs = provider._speech_client.recognize.call_args.kwargs
            config = call_kwargs["config"]
            assert config.enable_automatic_punctuation is True

    @patch("aisuite.providers.google_provider.vertexai.init")
    def test_speech_contexts_passthrough(self, mock_vertexai_init):
        """Test speech_contexts parameter (from prompt mapping) reaches Google SDK."""
        provider = GoogleProvider()
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_alternative = MagicMock()
        mock_alternative.transcript = "Technical terms"
        mock_alternative.confidence = 0.95
        mock_alternative.words = []
        mock_result.alternatives = [mock_alternative]
        mock_response.results = [mock_result]

        provider._speech_client = MagicMock()
        provider._speech_client.recognize.return_value = mock_response

        with patch("builtins.open", mock_open(read_data=b"audio")):
            # Note: This would come in as speech_contexts after validation layer transforms prompt
            provider.audio.transcriptions.create(
                model="latest_long",
                file="test.wav",
                language_code="en-US",
                speech_contexts=[{"phrases": ["technical terms"]}],
            )

            call_kwargs = provider._speech_client.recognize.call_args.kwargs
            config = call_kwargs["config"]
            assert len(config.speech_contexts) == 1
            assert config.speech_contexts[0].phrases == ["technical terms"]
