from unittest.mock import Mock, patch
import io

import pytest

from aisuite import Client
from aisuite.framework.message import TranscriptionResult
from aisuite.provider import ASRError


@pytest.fixture(scope="module")
def provider_configs():
    return {
        "openai": {"api_key": "test_openai_api_key"},
        "aws": {
            "aws_access_key": "test_aws_access_key",
            "aws_secret_key": "test_aws_secret_key",
            "aws_session_token": "test_aws_session_token",
            "aws_region": "us-west-2",
        },
        "azure": {
            "api_key": "azure-api-key",
            "base_url": "https://model.ai.azure.com",
        },
        "groq": {
            "api_key": "groq-api-key",
        },
        "mistral": {
            "api_key": "mistral-api-key",
        },
        "google": {
            "project_id": "test_google_project_id",
            "region": "us-west4",
            "application_credentials": "test_google_application_credentials",
        },
        "fireworks": {
            "api_key": "fireworks-api-key",
        },
        "nebius": {
            "api_key": "nebius-api-key",
        },
        "inception": {
            "api_key": "inception-api-key",
        },
        "deepgram": {
            "api_key": "deepgram-api-key",
        },
    }


@pytest.mark.parametrize(
    argnames=("patch_target", "provider", "model"),
    argvalues=[
        (
            "aisuite.providers.openai_provider.OpenaiProvider.chat_completions_create",
            "openai",
            "gpt-4o",
        ),
        (
            "aisuite.providers.mistral_provider.MistralProvider.chat_completions_create",
            "mistral",
            "mistral-model",
        ),
        (
            "aisuite.providers.groq_provider.GroqProvider.chat_completions_create",
            "groq",
            "groq-model",
        ),
        (
            "aisuite.providers.aws_provider.AwsProvider.chat_completions_create",
            "aws",
            "claude-v3",
        ),
        (
            "aisuite.providers.azure_provider.AzureProvider.chat_completions_create",
            "azure",
            "azure-model",
        ),
        (
            "aisuite.providers.anthropic_provider.AnthropicProvider.chat_completions_create",
            "anthropic",
            "anthropic-model",
        ),
        (
            "aisuite.providers.google_provider.GoogleProvider.chat_completions_create",
            "google",
            "google-model",
        ),
        (
            "aisuite.providers.fireworks_provider.FireworksProvider.chat_completions_create",
            "fireworks",
            "fireworks-model",
        ),
        (
            "aisuite.providers.nebius_provider.NebiusProvider.chat_completions_create",
            "nebius",
            "nebius-model",
        ),
        (
            "aisuite.providers.inception_provider.InceptionProvider.chat_completions_create",
            "inception",
            "mercury",
        ),
    ],
)
def test_client_chat_completions(
    provider_configs: dict, patch_target: str, provider: str, model: str
):
    expected_response = f"{patch_target}_{provider}_{model}"
    with patch(patch_target) as mock_provider:
        mock_provider.return_value = expected_response
        client = Client()
        client.configure(provider_configs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]

        model_str = f"{provider}:{model}"
        model_response = client.chat.completions.create(model_str, messages=messages)
        assert model_response == expected_response


def test_invalid_provider_in_client_config():
    # Testing an invalid provider name in the configuration
    invalid_provider_configs = {
        "invalid_provider": {"api_key": "invalid_api_key"},
    }

    # With lazy loading, Client initialization should succeed
    client = Client()
    client.configure(invalid_provider_configs)

    messages = [
        {"role": "user", "content": "Hello"},
    ]

    # Expect ValueError when actually trying to use the invalid provider
    with pytest.raises(
        ValueError,
        match=r"Invalid provider key 'invalid_provider'. Supported providers: ",
    ):
        client.chat.completions.create("invalid_provider:some-model", messages=messages)


def test_invalid_model_format_in_create(monkeypatch):
    from aisuite.providers.openai_provider import OpenaiProvider

    monkeypatch.setattr(
        target=OpenaiProvider,
        name="chat_completions_create",
        value=Mock(),
    )

    # Valid provider configurations
    provider_configs = {
        "openai": {"api_key": "test_openai_api_key"},
    }

    # Initialize the client with valid provider
    client = Client()
    client.configure(provider_configs)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]

    # Invalid model format
    invalid_model = "invalidmodel"

    # Expect ValueError when calling create with invalid model format and verify message
    with pytest.raises(
        ValueError, match=r"Invalid model format. Expected 'provider:model'"
    ):
        client.chat.completions.create(invalid_model, messages=messages)


class TestClientASR:
    """Test suite for Client ASR functionality - essential tests only."""

    def test_audio_interface_initialization(self):
        """Test that Audio interface is properly initialized."""
        client = Client()
        assert hasattr(client, "audio")
        assert hasattr(client.audio, "transcriptions")

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_transcriptions_create_success(
        self, mock_create_provider, provider_configs
    ):
        """Test successful audio transcription with OpenAI."""
        mock_result = TranscriptionResult(
            text="Hello, this is a test transcription.",
            language="en",
            confidence=0.95,
            task="transcribe",
        )

        # Create a mock provider with audio support
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        client = Client()
        client.configure(provider_configs)

        audio_data = io.BytesIO(b"fake audio data")
        result = client.audio.transcriptions.create(
            model="openai:whisper-1", file=audio_data, language="en"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription."
        mock_provider.audio.transcriptions.create.assert_called_once()

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_transcriptions_create_deepgram(
        self, mock_create_provider, provider_configs
    ):
        """Test audio transcription with Deepgram provider."""
        mock_result = TranscriptionResult(
            text="Deepgram transcription result.",
            language="en",
            confidence=0.92,
            task="transcribe",
        )

        # Create a mock provider with audio support
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        client = Client()
        client.configure(provider_configs)

        result = client.audio.transcriptions.create(
            model="deepgram:nova-2", file="test_audio.wav", language="en"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Deepgram transcription result."
        mock_provider.audio.transcriptions.create.assert_called_once()

    def test_transcriptions_invalid_model_format(self, provider_configs):
        """Test that invalid model format raises ValueError."""
        client = Client()
        client.configure(provider_configs)

        with pytest.raises(ValueError, match="Invalid model format"):
            client.audio.transcriptions.create(
                model="invalid-format", file="test.wav", language="en"
            )

    def test_transcriptions_unsupported_provider(self, provider_configs):
        """Test error handling for unsupported ASR provider."""
        client = Client()
        client.configure(provider_configs)

        with pytest.raises(ValueError, match="Invalid provider key"):
            client.audio.transcriptions.create(
                model="unsupported:model", file="test.wav", language="en"
            )


class TestClientASRParameterValidation:
    """Test suite for Client-level ASR parameter validation."""

    def test_client_initialization_strict_mode(self):
        """Test Client initialization with strict extra_param_mode."""
        client = Client(extra_param_mode="strict")
        assert client.extra_param_mode == "strict"
        assert client.param_validator.extra_param_mode == "strict"

    def test_client_initialization_warn_mode(self):
        """Test Client initialization with warn extra_param_mode (default)."""
        client = Client()
        assert client.extra_param_mode == "warn"
        assert client.param_validator.extra_param_mode == "warn"

    def test_client_initialization_permissive_mode(self):
        """Test Client initialization with permissive extra_param_mode."""
        client = Client(extra_param_mode="permissive")
        assert client.extra_param_mode == "permissive"
        assert client.param_validator.extra_param_mode == "permissive"

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_strict_mode_rejects_unknown_param(self, mock_create_provider):
        """Test that strict mode raises ValueError for unknown parameters."""
        client = Client(
            provider_configs={"openai": {"api_key": "test"}}, extra_param_mode="strict"
        )

        # Mock provider shouldn't be called due to validation error
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(ValueError, match="Unknown parameters for openai"):
            client.audio.transcriptions.create(
                model="openai:whisper-1",
                file=io.BytesIO(b"audio"),
                language="en",
                invalid_param=True,  # Unknown param
            )

        # Provider should not have been called (validation failed first)
        mock_provider.audio.transcriptions.create.assert_not_called()

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_strict_mode_typo_detection(self, mock_create_provider):
        """Test that strict mode catches typos in parameter names."""
        client = Client(
            provider_configs={"openai": {"api_key": "test"}}, extra_param_mode="strict"
        )

        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        with pytest.raises(
            ValueError, match="Unknown parameters for openai: \\['langauge'\\]"
        ):
            client.audio.transcriptions.create(
                model="openai:whisper-1",
                file=io.BytesIO(b"audio"),
                langauge="en",  # TYPO: should be "language"
            )

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_warn_mode_continues_execution(self, mock_create_provider):
        """Test that warn mode continues execution after warning."""
        import warnings

        client = Client(
            provider_configs={"openai": {"api_key": "test"}}, extra_param_mode="warn"
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        # Should warn but continue
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = client.audio.transcriptions.create(
                model="openai:whisper-1",
                file=io.BytesIO(b"audio"),
                language="en",
                invalid_param=True,  # Unknown param
            )

            # Should have issued a warning
            assert len(w) == 1
            assert "Unknown parameters" in str(w[0].message)

            # But execution should continue
            assert result.text == "Test"
            mock_provider.audio.transcriptions.create.assert_called_once()

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_permissive_mode_allows_unknown_params(self, mock_create_provider):
        """Test that permissive mode allows unknown parameters."""
        import warnings

        client = Client(
            provider_configs={"openai": {"api_key": "test"}},
            extra_param_mode="permissive",
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        # Should not warn or raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = client.audio.transcriptions.create(
                model="openai:whisper-1",
                file=io.BytesIO(b"audio"),
                experimental_feature=True,  # Unknown param
            )

            # Should not have issued any warnings
            assert len(w) == 0

            # Execution should succeed
            assert result.text == "Test"
            mock_provider.audio.transcriptions.create.assert_called_once()

            # Unknown param should be passed through
            call_kwargs = mock_provider.audio.transcriptions.create.call_args.kwargs
            assert call_kwargs.get("experimental_feature") is True

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_common_param_mapping_at_client_level(self, mock_create_provider):
        """Test that common parameters are mapped correctly at Client level."""
        client = Client(
            provider_configs={"google": {"project_id": "test", "region": "us"}},
            extra_param_mode="strict",
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        # Use common param "language" which should map to "language_code" for Google
        result = client.audio.transcriptions.create(
            model="google:latest_long",
            file=io.BytesIO(b"audio"),
            language="en",  # Common param
        )

        assert result.text == "Test"
        mock_provider.audio.transcriptions.create.assert_called_once()

        # Verify parameter was mapped to language_code
        call_kwargs = mock_provider.audio.transcriptions.create.call_args.kwargs
        assert "language_code" in call_kwargs
        assert call_kwargs["language_code"] == "en-US"  # Expanded
        assert "language" not in call_kwargs  # Original key should be mapped

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_provider_specific_params_passthrough(self, mock_create_provider):
        """Test that provider-specific parameters pass through correctly."""
        client = Client(
            provider_configs={"deepgram": {"api_key": "test"}},
            extra_param_mode="strict",
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        result = client.audio.transcriptions.create(
            model="deepgram:nova-2",
            file=io.BytesIO(b"audio"),
            punctuate=True,
            diarize=True,
        )

        assert result.text == "Test"

        # Verify provider-specific params passed through
        call_kwargs = mock_provider.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["punctuate"] is True
        assert call_kwargs["diarize"] is True

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_mixed_common_and_provider_params(self, mock_create_provider):
        """Test mixing common and provider-specific parameters."""
        client = Client(
            provider_configs={"deepgram": {"api_key": "test"}},
            extra_param_mode="strict",
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        result = client.audio.transcriptions.create(
            model="deepgram:nova-2",
            file=io.BytesIO(b"audio"),
            language="en",  # Common param
            prompt="meeting",  # Common param that maps to keywords
            punctuate=True,  # Deepgram-specific
            diarize=True,  # Deepgram-specific
        )

        assert result.text == "Test"

        # Verify both common and provider params processed correctly
        call_kwargs = mock_provider.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["language"] == "en"
        assert call_kwargs["keywords"] == ["meeting"]  # prompt mapped to keywords
        assert call_kwargs["punctuate"] is True
        assert call_kwargs["diarize"] is True

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_validation_happens_before_provider_call(self, mock_create_provider):
        """Test that validation occurs before provider SDK is called."""
        client = Client(
            provider_configs={"openai": {"api_key": "test"}}, extra_param_mode="strict"
        )

        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Validation should fail before provider is even initialized
        with pytest.raises(ValueError, match="Unknown parameters"):
            client.audio.transcriptions.create(
                model="openai:whisper-1",
                file=io.BytesIO(b"audio"),
                completely_invalid_param=True,
            )

        # Provider create method should still have been called to initialize
        # but the transcription method should never be called
        mock_provider.audio.transcriptions.create.assert_not_called()

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_unsupported_common_param_ignored(self, mock_create_provider):
        """Test that unsupported common params are gracefully ignored."""
        client = Client(
            provider_configs={"deepgram": {"api_key": "test"}},
            extra_param_mode="strict",
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        # temperature is not supported by Deepgram (should be ignored)
        result = client.audio.transcriptions.create(
            model="deepgram:nova-2",
            file=io.BytesIO(b"audio"),
            language="en",
            temperature=0.5,  # Not supported by Deepgram
        )

        assert result.text == "Test"

        # Verify temperature was not passed to provider
        call_kwargs = mock_provider.audio.transcriptions.create.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert call_kwargs["language"] == "en"

    @patch("aisuite.provider.ProviderFactory.create_provider")
    def test_multiple_providers_with_same_client(self, mock_create_provider):
        """Test that the same client can handle multiple providers with different validation."""
        client = Client(
            provider_configs={
                "openai": {"api_key": "test1"},
                "deepgram": {"api_key": "test2"},
            },
            extra_param_mode="strict",
        )

        mock_result = TranscriptionResult(text="Test", language="en")
        mock_provider = Mock()
        mock_provider.audio.transcriptions.create.return_value = mock_result
        mock_create_provider.return_value = mock_provider

        # Test OpenAI with temperature (supported)
        result1 = client.audio.transcriptions.create(
            model="openai:whisper-1", file=io.BytesIO(b"audio"), temperature=0.5
        )
        assert result1.text == "Test"
        call_kwargs1 = mock_provider.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs1.get("temperature") == 0.5

        # Reset mock
        mock_provider.reset_mock()

        # Test Deepgram with temperature (not supported, should be ignored)
        result2 = client.audio.transcriptions.create(
            model="deepgram:nova-2", file=io.BytesIO(b"audio"), temperature=0.5
        )
        assert result2.text == "Test"
        call_kwargs2 = mock_provider.audio.transcriptions.create.call_args.kwargs
        assert "temperature" not in call_kwargs2
