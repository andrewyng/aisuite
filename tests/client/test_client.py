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

    @patch(
        "aisuite.providers.openai_provider.OpenaiProvider.audio_transcriptions_create"
    )
    def test_transcriptions_create_success(self, mock_transcribe, provider_configs):
        """Test successful audio transcription with OpenAI."""
        # Mock successful transcription
        mock_result = TranscriptionResult(
            text="Hello, this is a test transcription.",
            language="en",
            confidence=0.95,
            task="transcribe",
        )
        mock_transcribe.return_value = mock_result

        client = Client()
        client.configure(provider_configs)

        audio_data = io.BytesIO(b"fake audio data")
        result = client.audio.transcriptions.create(
            model="openai:whisper-1", file=audio_data
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription."
        mock_transcribe.assert_called_once()

    @patch(
        "aisuite.providers.deepgram_provider.DeepgramProvider.audio_transcriptions_create"
    )
    def test_transcriptions_create_deepgram(self, mock_transcribe, provider_configs):
        """Test audio transcription with Deepgram provider."""
        mock_result = TranscriptionResult(
            text="Deepgram transcription result.",
            language="en",
            confidence=0.92,
            task="transcribe",
        )
        mock_transcribe.return_value = mock_result

        client = Client()
        client.configure(provider_configs)

        result = client.audio.transcriptions.create(
            model="deepgram:nova-2", file="test_audio.wav"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Deepgram transcription result."
        mock_transcribe.assert_called_once()

    def test_transcriptions_invalid_model_format(self, provider_configs):
        """Test that invalid model format raises ValueError."""
        client = Client()
        client.configure(provider_configs)

        with pytest.raises(ValueError, match="Invalid model format"):
            client.audio.transcriptions.create(model="invalid-format", file="test.wav")

    def test_transcriptions_unsupported_provider(self, provider_configs):
        """Test error handling for unsupported ASR provider."""
        client = Client()
        client.configure(provider_configs)

        with pytest.raises(ValueError, match="Invalid provider key 'unsupported'"):
            client.audio.transcriptions.create(
                model="unsupported:model", file="test.wav"
            )
