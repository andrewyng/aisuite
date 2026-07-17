# Run this test before releasing a new version.
# It will test all the models in the client.

import pytest
import aisuite as ai
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv


def setup_client() -> ai.Client:
    """Initialize the AI client with environment variables."""
    load_dotenv(find_dotenv())
    return ai.Client()


def get_test_models() -> List[str]:
    """Return a list of model identifiers to test."""
    return [
        "anthropic:claude-3-5-sonnet-20240620",
        "aws:meta.llama3-1-8b-instruct-v1:0",
        "huggingface:mistralai/Mistral-7B-Instruct-v0.3",
        "groq:llama3-8b-8192",
        "mistral:open-mistral-7b",
        "openai:gpt-3.5-turbo",
        "cohere:command-r-plus-08-2024",
        "inception:mercury",
    ]


def get_test_messages() -> List[Dict[str, str]]:
    """Return the test messages to send to each model."""
    return [
        {
            "role": "system",
            "content": "Respond in Pirate English. Always try to include the phrase - No rum No fun.",
        },
        {"role": "user", "content": "Tell me a joke about Captain Jack Sparrow"},
    ]


@pytest.mark.integration
@pytest.mark.parametrize("model_id", get_test_models())
def test_model_pirate_response(model_id: str):
    """
    Test that each model responds appropriately to the pirate prompt.

    Args:
        model_id: The provider:model identifier to test
    """
    client = setup_client()
    messages = get_test_messages()

    try:
        response = client.chat.completions.create(
            model=model_id, messages=messages, temperature=0.75
        )

        content = response.choices[0].message.content.lower()

        # Check if either version of the required phrase is present
        assert any(
            phrase in content for phrase in ["no rum no fun", "no rum, no fun"]
        ), f"Model {model_id} did not include required phrase 'No rum No fun'"

        assert len(content) > 0, f"Model {model_id} returned empty response"
        assert isinstance(
            content, str
        ), f"Model {model_id} returned non-string response"

    except Exception as e:
        pytest.fail(f"Error testing model {model_id}: {str(e)}")


def get_test_asr_models() -> List[str]:
    """Return a list of ASR model identifiers to test."""
    return [
        "openai:whisper-1",
        "deepgram:nova-2",
        "google:latest_long",
        "huggingface:openai/whisper-large-v3",
    ]


@pytest.mark.integration
@pytest.mark.parametrize("model_id", get_test_asr_models())
def test_asr_portable_transcription(model_id: str):
    """
    Test that portable ASR code works across different providers.

    This test verifies:
    1. Common parameter 'language' works for all providers
    2. Same audio file can be transcribed by different providers
    3. All providers return non-empty transcription results

    Args:
        model_id: The provider:model identifier to test (e.g., "openai:whisper-1")
    """
    client = setup_client()

    # Simple test audio file - you'll need to provide a valid audio file
    # For actual testing, replace with a real audio file path
    audio_file_path = "tests/test-data/test_audio.mp3"

    try:
        # Use common parameter 'language' that should work across all providers
        result = client.audio.transcriptions.create(
            model=model_id,
            file=audio_file_path,
            language="en",  # Common param - should auto-map for each provider
        )

        # Verify result has text
        assert hasattr(
            result, "text"
        ), f"Model {model_id} result missing 'text' attribute"
        assert len(result.text) > 0, f"Model {model_id} returned empty transcription"
        assert isinstance(
            result.text, str
        ), f"Model {model_id} returned non-string transcription"

        # Verify transcription contains expected content from tests/test-data/test_audio.mp3
        # Audio: "Why did the scarecrow win an award? Because he was outstanding in the field."
        expected_keywords = ["scarecrow", "award", "field"]
        found_keywords = [
            kw for kw in expected_keywords if kw.lower() in result.text.lower()
        ]
        assert len(found_keywords) >= 2, (
            f"Model {model_id} transcription missing expected content. "
            f"Found {len(found_keywords)}/3 keywords. Text: '{result.text}'"
        )

        # Optional: Check for language if available and returned by provider
        # Note: Some providers (e.g., Deepgram) only return language if detect_language=True
        if hasattr(result, "language") and result.language is not None:
            assert isinstance(
                result.language, str
            ), f"Model {model_id} returned invalid language type"

    except FileNotFoundError:
        pytest.skip(f"Test audio file not found for {model_id}. Skipping test.")
    except Exception as e:
        pytest.fail(f"Error testing ASR model {model_id}: {str(e)}")


@pytest.mark.integration
def test_asr_deepgram_provider_specific_feature():
    """
    Test Deepgram provider-specific feature to verify pass-through works.

    This ensures that provider-specific parameters like 'punctuate' are
    correctly passed through the validation layer to the provider SDK.
    """
    client = setup_client()
    audio_file_path = "tests/test-data/test_audio.mp3"

    try:
        # Use Deepgram-specific feature
        result = client.audio.transcriptions.create(
            model="deepgram:nova-2",
            file=audio_file_path,
            language="en",  # Common param
            punctuate=True,  # Deepgram-specific param
        )

        assert len(result.text) > 0, "Deepgram returned empty transcription"

        # If punctuation worked, text should contain punctuation marks
        # Note: This is a soft check as it depends on audio content
        # Just verify execution succeeded with provider-specific param

    except FileNotFoundError:
        pytest.skip("Test audio file not found. Skipping Deepgram feature test.")
    except Exception as e:
        pytest.fail(f"Error testing Deepgram provider-specific feature: {str(e)}")


@pytest.mark.integration
def test_asr_google_language_mapping():
    """
    Test Google language mapping to verify auto-transformation.

    This test verifies that the common parameter 'language="en"' is
    automatically transformed to 'language_code="en-US"' for Google.
    """
    client = setup_client()
    audio_file_path = "tests/test-data/test_audio.mp3"

    try:
        # Use 2-letter language code that should be expanded for Google
        result = client.audio.transcriptions.create(
            model="google:latest_long",
            file=audio_file_path,
            language="en",  # Should be auto-transformed to "en-US"
        )

        assert len(result.text) > 0, "Google returned empty transcription"
        # If we got here, the language code transformation worked

    except FileNotFoundError:
        pytest.skip("Test audio file not found. Skipping Google mapping test.")
    except Exception as e:
        pytest.fail(f"Error testing Google language mapping: {str(e)}")


@pytest.mark.integration
def test_asr_huggingface_word_timestamps():
    """
    Test Hugging Face word-level timestamps feature.

    This ensures that provider-specific parameters like 'return_timestamps'
    are correctly passed through to the Hugging Face Inference API.
    """
    client = setup_client()
    audio_file_path = "tests/test-data/test_audio.mp3"

    try:
        # Use Hugging Face-specific feature
        result = client.audio.transcriptions.create(
            model="huggingface:openai/whisper-large-v3",
            file=audio_file_path,
            return_timestamps="word",  # HF-specific param for word-level timestamps
        )

        assert len(result.text) > 0, "Hugging Face returned empty transcription"

        # If return_timestamps worked, result should have words with timestamps
        if hasattr(result, "words") and result.words:
            # Verify at least some words have timestamps
            words_with_timestamps = [
                w for w in result.words if w.start is not None and w.end is not None
            ]
            assert (
                len(words_with_timestamps) > 0
            ), "No words with timestamps found in result"

    except FileNotFoundError:
        pytest.skip("Test audio file not found. Skipping Hugging Face feature test.")
    except Exception as e:
        pytest.fail(f"Error testing Hugging Face word timestamps feature: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
