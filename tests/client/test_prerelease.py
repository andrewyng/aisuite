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


def setup_async_client() -> ai.AsyncClient:
    """Initialize the async AI client with environment variables."""
    load_dotenv(find_dotenv())
    return ai.AsyncClient()


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
    ]


def get_test_async_models() -> List[str]:
    """Return a list of model identifiers to test."""
    return [
        "anthropic:claude-3-5-sonnet-20240620",
        "mistral:open-mistral-7b",
        "openai:gpt-3.5-turbo"
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", get_test_models())
async def test_async_model_pirate_response(model_id: str):
    """
    Test that each model responds appropriately to the pirate prompt using async client.

    Args:
        model_id: The provider:model identifier to test
    """
    client = setup_async_client()
    messages = get_test_messages()

    try:
        response = await client.chat.completions.create(
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
