from unittest.mock import patch

import pytest

from aisuite import AsyncClient


@pytest.fixture(scope="module")
def provider_configs():
    return {
        "openai": {"api_key": "test_openai_api_key"},
    }


@pytest.mark.asyncio
async def test_async_client_chat_completions(provider_configs):
    expected_response = "async-response"
    with patch(
        "aisuite.providers.openai_provider.OpenaiProvider.chat_completions_create"
    ) as mock_provider:
        mock_provider.return_value = expected_response
        client = AsyncClient()
        client.configure(provider_configs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]

        model_response = await client.chat.completions.create(
            "openai:gpt-4o", messages=messages
        )
        assert model_response == expected_response


@pytest.mark.asyncio
async def test_async_client_invalid_model_format(provider_configs):
    client = AsyncClient()
    client.configure(provider_configs)

    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(
        ValueError, match=r"Invalid model format. Expected 'provider:model'"
    ):
        await client.chat.completions.create("invalidmodel", messages=messages)
