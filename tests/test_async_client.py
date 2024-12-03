"""
Tests for the async client functionality.
"""
import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from aisuite import AsyncClient
from aisuite.framework.chat_completion_response import ChatCompletionResponse
from aisuite.providers.async_openai_provider import AsyncOpenAIProvider

@pytest.fixture
def mock_response():
    response = ChatCompletionResponse()
    response.choices[0].message.content = "Test response"
    return response

@pytest.fixture
def mock_provider():
    provider = AsyncMock(spec=AsyncOpenAIProvider)
    provider.async_chat_completions_create.return_value = ChatCompletionResponse()
    return provider

@pytest.fixture
def mock_provider_factory():
    with patch("aisuite.provider.ProviderFactory") as factory:
        yield factory

@pytest.mark.asyncio
async def test_async_chat_completion(mock_provider, mock_provider_factory, mock_response):
    """Test async chat completion."""
    # Setup
    mock_provider.async_chat_completions_create.return_value = mock_response
    mock_provider_factory.create_provider.return_value = mock_provider
    mock_provider_factory.get_supported_providers.return_value = ["openai"]

    client = AsyncClient({"openai": {"api_key": "test_key"}})
    messages = [{"role": "user", "content": "Hello"}]

    # Test
    response = await client.chat.completions.create(
        model="openai:gpt-4",
        messages=messages
    )

    # Assert
    assert response == mock_response
    mock_provider.async_chat_completions_create.assert_called_once_with(
        "gpt-4",
        messages
    )

@pytest.mark.asyncio
async def test_async_chat_completion_stream(mock_provider, mock_provider_factory, mock_response):
    """Test async chat completion streaming."""
    # Setup
    async def mock_stream():
        yield mock_response
        yield mock_response

    mock_provider.async_chat_completions_create_stream.return_value = mock_stream()
    mock_provider_factory.create_provider.return_value = mock_provider
    mock_provider_factory.get_supported_providers.return_value = ["openai"]

    client = AsyncClient({"openai": {"api_key": "test_key"}})
    messages = [{"role": "user", "content": "Hello"}]

    # Test
    responses = []
    async for response in client.chat.completions.create(
        model="openai:gpt-4",
        messages=messages,
        stream=True
    ):
        responses.append(response)

    # Assert
    assert len(responses) == 2
    assert all(r == mock_response for r in responses)
    mock_provider.async_chat_completions_create_stream.assert_called_once_with(
        "gpt-4",
        messages
    )

@pytest.mark.asyncio
async def test_async_chat_completion_invalid_model():
    """Test async chat completion with invalid model format."""
    client = AsyncClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Invalid model format"):
        await client.chat.completions.create(
            model="invalid-model",
            messages=messages
        )

@pytest.mark.asyncio
async def test_async_chat_completion_unsupported_provider(mock_provider_factory):
    """Test async chat completion with unsupported provider."""
    mock_provider_factory.get_supported_providers.return_value = ["openai"]
    
    client = AsyncClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Invalid provider key"):
        await client.chat.completions.create(
            model="unsupported:model",
            messages=messages
        )

@pytest.mark.asyncio
async def test_async_client_context_manager():
    """Test async client context manager."""
    async with AsyncClient() as client:
        assert isinstance(client, AsyncClient)

@pytest.mark.asyncio
async def test_concurrent_requests(mock_provider, mock_provider_factory, mock_response):
    """Test concurrent requests to multiple providers."""
    # Setup
    mock_provider.async_chat_completions_create.return_value = mock_response
    mock_provider_factory.create_provider.return_value = mock_provider
    mock_provider_factory.get_supported_providers.return_value = ["openai", "anthropic"]

    client = AsyncClient({
        "openai": {"api_key": "test_key"},
        "anthropic": {"api_key": "test_key"}
    })
    messages = [{"role": "user", "content": "Hello"}]
    models = ["openai:gpt-4", "anthropic:claude-3"]

    # Test
    tasks = [
        client.chat.completions.create(model=model, messages=messages)
        for model in models
    ]
    responses = await asyncio.gather(*tasks)

    # Assert
    assert len(responses) == 2
    assert all(r == mock_response for r in responses)
