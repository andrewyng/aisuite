"""Tests for the Parasail provider."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aisuite.provider import LLMError, ProviderFactory
from aisuite.providers.parasail_provider import DEFAULT_BASE_URL, ParasailProvider


@pytest.fixture(autouse=True)
def parasail_environment(monkeypatch):
    monkeypatch.setenv("PARASAIL_API_KEY", "test-parasail-api-key")


@pytest.fixture
def mock_clients():
    with (
        patch("aisuite.providers.parasail_provider.openai.OpenAI") as sync_client,
        patch("aisuite.providers.parasail_provider.openai.AsyncOpenAI") as async_client,
    ):
        yield sync_client, async_client


@pytest.fixture
def provider(mock_clients):
    return ParasailProvider()


def test_default_configuration_uses_parasail_gateway(provider, mock_clients):
    sync_client, async_client = mock_clients

    expected_config = {
        "api_key": "test-parasail-api-key",
        "base_url": DEFAULT_BASE_URL,
    }
    sync_client.assert_called_once_with(**expected_config)
    async_client.assert_called_once_with(**expected_config)
    assert provider.audio is None


def test_explicit_configuration_overrides_defaults(mock_clients):
    sync_client, async_client = mock_clients

    ParasailProvider(
        api_key="configured-key",
        base_url="https://parasail.example/v1",
        timeout=30,
    )

    expected_config = {
        "api_key": "configured-key",
        "base_url": "https://parasail.example/v1",
        "timeout": 30,
    }
    sync_client.assert_called_once_with(**expected_config)
    async_client.assert_called_once_with(**expected_config)


def test_missing_api_key_fails_before_creating_clients(monkeypatch, mock_clients):
    monkeypatch.delenv("PARASAIL_API_KEY")

    with pytest.raises(ValueError, match="Parasail API key is missing"):
        ParasailProvider()

    sync_client, async_client = mock_clients
    sync_client.assert_not_called()
    async_client.assert_not_called()


def test_provider_factory_discovers_parasail(mock_clients):
    assert "parasail" in ProviderFactory.get_supported_providers()

    provider = ProviderFactory.create_provider(
        "parasail", {"api_key": "configured-key"}
    )

    assert isinstance(provider, ParasailProvider)


def test_chat_completion_converts_and_forwards_request(provider):
    original_messages = [{"role": "user", "content": "Hello"}]
    converted_messages = [{"role": "user", "content": "Converted"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    response = MagicMock()
    provider.transformer.convert_request = MagicMock(return_value=converted_messages)
    provider.client.chat.completions.create.return_value = response

    actual = provider.chat_completions_create(
        "parasail-deepseek-r1",
        original_messages,
        temperature=0.2,
        tools=tools,
    )

    assert actual is response
    provider.transformer.convert_request.assert_called_once_with(original_messages)
    provider.client.chat.completions.create.assert_called_once_with(
        model="parasail-deepseek-r1",
        messages=converted_messages,
        temperature=0.2,
        tools=tools,
    )


def test_chat_completion_normalizes_errors(provider):
    provider.client.chat.completions.create.side_effect = RuntimeError(
        "request rejected"
    )

    with pytest.raises(LLMError, match="Parasail chat completion failed") as exc_info:
        provider.chat_completions_create(
            "parasail-deepseek-r1",
            [{"role": "user", "content": "Hello"}],
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_async_chat_completion_uses_async_client(provider):
    response = MagicMock()
    provider.aclient.chat.completions.create = AsyncMock(return_value=response)

    actual = asyncio.run(
        provider.achat_completions_create(
            "parasail-deepseek-r1",
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
        )
    )

    assert actual is response
    provider.aclient.chat.completions.create.assert_awaited_once_with(
        model="parasail-deepseek-r1",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
    )


def test_sync_stream_yields_chunks_and_enables_streaming(provider):
    chunks = [MagicMock(), MagicMock()]
    provider.client.chat.completions.create.return_value = iter(chunks)

    actual = list(
        provider.chat_completions_create_stream(
            "parasail-deepseek-r1",
            [{"role": "user", "content": "Hello"}],
        )
    )

    assert actual == chunks
    provider.client.chat.completions.create.assert_called_once_with(
        model="parasail-deepseek-r1",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )


def test_async_stream_yields_chunks_and_enables_streaming(provider):
    chunks = [MagicMock(), MagicMock()]

    async def chunk_stream():
        for chunk in chunks:
            yield chunk

    provider.aclient.chat.completions.create = AsyncMock(return_value=chunk_stream())

    async def collect_chunks():
        return [
            chunk
            async for chunk in provider.achat_completions_create_stream(
                "parasail-deepseek-r1",
                [{"role": "user", "content": "Hello"}],
            )
        ]

    actual = asyncio.run(collect_chunks())

    assert actual == chunks
    provider.aclient.chat.completions.create.assert_awaited_once_with(
        model="parasail-deepseek-r1",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )


def test_stream_iteration_errors_are_normalized(provider):
    def failing_stream():
        yield MagicMock()
        raise RuntimeError("stream interrupted")

    provider.client.chat.completions.create.return_value = failing_stream()

    with pytest.raises(
        LLMError, match="Parasail chat completion stream failed"
    ) as exc_info:
        list(
            provider.chat_completions_create_stream(
                "parasail-deepseek-r1",
                [{"role": "user", "content": "Hello"}],
            )
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
