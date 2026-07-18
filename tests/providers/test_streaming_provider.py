"""Tests for the streaming provider contract (chat_completions_create_stream)."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aisuite.provider import Provider, LLMError
from aisuite.providers.openai_provider import OpenaiProvider


class _NonStreamingProvider(Provider):
    """A provider that never opted into streaming."""

    def chat_completions_create(self, model, messages, **kwargs):  # pragma: no cover
        return SimpleNamespace()


class _SyncStreamProvider(Provider):
    """A provider with only a synchronous stream (no native async)."""

    def __init__(self, chunks=(), fail_after=None):
        super().__init__()
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.streamed_on_thread = None

    def chat_completions_create(self, model, messages, **kwargs):  # pragma: no cover
        return SimpleNamespace()

    def chat_completions_create_stream(self, model, messages, **kwargs):
        self.streamed_on_thread = threading.current_thread().name
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise RuntimeError("stream broke")
            yield chunk


def test_base_sync_stream_reports_unsupported():
    provider = _NonStreamingProvider()
    with pytest.raises(LLMError, match="does not support streaming"):
        provider.chat_completions_create_stream("m", [])


@pytest.mark.asyncio
async def test_default_async_stream_bridges_sync_off_the_loop():
    """The base achat_completions_create_stream drains the provider's sync
    stream in a worker thread and yields the chunks in order."""
    provider = _SyncStreamProvider(chunks=["a", "b", "c"])

    received = [
        chunk
        async for chunk in provider.achat_completions_create_stream(
            "m", [{"role": "user", "content": "hi"}]
        )
    ]

    assert received == ["a", "b", "c"]
    assert provider.streamed_on_thread != threading.main_thread().name


@pytest.mark.asyncio
async def test_default_async_stream_propagates_errors():
    provider = _SyncStreamProvider(chunks=["a", "b"], fail_after=1)

    received = []
    with pytest.raises(RuntimeError, match="stream broke"):
        async for chunk in provider.achat_completions_create_stream("m", []):
            received.append(chunk)

    assert received == ["a"]


@pytest.mark.asyncio
async def test_default_async_stream_surfaces_unsupported():
    provider = _NonStreamingProvider()
    with pytest.raises(LLMError, match="does not support streaming"):
        async for _ in provider.achat_completions_create_stream("m", []):
            pass  # pragma: no cover


# -- OpenAI: chunks pass through, stream=True reaches the SDK ---------------------------
@pytest.fixture
def openai_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    return OpenaiProvider()


def test_openai_sync_stream_passes_through(openai_provider):
    chunks = [SimpleNamespace(id="1"), SimpleNamespace(id="2")]
    openai_provider.client.chat.completions.create = Mock(return_value=iter(chunks))

    received = list(
        openai_provider.chat_completions_create_stream(
            "gpt-4o", [{"role": "user", "content": "hi"}], temperature=0.1
        )
    )

    assert received == chunks
    call = openai_provider.client.chat.completions.create.call_args
    assert call.kwargs["stream"] is True
    assert call.kwargs["temperature"] == 0.1
    assert call.kwargs["model"] == "gpt-4o"


def test_openai_sync_stream_wraps_errors(openai_provider):
    openai_provider.client.chat.completions.create = Mock(
        side_effect=RuntimeError("boom")
    )
    with pytest.raises(LLMError, match="boom"):
        list(
            openai_provider.chat_completions_create_stream(
                "gpt-4o", [{"role": "user", "content": "hi"}]
            )
        )


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


@pytest.mark.asyncio
async def test_openai_async_stream_passes_through(openai_provider):
    chunks = [SimpleNamespace(id="1"), SimpleNamespace(id="2")]
    openai_provider.aclient.chat.completions.create = AsyncMock(
        return_value=_AsyncIter(chunks)
    )

    received = [
        chunk
        async for chunk in openai_provider.achat_completions_create_stream(
            "gpt-4o", [{"role": "user", "content": "hi"}]
        )
    ]

    assert received == chunks
    call = openai_provider.aclient.chat.completions.create.await_args
    assert call.kwargs["stream"] is True
