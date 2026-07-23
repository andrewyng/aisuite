"""Tests for the async provider contract (achat_completions_create)."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aisuite.provider import Provider
from aisuite.providers.openai_provider import OpenaiProvider


class _SyncOnlyProvider(Provider):
    """A provider that only implements the sync method (no native async)."""

    def __init__(self):
        super().__init__()
        self.called_on_thread = None

    def chat_completions_create(self, model, messages, **kwargs):
        self.called_on_thread = threading.current_thread().name
        return SimpleNamespace(model=model, messages=messages, kwargs=kwargs)


@pytest.mark.asyncio
async def test_default_async_offloads_sync_to_thread():
    """The base achat_completions_create runs the sync method off the loop."""
    provider = _SyncOnlyProvider()
    result = await provider.achat_completions_create(
        "m", [{"role": "user", "content": "hi"}], temperature=0.5
    )
    assert result.model == "m"
    assert result.kwargs == {"temperature": 0.5}
    # It ran in a worker thread, not the event loop's main thread.
    assert provider.called_on_thread != threading.main_thread().name


@pytest.mark.asyncio
async def test_openai_native_async(monkeypatch):
    """OpenAI overrides with a native AsyncOpenAI-backed implementation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenaiProvider()

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="async hi"))]
    )
    provider.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    response = await provider.achat_completions_create(
        "gpt-4o", [{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == "async hi"
    provider.aclient.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_native_async_maps_max_tokens_for_newer_models(monkeypatch):
    """The native async OpenAI path applies the same token-limit abstraction."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenaiProvider()

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="async hi"))]
    )
    provider.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    response = await provider.achat_completions_create(
        "gpt-5.4-mini",
        [{"role": "user", "content": "hi"}],
        max_tokens=100,
    )

    assert response.choices[0].message.content == "async hi"
    call_kwargs = provider.aclient.chat.completions.create.await_args.kwargs
    assert call_kwargs["max_completion_tokens"] == 100
    assert "max_tokens" not in call_kwargs
