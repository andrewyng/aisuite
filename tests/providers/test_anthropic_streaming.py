"""Tests for Anthropic streaming: event normalization into OpenAI-shaped chunks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aisuite.providers.anthropic_provider import (
    AnthropicMessageConverter,
    AnthropicProvider,
)


def _ev(**kwargs):
    return SimpleNamespace(**kwargs)


def _events():
    """A representative Anthropic stream: text, then one tool call.

    Note the tool_use block has Anthropic content-block index 1 (the text block
    is index 0) but must surface as OpenAI tool-call index 0.
    """
    return [
        _ev(
            type="message_start",
            message=_ev(usage=_ev(input_tokens=10, cache_read_input_tokens=3)),
        ),
        _ev(type="content_block_start", index=0, content_block=_ev(type="text")),
        _ev(
            type="content_block_delta",
            index=0,
            delta=_ev(type="text_delta", text="Hel"),
        ),
        _ev(
            type="content_block_delta", index=0, delta=_ev(type="text_delta", text="lo")
        ),
        _ev(type="content_block_stop", index=0),
        _ev(
            type="content_block_start",
            index=1,
            content_block=_ev(type="tool_use", id="toolu_1", name="get_weather"),
        ),
        _ev(
            type="content_block_delta",
            index=1,
            delta=_ev(type="input_json_delta", partial_json='{"location"'),
        ),
        _ev(
            type="content_block_delta",
            index=1,
            delta=_ev(type="input_json_delta", partial_json=': "SF"}'),
        ),
        _ev(type="content_block_stop", index=1),
        _ev(
            type="message_delta",
            delta=_ev(stop_reason="tool_use"),
            usage=_ev(output_tokens=7),
        ),
        _ev(type="message_stop"),
    ]


def _convert_all(events):
    converter = AnthropicMessageConverter()
    state = {}
    chunks = [converter.convert_stream_event(event, state) for event in events]
    return [chunk for chunk in chunks if chunk is not None]


def test_stream_events_normalize_to_openai_chunks():
    chunks = _convert_all(_events())

    # message_start → role chunk
    assert chunks[0].choices[0].delta.role == "assistant"

    # text deltas surface incrementally
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "Hello"

    # tool call opens with id/name at OpenAI index 0 (not Anthropic block index 1)
    tool_start = chunks[3].choices[0].delta.tool_calls[0]
    assert tool_start.index == 0
    assert tool_start.id == "toolu_1"
    assert tool_start.type == "function"
    assert tool_start.function.name == "get_weather"

    # argument fragments concatenate to the full JSON
    arguments = "".join(
        fragment.function.arguments
        for chunk in chunks
        if chunk.choices[0].delta.tool_calls
        for fragment in chunk.choices[0].delta.tool_calls
    )
    assert arguments == '{"location": "SF"}'

    # final chunk: normalized finish reason + usage
    final = chunks[-1]
    assert final.choices[0].finish_reason == "tool_calls"
    assert final.usage.prompt_tokens == 10
    assert final.usage.completion_tokens == 7
    assert final.usage.total_tokens == 17
    assert final.usage.prompt_tokens_details.cached_tokens == 3


def test_stop_reason_maps_like_non_streaming():
    converter = AnthropicMessageConverter()
    chunk = converter.convert_stream_event(
        _ev(type="message_delta", delta=_ev(stop_reason="end_turn"), usage=None), {}
    )
    assert chunk.choices[0].finish_reason == "stop"


def test_uninteresting_events_yield_nothing():
    converter = AnthropicMessageConverter()
    state = {}
    for event in (
        _ev(type="content_block_start", index=0, content_block=_ev(type="text")),
        _ev(type="content_block_stop", index=0),
        _ev(type="message_stop"),
        _ev(type="ping"),
        _ev(
            type="content_block_delta",
            index=0,
            delta=_ev(type="thinking_delta", thinking="hmm"),
        ),
    ):
        assert converter.convert_stream_event(event, state) is None


# -- provider wiring ---------------------------------------------------------------------
@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-api-key")


def test_sync_stream_converts_request_and_streams(provider):
    provider.client.messages.create = Mock(return_value=iter(_events()))

    chunks = list(
        provider.chat_completions_create_stream(
            "model-x",
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
        )
    )

    call = provider.client.messages.create.call_args
    assert call.kwargs["stream"] is True
    assert call.kwargs["system"] == "Be brief."
    assert call.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call.kwargs["max_tokens"] == 4096  # default applied, as non-streaming
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


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
async def test_async_stream_uses_native_async_client(provider):
    provider.async_client.messages.create = AsyncMock(
        return_value=_AsyncIter(_events())
    )

    chunks = [
        chunk
        async for chunk in provider.achat_completions_create_stream(
            "model-x", [{"role": "user", "content": "hi"}]
        )
    ]

    assert provider.async_client.messages.create.await_args.kwargs["stream"] is True
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello"
    assert chunks[-1].usage.total_tokens == 17
