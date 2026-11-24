"""Prompt-token totals include uncached input, cache reads, and cache writes."""

import pytest
from anthropic.types import Message, RawMessageDeltaEvent, RawMessageStartEvent

from aisuite.providers.anthropic_provider import AnthropicMessageConverter


@pytest.mark.parametrize(
    "cache_usage",
    [
        {"cache_read_input_tokens": 17},
        {"cache_creation_input_tokens": 23},
        {"cache_read_input_tokens": 17, "cache_creation_input_tokens": 23},
        {},
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_anthropic_usage_counts_all_input_tokens(cache_usage, streaming):
    message = Message(
        id="msg_cached",
        model="claude-sonnet-4-5",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "done"}],
        stop_reason="end_turn",
        usage={"input_tokens": 5, "output_tokens": 7, **cache_usage},
    )
    converter = AnthropicMessageConverter()
    if streaming:
        state = {}
        converter.convert_stream_event(
            RawMessageStartEvent(type="message_start", message=message), state
        )
        final = converter.convert_stream_event(
            RawMessageDeltaEvent(
                type="message_delta",
                delta={"stop_reason": "end_turn", "stop_sequence": None},
                usage={"output_tokens": 7},
            ),
            state,
        )
        usage = final.usage
    else:
        usage = converter.convert_response(message).usage

    prompt_tokens = 5 + sum(cache_usage.values())
    assert usage.prompt_tokens == prompt_tokens
    assert usage.completion_tokens == 7
    assert usage.total_tokens == prompt_tokens + 7
    assert (usage.prompt_tokens_details.cached_tokens or 0) == cache_usage.get(
        "cache_read_input_tokens", 0
    )
