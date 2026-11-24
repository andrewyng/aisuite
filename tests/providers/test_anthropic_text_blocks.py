"""Preserve the complete text of native Anthropic responses."""

import json

import pytest
from anthropic.types import Message as AnthropicMessage

from aisuite.providers.anthropic_provider import AnthropicMessageConverter


@pytest.mark.parametrize("with_tool", [False, True])
def test_response_preserves_all_text_blocks(with_tool):
    content = [
        {"type": "text", "text": "First paragraph.\n"},
        {"type": "thinking", "thinking": "private reasoning", "signature": "signed"},
        {"type": "text", "text": "Second paragraph."},
    ]
    if with_tool:
        content.insert(
            1,
            {
                "type": "tool_use",
                "id": "tool_weather",
                "name": "weather",
                "input": {"city": "Paris"},
            },
        )
    response = AnthropicMessage(
        id="msg_text_blocks",
        model="claude-sonnet-4-5",
        role="assistant",
        type="message",
        stop_reason="tool_use" if with_tool else "end_turn",
        content=content,
        usage={"input_tokens": 10, "output_tokens": 8},
    )

    converted = AnthropicMessageConverter().convert_response(response)
    message = converted.choices[0].message

    assert message.content == "First paragraph.\nSecond paragraph."
    if with_tool:
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "tool_weather"
        assert json.loads(message.tool_calls[0].function.arguments) == {"city": "Paris"}
    else:
        assert message.tool_calls is None


def test_response_without_text_keeps_content_none():
    response = AnthropicMessage(
        id="msg_thinking",
        model="claude-sonnet-4-5",
        role="assistant",
        type="message",
        stop_reason="max_tokens",
        content=[{"type": "thinking", "thinking": "reasoning", "signature": "signed"}],
        usage={"input_tokens": 10, "output_tokens": 8},
    )
    assert (
        AnthropicMessageConverter()
        .convert_response(response)
        .choices[0]
        .message.content
        is None
    )
