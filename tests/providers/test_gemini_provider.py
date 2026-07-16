"""Tests for the Gemini provider (Gemini Developer API via google-genai)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aisuite.framework.message import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
)
from aisuite.providers.gemini_provider import GeminiMessageConverter, GeminiProvider

PIXEL = "aGk="  # any base64 payload will do for conversion tests
DATA_URL = f"data:image/png;base64,{PIXEL}"


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")


@pytest.fixture
def converter():
    return GeminiMessageConverter()


# -- request conversion ------------------------------------------------------------------
def test_system_message_becomes_system_instruction(converter):
    system, contents = converter.convert_request(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system == "Be brief."
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_tool_round_trip_matches_results_by_name(converter):
    """Gemini calls carry no ids; results map back to the function NAME."""
    _, contents = converter.convert_request(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": '{"temp": 21}'},
        ]
    )
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"function_call": {"name": "get_weather", "args": {"city": "Paris"}}}
        ],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "name": "get_weather",
                    "response": {"temp": 21},
                }
            }
        ],
    }


def test_message_objects_are_accepted(converter):
    """The tool runner appends framework Message objects, not dicts."""
    message = Message(
        role="assistant",
        content=None,
        refusal=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_0",
                type="function",
                function=Function(name="lookup", arguments='{"q": "x"}'),
            )
        ],
    )
    _, contents = converter.convert_request(
        [{"role": "user", "content": "go"}, message]
    )
    assert contents[1]["parts"][0]["function_call"]["name"] == "lookup"


def test_consecutive_same_role_messages_fold(converter):
    _, contents = converter.convert_request(
        [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
    )
    assert contents == [{"role": "user", "parts": [{"text": "one"}, {"text": "two"}]}]


def test_leading_model_turn_gets_placeholder_user(converter):
    _, contents = converter.convert_request(
        [{"role": "assistant", "content": "earlier answer"}]
    )
    assert contents[0] == {"role": "user", "parts": [{"text": "(continued)"}]}


def test_data_url_image_becomes_inline_data(converter):
    _, contents = converter.convert_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ],
            }
        ]
    )
    assert contents[0]["parts"] == [
        {"text": "What is this?"},
        {"inline_data": {"mime_type": "image/png", "data": PIXEL}},
    ]


def test_http_image_url_raises(converter):
    """Gemini's API cannot fetch plain URLs — fail clearly instead of a 400."""
    with pytest.raises(ValueError, match="cannot fetch"):
        converter.convert_request(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.jpg"},
                        }
                    ],
                }
            ]
        )


def test_unknown_part_type_raises(converter):
    with pytest.raises(ValueError, match="content part type"):
        converter.convert_request(
            [{"role": "user", "content": [{"type": "input_audio"}]}]
        )


def test_tool_spec_sanitizes_schema_and_omits_empty_parameters(converter):
    tools = converter.convert_tool_spec(
        [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search things.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "properties": {
                            "q": {"type": "string", "additionalProperties": False}
                        },
                        "required": ["q"],
                    },
                },
            },
            {
                "type": "function",
                "function": {"name": "ping", "parameters": {"type": "object"}},
            },
        ]
    )
    declarations = tools[0]["function_declarations"]
    assert declarations[0]["parameters"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    assert "parameters" not in declarations[1]  # empty object omitted entirely


# -- response conversion -----------------------------------------------------------------
def _response(parts, finish="STOP", usage=True):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=SimpleNamespace(name=finish),
            )
        ],
        usage_metadata=(
            SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
                cached_content_token_count=2,
            )
            if usage
            else None
        ),
    )


def _text_part(text):
    return SimpleNamespace(text=text, function_call=None)


def _call_part(name, args):
    return SimpleNamespace(
        text=None, function_call=SimpleNamespace(name=name, args=args)
    )


def test_convert_response_text(converter):
    response = converter.convert_response(_response([_text_part("Hello")]))
    assert response.choices[0].message.content == "Hello"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 15
    assert response.usage.prompt_tokens_details.cached_tokens == 2


def test_convert_response_tool_calls_get_synthesized_ids(converter):
    response = converter.convert_response(
        _response(
            [
                _call_part("get_weather", {"city": "Paris"}),
                _call_part("get_time", {"tz": "CET"}),
            ]
        )
    )
    calls = response.choices[0].message.tool_calls
    assert [call.id for call in calls] == ["call_0", "call_1"]
    assert json.loads(calls[0].function.arguments) == {"city": "Paris"}
    assert response.choices[0].finish_reason == "tool_calls"


def test_finish_reason_max_tokens_maps_to_length(converter):
    response = converter.convert_response(
        _response([_text_part("x")], finish="MAX_TOKENS")
    )
    assert response.choices[0].finish_reason == "length"


# -- provider wiring ---------------------------------------------------------------------
def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key is missing"):
        GeminiProvider()


def test_create_sends_converted_request():
    provider = GeminiProvider()
    provider.client = Mock()
    provider.client.models.generate_content = Mock(
        return_value=_response([_text_part("hi there")])
    )

    response = provider.chat_completions_create(
        "gemini-2.5-flash",
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ],
        temperature=0.5,
        max_tokens=128,
        frequency_penalty=0.5,  # unsupported → dropped
    )

    call = provider.client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert call.kwargs["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    config = call.kwargs["config"]
    assert config["system_instruction"] == "Be brief."
    assert config["temperature"] == 0.5
    assert config["max_output_tokens"] == 128
    assert "frequency_penalty" not in config
    assert response.choices[0].message.content == "hi there"


def test_sync_stream_yields_openai_shaped_chunks():
    provider = GeminiProvider()
    provider.client = Mock()
    provider.client.models.generate_content_stream = Mock(
        return_value=iter(
            [
                _response([_text_part("Hel")], finish=None, usage=False),
                _response([_text_part("lo")], finish=None, usage=False),
                _response([_call_part("get_weather", {"city": "Paris"})]),
            ]
        )
    )

    chunks = list(
        provider.chat_completions_create_stream(
            "gemini-2.5-flash", [{"role": "user", "content": "hi"}]
        )
    )

    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert text == "Hello"
    fragments = [
        fragment
        for chunk in chunks
        for fragment in (chunk.choices[0].delta.tool_calls or [])
    ]
    assert fragments[0].id == "call_0"
    assert fragments[0].function.name == "get_weather"
    assert json.loads(fragments[0].function.arguments) == {"city": "Paris"}
    final = chunks[-1]
    assert final.choices[0].finish_reason == "tool_calls"
    assert final.usage.total_tokens == 15


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
async def test_async_paths_use_native_async_client():
    provider = GeminiProvider()
    provider.client = Mock()
    provider.client.aio.models.generate_content = AsyncMock(
        return_value=_response([_text_part("async hi")])
    )
    provider.client.aio.models.generate_content_stream = AsyncMock(
        return_value=_AsyncIter([_response([_text_part("streamed")])])
    )

    response = await provider.achat_completions_create(
        "gemini-2.5-flash", [{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == "async hi"

    chunks = [
        chunk
        async for chunk in provider.achat_completions_create_stream(
            "gemini-2.5-flash", [{"role": "user", "content": "hi"}]
        )
    ]
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert text == "streamed"
    assert chunks[-1].choices[0].finish_reason == "stop"
