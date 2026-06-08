"""Tests for the async client path (Completions.acreate)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aisuite import Client
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message


def _chat_response(content=None, tool_calls=None):
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id, type="function", function=Function(name=name, arguments=arguments)
    )


@pytest.mark.asyncio
@patch("aisuite.provider.ProviderFactory.create_provider")
async def test_acreate_runs_full_tool_loop(mock_create_provider):
    provider = Mock()
    provider.achat_completions_create = AsyncMock(
        side_effect=[
            _chat_response(
                tool_calls=[_tool_call("get_weather", '{"location": "San Francisco"}')]
            ),
            _chat_response(content="It is sunny in San Francisco."),
        ]
    )
    mock_create_provider.return_value = provider

    def get_weather(location: str):
        """Get the weather for a location."""
        return {"location": location, "condition": "sunny"}

    client = Client()
    response = await client.chat.completions.acreate(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "What is the weather?"}],
        tools=[get_weather],
        max_turns=2,
    )

    assert response.choices[0].message.content == "It is sunny in San Francisco."
    assert provider.achat_completions_create.await_count == 2


@pytest.mark.asyncio
@patch("aisuite.provider.ProviderFactory.create_provider")
async def test_acreate_without_tools(mock_create_provider):
    provider = Mock()
    provider.achat_completions_create = AsyncMock(
        return_value=_chat_response(content="hello")
    )
    mock_create_provider.return_value = provider

    client = Client()
    response = await client.chat.completions.acreate(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.choices[0].message.content == "hello"
    provider.achat_completions_create.assert_awaited_once()


@pytest.mark.asyncio
@patch("aisuite.provider.ProviderFactory.create_provider")
async def test_acreate_awaits_async_tool(mock_create_provider):
    """An async def tool is awaited inside the async tool loop."""
    provider = Mock()
    provider.achat_completions_create = AsyncMock(
        side_effect=[
            _chat_response(tool_calls=[_tool_call("lookup", '{"q": "x"}')]),
            _chat_response(content="done"),
        ]
    )
    mock_create_provider.return_value = provider

    async def lookup(q: str):
        """An async tool."""
        return {"q": q, "value": 42}

    client = Client()
    response = await client.chat.completions.acreate(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "look it up"}],
        tools=[lookup],
        max_turns=3,
    )

    assert response.choices[0].message.content == "done"
