"""Tests for the streaming client path (create/acreate with stream=True)."""

from unittest.mock import Mock, patch

import pytest

from aisuite import Client


def _stream_provider(chunks):
    provider = Mock()
    provider.chat_completions_create_stream = Mock(return_value=iter(chunks))
    return provider


@patch("aisuite.provider.ProviderFactory.create_provider")
def test_create_stream_routes_to_provider_stream(mock_create_provider):
    chunks = [object(), object()]
    provider = _stream_provider(chunks)
    mock_create_provider.return_value = provider

    client = Client()
    stream = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        temperature=0.2,
    )

    assert list(stream) == chunks
    call = provider.chat_completions_create_stream.call_args
    assert call.args == ("gpt-4o", [{"role": "user", "content": "hi"}])
    # `stream` is consumed by the client; extra settings pass through.
    assert call.kwargs == {"temperature": 0.2}
    provider.chat_completions_create.assert_not_called()


@patch("aisuite.provider.ProviderFactory.create_provider")
def test_create_stream_converts_tools(mock_create_provider):
    """Streaming is manual tool calling: schemas pass through, callables
    become specs — same contract as the non-streaming manual path (#266)."""
    provider = _stream_provider([])
    mock_create_provider.return_value = provider

    def get_weather(location: str):
        """Get the weather for a location."""
        return {"location": location}

    schema = {
        "type": "function",
        "function": {
            "name": "manual_tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    client = Client()
    client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[schema, get_weather],
        stream=True,
    )

    sent = provider.chat_completions_create_stream.call_args.kwargs["tools"]
    assert sent[0] == schema
    assert sent[1]["function"]["name"] == "get_weather"


@patch("aisuite.provider.ProviderFactory.create_provider")
def test_create_stream_with_max_turns_raises(mock_create_provider):
    mock_create_provider.return_value = _stream_provider([])

    client = Client()
    with pytest.raises(ValueError, match="max_turns"):
        client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
            max_turns=3,
            stream=True,
        )


@patch("aisuite.provider.ProviderFactory.create_provider")
def test_create_stream_rejects_mcp_configs(mock_create_provider):
    mock_create_provider.return_value = _stream_provider([])

    client = Client()
    with pytest.raises(ValueError, match="MCP"):
        client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "mcp", "name": "fs", "command": "npx", "args": []}],
            stream=True,
        )


@patch("aisuite.provider.ProviderFactory.create_provider")
def test_stream_false_keeps_non_streaming_path(mock_create_provider):
    provider = Mock()
    provider.chat_completions_create = Mock(return_value=Mock(choices=[]))
    mock_create_provider.return_value = provider

    client = Client()
    client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )

    provider.chat_completions_create.assert_called_once()
    provider.chat_completions_create_stream.assert_not_called()


@pytest.mark.asyncio
@patch("aisuite.provider.ProviderFactory.create_provider")
async def test_acreate_stream_returns_async_iterator(mock_create_provider):
    chunks = [object(), object()]

    async def astream(model, messages, **kwargs):
        for chunk in chunks:
            yield chunk

    provider = Mock()
    provider.achat_completions_create_stream = astream
    mock_create_provider.return_value = provider

    client = Client()
    stream = await client.chat.completions.acreate(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    received = [chunk async for chunk in stream]
    assert received == chunks
    provider.achat_completions_create.assert_not_called()


@pytest.mark.asyncio
@patch("aisuite.provider.ProviderFactory.create_provider")
async def test_acreate_stream_with_max_turns_raises_eagerly(mock_create_provider):
    """The guard fires at the acreate call, not on first iteration."""
    mock_create_provider.return_value = Mock()

    client = Client()
    with pytest.raises(ValueError, match="max_turns"):
        await client.chat.completions.acreate(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
            max_turns=2,
            stream=True,
        )
