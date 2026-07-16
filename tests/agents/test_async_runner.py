"""Tests for the async Runner (Runner.run / continue_run) and sync wrappers."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aisuite import Agent, Client, Runner
from tests.agents.helpers import chat_response


@pytest.mark.asyncio
async def test_run_drives_async_client():
    client = Client()
    client.chat.completions.acreate = AsyncMock(return_value=chat_response("hello"))
    agent = Agent(name="assistant", model="openai:gpt-4o")

    result = await Runner.run(agent, "Say hi", client=client)

    assert result.final_output == "hello"
    assert result.status == "completed"
    client.chat.completions.acreate.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_matches_run_sync_output():
    """Async run and sync run produce equivalent results for the same input."""
    sync_client = Client()
    sync_client.chat.completions.create = Mock(return_value=chat_response("hi"))
    async_client = Client()
    async_client.chat.completions.acreate = AsyncMock(return_value=chat_response("hi"))
    agent = Agent(name="assistant", model="openai:gpt-4o")

    sync_result = Runner.run_sync(agent, "hey", client=sync_client)
    async_result = await Runner.run(agent, "hey", client=async_client)

    assert async_result.final_output == sync_result.final_output
    assert async_result.messages == sync_result.messages


@pytest.mark.asyncio
async def test_concurrent_runs_overlap():
    """Multiple Runner.run calls can be gathered and run concurrently."""
    client = Client()

    async def fake_acreate(*, model, messages, **kwargs):
        await asyncio.sleep(0.05)
        return chat_response(f"reply-{messages[-1]['content']}")

    client.chat.completions.acreate = fake_acreate
    agent = Agent(name="a", model="openai:gpt-4o")

    results = await asyncio.gather(
        *[Runner.run(agent, f"q{i}", client=client) for i in range(5)]
    )

    assert [r.final_output for r in results] == [f"reply-q{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_run_sync_works_inside_running_loop():
    """run_sync must still work when called from within an active event loop."""
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("hi"))
    agent = Agent(name="a", model="openai:gpt-4o")

    # We are inside the pytest-asyncio loop here; the nest_asyncio fallback runs.
    result = Runner.run_sync(agent, "hello", client=client)

    assert result.final_output == "hi"


@pytest.mark.asyncio
async def test_continue_run_async():
    client = Client()
    client.chat.completions.acreate = AsyncMock(
        side_effect=[chat_response("first"), chat_response("second")]
    )
    agent = Agent(name="assistant", model="openai:gpt-4o")

    first = await Runner.run(agent, "one", client=client)
    second = await Runner.continue_run(first, "two")

    assert first.final_output == "first"
    assert second.final_output == "second"
