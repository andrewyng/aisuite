"""Tests for async tool execution (Tools.aexecute_tool)."""

import asyncio
import json
import threading

import pytest

from aisuite.utils.tools import Tools


def _tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
async def test_aexecute_awaits_async_tool():
    """An async def tool is awaited and its result captured."""

    async def fetch(city: str):
        """Fetch the weather for a city."""
        await asyncio.sleep(0)
        return {"city": city, "weather": "sunny"}

    tools = Tools([fetch])
    results, messages = await tools.aexecute_tool(_tool_call("fetch", {"city": "SF"}))

    assert results[0] == {"city": "SF", "weather": "sunny"}
    assert messages[0]["role"] == "tool"
    assert json.loads(messages[0]["content"]) == {"city": "SF", "weather": "sunny"}
    assert messages[0]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_aexecute_runs_sync_tool_off_the_loop():
    """A blocking sync tool runs in a worker thread, not the event loop."""
    seen = {}

    def blocking(city: str):
        """A synchronous tool."""
        seen["thread"] = threading.current_thread().name
        return {"city": city}

    tools = Tools([blocking])
    results, _ = await tools.aexecute_tool(_tool_call("blocking", {"city": "SF"}))

    assert results[0] == {"city": "SF"}
    assert seen["thread"] != threading.main_thread().name


@pytest.mark.asyncio
async def test_aexecute_respects_deny_policy():
    """Policy denial short-circuits before the tool runs, same as sync."""
    ran = {"value": False}

    async def secret(key: str):
        """A tool that should never run when denied."""
        ran["value"] = True
        return {"key": key}

    tools = Tools([secret])
    results, messages = await tools.aexecute_tool(
        _tool_call("secret", {"key": "token"}),
        tool_policy=lambda context: False,
    )

    assert ran["value"] is False
    assert results[0]["error"] == "Tool call denied by policy"
    # The pre-invocation events record the denial.
    assert any(e.get("allowed") is False for e in tools.last_tool_events)


def test_sync_execute_tool_still_works():
    """The sync path is unchanged after the refactor."""

    def add(a: int, b: int):
        """Add two numbers."""
        return a + b

    tools = Tools([add])
    results, messages = tools.execute_tool(_tool_call("add", {"a": 2, "b": 3}))
    assert results[0] == 5
    assert messages[0]["tool_call_id"] == "call_1"
