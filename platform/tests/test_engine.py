"""P2 gate tests — turn engine + event bus (scripted provider, no network)."""

from __future__ import annotations

import asyncio

import aisuite as ai
from coworker.engine import ApprovalOutcome, PermissionRequest, TurnEngine
from coworker.events import EventType
from coworker.permissions import PermissionEngine
from coworker.providers import (
    AssistantTurn,
    ModelCapabilities,
    ProviderClient,
    StreamChunk,
    ToolCall,
)
from coworker.tools import ToolRegistry


def _text_turn(text):
    return AssistantTurn(text=text, finish_reason="stop")


def _tool_turn(name, args, call_id="call_1"):
    return AssistantTurn(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


class ScriptedProvider(ProviderClient):
    """Returns queued AssistantTurns; streams via the base default (one final chunk)."""

    def __init__(self, turns, *, loop=False):
        self._turns = list(turns)
        self._loop = loop
        self.calls = 0

    def complete(self, *, model, messages, tools=None, **settings):
        self.calls += 1
        return self._turns[0] if self._loop else self._turns.pop(0)

    def capabilities(self, model):
        return ModelCapabilities()


def _engine(tmp_path, turns, *, approver=None, loop=False, max_iterations=12):
    provider = ScriptedProvider(turns, loop=loop)
    registry = ToolRegistry()
    registry.register_all(ai.toolkits.files(root=str(tmp_path), allow_write=True))
    permissions = PermissionEngine(workspace_root=tmp_path)
    engine = TurnEngine(
        provider=provider,
        registry=registry,
        permissions=permissions,
        model="gpt-5.5",
        approver=approver,
        max_iterations=max_iterations,
    )
    return engine, provider


def _collect(engine, user_input):
    async def _run():
        return [ev async for ev in engine.run(user_input)]

    return asyncio.run(_run())


def _types(events):
    return [ev.type for ev in events]


# -- tests ----------------------------------------------------------------------


def test_no_tool_turn(tmp_path):
    engine, _ = _engine(tmp_path, [_text_turn("all done")])
    events = _collect(engine, "hi")
    assert _types(events) == [
        EventType.TURN_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.TURN_END,
    ]
    assert events[1].data["text"] == "all done"
    assert events[-1].data["status"] == "completed"


def test_tool_turn_order_and_execution(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    engine, _ = _engine(
        tmp_path,
        [_tool_turn("read_file", {"path": "a.txt"}), _text_turn("it says hello")],
    )
    events = _collect(engine, "read a.txt")
    assert EventType.PERMISSION_REQUIRED not in _types(events)
    assert _types(events) == [
        EventType.TURN_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.TOOL_PROPOSED,
        EventType.TOOL_STARTED,
        EventType.TOOL_FINISHED,
        EventType.ITERATION_END,
        EventType.ASSISTANT_MESSAGE,
        EventType.TURN_END,
    ]
    finished = next(e for e in events if e.type == EventType.TOOL_FINISHED)
    assert finished.data["status"] == "ok"
    assert any(
        m.get("role") == "tool" and "hello" in m["content"] for m in engine.messages
    )


def test_write_requires_approval_then_approved(tmp_path):
    async def approve_once(_req: PermissionRequest):
        return ApprovalOutcome.ONCE

    engine, _ = _engine(
        tmp_path,
        [
            _tool_turn("write_file", {"path": "new.py", "content": "print(1)\n"}),
            _text_turn("wrote new.py"),
        ],
        approver=approve_once,
    )
    events = _collect(engine, "create new.py")
    assert EventType.PERMISSION_REQUIRED in _types(events)
    assert (tmp_path / "new.py").read_text() == "print(1)\n"


def test_denied_tool_yields_error_and_continues(tmp_path):
    async def deny(_req: PermissionRequest):
        return ApprovalOutcome.DENY

    engine, _ = _engine(
        tmp_path,
        [
            _tool_turn("write_file", {"path": "new.py", "content": "x"}),
            _text_turn("ok, skipped it"),
        ],
        approver=deny,
    )
    events = _collect(engine, "create new.py")
    assert not (tmp_path / "new.py").exists()
    finished = next(e for e in events if e.type == EventType.TOOL_FINISHED)
    assert finished.data["status"] == "denied"
    assert _types(events)[-1] == EventType.TURN_END
    assert any(
        m.get("role") == "tool" and "not executed" in m["content"]
        for m in engine.messages
    )


def test_max_iterations_rail(tmp_path):
    engine, provider = _engine(
        tmp_path, [_tool_turn("list_files", {})], loop=True, max_iterations=3
    )
    events = _collect(engine, "loop forever")
    end = events[-1]
    assert end.type == EventType.TURN_END
    assert end.data["status"] == "max_iterations_exceeded"
    assert provider.calls == 3


def test_interrupt_between_iterations(tmp_path):
    engine_holder = {}

    async def approve_and_interrupt(_req: PermissionRequest):
        engine_holder["engine"].request_interrupt()
        return ApprovalOutcome.ONCE

    engine, provider = _engine(
        tmp_path,
        [
            _tool_turn("write_file", {"path": "x.py", "content": "x"}),
            _text_turn("should not be reached"),
        ],
        approver=approve_and_interrupt,
    )
    engine_holder["engine"] = engine
    events = _collect(engine, "do a thing")
    assert events[-1].type == EventType.INTERRUPTED
    assert provider.calls == 1


def test_steering_injects_next_turn(tmp_path):
    engine, provider = _engine(tmp_path, [_text_turn("first"), _text_turn("second")])
    engine.queue_steering("actually, also do this")
    events = _collect(engine, "do the first thing")
    assert provider.calls == 2
    assert any(
        m.get("role") == "user" and m["content"] == "actually, also do this"
        for m in engine.messages
    )
    assert events[-1].data["status"] == "completed"


class StreamingProvider(ProviderClient):
    def complete(self, **kwargs):  # pragma: no cover - streamed instead
        raise NotImplementedError

    def capabilities(self, model):
        return ModelCapabilities()

    def stream(self, *, model, messages, tools=None, **settings):
        for piece in ["Hel", "lo, ", "world"]:
            yield StreamChunk(text_delta=piece)
        yield StreamChunk(turn=AssistantTurn(text="Hello, world", finish_reason="stop"))


def test_streaming_emits_deltas(tmp_path):
    registry = ToolRegistry()
    permissions = PermissionEngine(workspace_root=tmp_path)
    engine = TurnEngine(
        provider=StreamingProvider(),
        registry=registry,
        permissions=permissions,
        model="gpt-5.5",
    )
    events = _collect(engine, "say hi")
    deltas = [e.data["text"] for e in events if e.type == EventType.ASSISTANT_DELTA]
    assert deltas == ["Hel", "lo, ", "world"]
    final = next(e for e in events if e.type == EventType.ASSISTANT_MESSAGE)
    assert final.data["text"] == "Hello, world"
    assert events[-1].type == EventType.TURN_END
