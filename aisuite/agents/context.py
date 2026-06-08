from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

from ..client import Client
from ..tracing.sinks import TraceSink
from .types import ToolPolicy


@dataclass
class ActiveRunContext:
    client: Client
    trace_id: str
    group_id: Optional[str]
    tags: list[str]
    metadata: dict[str, Any]
    trace_sinks: list[TraceSink]
    tool_policy: Optional[ToolPolicy]


_active_run_context: ContextVar[Optional[ActiveRunContext]] = ContextVar(
    "aisuite_active_run_context",
    default=None,
)


def get_active_run_context() -> Optional[ActiveRunContext]:
    return _active_run_context.get()


def set_active_run_context(context: ActiveRunContext):
    return _active_run_context.set(context)


def reset_active_run_context(token) -> None:
    _active_run_context.reset(token)
