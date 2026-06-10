"""Agent — a top-level surface (Code / Chat / Cowork).

An agent owns its system prompt + base toolset + whether it needs a workspace. Distinct
from a Skill: skills are Anthropic-format, loadable capabilities that ANY agent can pull
in (see coworker.skills).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..tools.todo import TodoList


@dataclass
class AgentContext:
    workspace: Optional[Path] = None
    executor: Optional[Any] = None
    todo: Optional[TodoList] = None
    # Shared, mutable list of RootDir the session may touch (primary scratch + added folders).
    # When None, tools fall back to the single `workspace` root. Held by reference so runtime
    # add/remove of folders is seen by the file tools built from it.
    roots: Optional[list] = None


@dataclass
class Agent:
    name: str
    title: str
    system_prompt: str
    needs_workspace: bool = False
    tool_factory: Optional[Callable[[AgentContext], list]] = None

    def build_tools(self, context: AgentContext) -> list:
        return list(self.tool_factory(context)) if self.tool_factory else []
