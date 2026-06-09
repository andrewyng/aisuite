"""The `remember` tool — the agent's explicit write path into memory."""

from __future__ import annotations

from typing import Optional

import aisuite as ai

from .base import MemoryStore, Scope

_SCOPES = {s.value for s in Scope}


def memory_tools(store: MemoryStore, *, workspace: Optional[str]) -> list:
    def remember(content: str, scope: str = "workspace") -> dict:
        """Save a durable memory (a fact or preference) to recall in future sessions.

        Args:
            content (str): The thing to remember.
            scope (str): "workspace" (this project) or "global" (everywhere).
        """
        chosen = Scope(scope) if scope in _SCOPES else Scope.WORKSPACE
        item = store.add(
            content,
            scope=chosen,
            workspace=workspace if chosen is Scope.WORKSPACE else None,
        )
        return {"id": item.id, "scope": item.scope.value, "saved": True}

    return [
        ai.tool(
            remember,
            metadata=ai.ToolMetadata(
                category="memory",
                risk_level="low",
                capabilities=["remember"],
            ),
        )
    ]
