from __future__ import annotations

import re
from typing import Optional

from .context import get_active_run_context
from .runner import Runner
from .types import Agent


def _tool_name(value: str) -> str:
    name = re.sub(r"\W+", "_", value).strip("_")
    return name or "agent_tool"


def agent_tool(
    agent: Agent,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Expose an Agent as a callable tool for another Agent."""

    tool_name = _tool_name(name or agent.name)

    def run_subagent(input: str) -> str:
        """Run a subagent with the provided input."""
        context = get_active_run_context()
        result = Runner.run_sync(
            agent,
            input,
            client=context.client if context else None,
            run_name=tool_name,
            parent_run_id=context.trace_id if context else None,
            group_id=context.group_id if context else None,
            tags=context.tags if context else None,
            metadata=context.metadata if context else None,
            tool_policy=context.tool_policy if context else None,
            trace_sinks=context.trace_sinks if context else None,
            artifact_store=context.artifact_store if context else None,
        )
        return "" if result.final_output is None else str(result.final_output)

    run_subagent.__name__ = tool_name
    run_subagent.__doc__ = description or (
        f"Run the {agent.name} agent and return its final output."
    )
    return run_subagent
