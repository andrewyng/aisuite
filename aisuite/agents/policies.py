from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from .types import ToolMetadata, ToolPolicyContext, ToolPolicyDecision


def tool(fn: Callable, *, metadata: Optional[ToolMetadata] = None) -> Callable:
    """Attach aisuite metadata to a Python callable tool."""
    if metadata is not None and metadata.name is None:
        metadata.name = fn.__name__
    setattr(fn, "__aisuite_tool_metadata__", metadata)
    return fn


class AllowAllToolPolicy:
    def evaluate(self, context: ToolPolicyContext) -> ToolPolicyDecision:
        return ToolPolicyDecision(allowed=True)


class DenyAllToolPolicy:
    def __init__(self, reason: Optional[str] = None):
        self.reason = reason

    def evaluate(self, context: ToolPolicyContext) -> ToolPolicyDecision:
        return ToolPolicyDecision(allowed=False, reason=self.reason)


class AllowToolsPolicy:
    def __init__(self, allowed_tools: list[str], reason: Optional[str] = None):
        self.allowed_tools = set(allowed_tools)
        self.reason = reason

    def evaluate(self, context: ToolPolicyContext) -> ToolPolicyDecision:
        allowed = context.tool_name in self.allowed_tools
        return ToolPolicyDecision(
            allowed=allowed,
            reason=None if allowed else self.reason or "tool not in allowlist",
        )


class RequireApprovalPolicy:
    def __init__(
        self,
        callback: Callable[[ToolPolicyContext], bool | ToolPolicyDecision],
    ):
        self.callback = callback

    def evaluate(self, context: ToolPolicyContext) -> ToolPolicyDecision:
        decision = self.callback(context)
        if isinstance(decision, ToolPolicyDecision):
            return decision
        if isinstance(decision, bool):
            return ToolPolicyDecision(
                allowed=decision,
                reason="approved" if decision else "denied",
            )
        raise TypeError(
            "Approval callback must return a bool or ToolPolicyDecision instance."
        )
