from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, TextIO

from ..client import Client

RunStatus = Literal["completed", "requires_input", "max_turns_exceeded", "failed"]
ToolRiskLevel = Literal["low", "medium", "high"]
RunStepType = Literal[
    "agent",
    "model_response",
    "tool_call",
    "tool_result",
    "handoff",
    "custom",
]


def ensure_json_serializable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise TypeError(
            "RunState contains values that are not JSON serializable."
        ) from exc
    return value


@dataclass(kw_only=True)
class Agent:
    """Declarative agent definition."""

    name: str
    model: str
    instructions: Optional[str] = None
    tools: list[Callable] = field(default_factory=list)
    model_settings: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunStep:
    id: str
    type: RunStepType
    name: Optional[str]
    trace_id: str
    parent_id: Optional[str] = None
    started_at: str = ""
    ended_at: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "data": copy.deepcopy(self.data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunStep":
        return cls(
            id=data["id"],
            type=data["type"],
            name=data.get("name"),
            trace_id=data["trace_id"],
            parent_id=data.get("parent_id"),
            started_at=data.get("started_at", ""),
            ended_at=data.get("ended_at"),
            data=copy.deepcopy(data.get("data", {})),
        )


@dataclass
class ToolPolicyDecision:
    allowed: bool
    reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetadata:
    name: Optional[str] = None
    category: Optional[str] = None
    risk_level: ToolRiskLevel = "low"
    capabilities: list[str] = field(default_factory=list)
    requires_approval: bool = False
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return ensure_json_serializable(
            {
                "name": self.name,
                "category": self.category,
                "risk_level": self.risk_level,
                "capabilities": copy.deepcopy(self.capabilities),
                "requires_approval": self.requires_approval,
                "description": self.description,
                "metadata": copy.deepcopy(self.metadata),
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolMetadata":
        return cls(
            name=data.get("name"),
            category=data.get("category"),
            risk_level=data.get("risk_level", "low"),
            capabilities=copy.deepcopy(data.get("capabilities", [])),
            requires_approval=data.get("requires_approval", False),
            description=data.get("description"),
            metadata=copy.deepcopy(data.get("metadata", {})),
        )


@dataclass
class ToolPolicyContext:
    agent_name: str
    tool_name: str
    arguments: dict[str, Any]
    run_name: Optional[str]
    trace_id: Optional[str]
    group_id: Optional[str]
    tags: list[str]
    metadata: dict[str, Any]
    messages: list[dict[str, Any]]
    parent_run_id: Optional[str] = None
    tool_metadata: Optional[ToolMetadata] = None


class ToolPolicy(Protocol):
    def evaluate(self, context: ToolPolicyContext) -> bool | ToolPolicyDecision: ...


@dataclass
class RunState:
    agent_name: str
    messages: list[dict[str, Any]]
    status: RunStatus = "completed"
    run_name: Optional[str] = None
    trace_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    group_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[RunStep] = field(default_factory=list)
    max_turns: int = 5

    def add_user_message(self, input: str | list[dict[str, Any]]) -> None:
        from .utils import build_input_messages

        self.messages.extend(build_input_messages(input))

    def to_dict(self) -> dict[str, Any]:
        data = {
            "agent_name": self.agent_name,
            "messages": copy.deepcopy(self.messages),
            "status": self.status,
            "run_name": self.run_name,
            "trace_id": self.trace_id,
            "parent_run_id": self.parent_run_id,
            "group_id": self.group_id,
            "tags": copy.deepcopy(self.tags),
            "metadata": copy.deepcopy(self.metadata),
            "steps": [step.to_dict() for step in self.steps],
            "max_turns": self.max_turns,
        }
        return ensure_json_serializable(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(
            agent_name=data["agent_name"],
            messages=copy.deepcopy(data.get("messages", [])),
            status=data.get("status", "completed"),
            run_name=data.get("run_name"),
            trace_id=data.get("trace_id"),
            parent_run_id=data.get("parent_run_id"),
            group_id=data.get("group_id"),
            tags=copy.deepcopy(data.get("tags", [])),
            metadata=copy.deepcopy(data.get("metadata", {})),
            steps=[RunStep.from_dict(step) for step in data.get("steps", [])],
            max_turns=data.get("max_turns", 5),
        )


@dataclass
class RunResult:
    final_output: Any
    status: RunStatus
    agent: Agent
    last_agent: Agent
    input: str | list[dict[str, Any]] | RunState
    messages: list[dict[str, Any]]
    new_items: list[dict[str, Any]]
    raw_responses: list[Any]
    run_name: Optional[str]
    trace_id: str
    parent_run_id: Optional[str]
    group_id: Optional[str]
    tags: list[str]
    metadata: dict[str, Any]
    steps: list[RunStep]
    max_turns: int
    _client: Optional[Client] = field(default=None, repr=False, compare=False)

    def to_state(self) -> RunState:
        return RunState(
            agent_name=self.last_agent.name,
            messages=copy.deepcopy(self.messages),
            status=self.status,
            run_name=self.run_name,
            trace_id=self.trace_id,
            parent_run_id=self.parent_run_id,
            group_id=self.group_id,
            tags=copy.deepcopy(self.tags),
            metadata=copy.deepcopy(self.metadata),
            steps=copy.deepcopy(self.steps),
            max_turns=self.max_turns,
        )

    def trace_to_dict(self) -> dict[str, Any]:
        data = {
            "trace_id": self.trace_id,
            "parent_run_id": self.parent_run_id,
            "group_id": self.group_id,
            "run_name": self.run_name,
            "agent_name": self.last_agent.name,
            "status": self.status,
            "tags": copy.deepcopy(self.tags),
            "metadata": copy.deepcopy(self.metadata),
            "final_output": copy.deepcopy(self.final_output),
            "messages": copy.deepcopy(self.messages),
            "new_items": copy.deepcopy(self.new_items),
            "message_count": len(self.messages),
            "step_count": len(self.steps),
            "steps": [step.to_dict() for step in self.steps],
        }
        return ensure_json_serializable(data)

    def write_trace_jsonl(self, path: str | Path) -> None:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(self.trace_to_dict()) + "\n")

    def print_trace(self, file: Optional[TextIO] = None) -> None:
        output = file or sys.stdout
        print(
            f"Trace {self.trace_id} "
            f"run={self.run_name or '-'} "
            f"agent={self.last_agent.name} "
            f"status={self.status}",
            file=output,
        )
        if self.group_id:
            print(f"Group: {self.group_id}", file=output)
        if self.tags:
            print(f"Tags: {', '.join(self.tags)}", file=output)
        if self.metadata:
            metadata = ", ".join(
                f"{key}={value}" for key, value in sorted(self.metadata.items())
            )
            print(f"Metadata: {metadata}", file=output)
        if self.final_output is not None:
            print(f"Final output: {self.final_output}", file=output)

        for step in self.steps:
            name = step.name or "-"
            status = step.data.get("status")
            allowed = step.data.get("allowed")
            details = []
            if allowed is not None:
                details.append(f"allowed={allowed}")
            if status:
                details.append(f"status={status}")
            if step.data.get("reason"):
                details.append(f"reason={step.data['reason']}")
            suffix = f" ({', '.join(details)})" if details else ""
            print(f"- {step.type}: {name}{suffix}", file=output)
