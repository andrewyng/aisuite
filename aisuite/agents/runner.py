from __future__ import annotations

import copy
from typing import Any, Callable, Optional

from ..client import Client
from .types import Agent, RunResult, RunState, RunStatus, RunStep, ToolPolicy
from .utils import (
    build_input_messages,
    extract_final_message,
    extract_final_output,
    extract_response_messages,
    merge_tags,
    new_id,
    now,
)


class Runner:
    @staticmethod
    async def run(
        agent: Agent,
        input: str | list[dict[str, Any]] | RunState,
        *,
        client: Optional[Client] = None,
        max_turns: int = 5,
        run_name: Optional[str] = None,
        group_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tool_policy: Optional[ToolPolicy | Callable] = None,
        tracing_disabled: bool = False,
        **kwargs: Any,
    ) -> RunResult:
        return Runner.run_sync(
            agent,
            input,
            client=client,
            max_turns=max_turns,
            run_name=run_name,
            group_id=group_id,
            tags=tags,
            metadata=metadata,
            tool_policy=tool_policy,
            tracing_disabled=tracing_disabled,
            **kwargs,
        )

    @staticmethod
    def run_sync(
        agent: Agent,
        input: str | list[dict[str, Any]] | RunState,
        *,
        client: Optional[Client] = None,
        max_turns: int = 5,
        run_name: Optional[str] = None,
        group_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tool_policy: Optional[ToolPolicy | Callable] = None,
        tracing_disabled: bool = False,
        **kwargs: Any,
    ) -> RunResult:
        active_client = client or Client()
        trace_id = None if tracing_disabled else new_id("trace")
        if isinstance(input, RunState):
            messages = copy.deepcopy(input.messages)
            effective_run_name = run_name if run_name is not None else input.run_name
            effective_group_id = group_id if group_id is not None else input.group_id
            effective_tags = merge_tags(agent.tags, input.tags, tags)
            effective_metadata = {
                **agent.metadata,
                **input.metadata,
                **(metadata or {}),
            }
            effective_max_turns = max_turns if max_turns != 5 else input.max_turns
            prior_steps = copy.deepcopy(input.steps)
        else:
            messages = Runner._build_messages(agent, input)
            effective_run_name = run_name
            effective_group_id = group_id
            effective_tags = merge_tags(agent.tags, tags)
            effective_metadata = {**agent.metadata, **(metadata or {})}
            effective_max_turns = max_turns
            prior_steps = []

        request_kwargs = {**agent.model_settings, **kwargs}
        if agent.tools:
            request_kwargs["tools"] = agent.tools
            request_kwargs["max_turns"] = effective_max_turns
        if tool_policy is not None:
            request_kwargs["tool_policy"] = tool_policy
            request_kwargs["tool_policy_context"] = {
                "agent_name": agent.name,
                "run_name": effective_run_name,
                "trace_id": trace_id,
                "group_id": effective_group_id,
                "tags": effective_tags,
                "metadata": effective_metadata,
                "messages": copy.deepcopy(messages),
            }

        agent_step = RunStep(
            id=new_id("step"),
            type="agent",
            name=agent.name,
            trace_id=trace_id or "",
            started_at=now(),
            data={
                "agent_name": agent.name,
                "model": agent.model,
                "run_name": effective_run_name,
            },
        )

        try:
            response = active_client.chat.completions.create(
                model=agent.model,
                messages=copy.deepcopy(messages),
                **request_kwargs,
            )
            status: RunStatus = "completed"
        except Exception:
            agent_step.ended_at = now()
            raise

        agent_step.ended_at = now()
        all_messages = extract_response_messages(response, messages)
        raw_responses = [
            *getattr(response, "intermediate_responses", []),
            response,
        ]
        steps = [
            *prior_steps,
            agent_step,
            *Runner._build_response_steps(raw_responses, trace_id or ""),
            *Runner._build_tool_steps(response, trace_id or ""),
        ]

        return RunResult(
            final_output=extract_final_output(response),
            status=status,
            agent=agent,
            last_agent=agent,
            input=input,
            messages=all_messages,
            new_items=all_messages[len(messages) :],
            raw_responses=raw_responses,
            run_name=effective_run_name,
            trace_id=trace_id or "",
            group_id=effective_group_id,
            tags=effective_tags,
            metadata=effective_metadata,
            steps=steps,
            max_turns=effective_max_turns,
            _client=active_client,
        )

    @staticmethod
    async def continue_run(
        result: RunResult,
        input: str | list[dict[str, Any]],
        **overrides: Any,
    ) -> RunResult:
        return Runner.continue_sync(result, input, **overrides)

    @staticmethod
    def continue_sync(
        result: RunResult,
        input: str | list[dict[str, Any]],
        **overrides: Any,
    ) -> RunResult:
        state = result.to_state()
        state.add_user_message(input)
        return Runner.run_sync(
            result.last_agent,
            state,
            client=overrides.pop("client", result._client),
            **overrides,
        )

    @staticmethod
    def _build_messages(
        agent: Agent, input: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        messages = build_input_messages(input)
        if not agent.instructions:
            return messages
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": agent.instructions}, *messages]

    @staticmethod
    def _build_response_steps(raw_responses: list[Any], trace_id: str) -> list[RunStep]:
        steps = []
        for response in raw_responses:
            message = extract_final_message(response)
            data = {
                "has_message": message is not None,
                "finish_reason": getattr(
                    getattr(response, "choices", [None])[0], "finish_reason", None
                )
                if getattr(response, "choices", None)
                else None,
            }
            ended_at = now()
            steps.append(
                RunStep(
                    id=new_id("step"),
                    type="model_response",
                    name="model_response",
                    trace_id=trace_id,
                    started_at=ended_at,
                    ended_at=ended_at,
                    data=data,
                )
            )
        return steps

    @staticmethod
    def _build_tool_steps(response: Any, trace_id: str) -> list[RunStep]:
        events = getattr(response, "tool_events", [])
        steps = []
        for event in events:
            ended_at = now()
            step_type = (
                "tool_result" if event.get("type") == "tool_result" else "tool_call"
            )
            steps.append(
                RunStep(
                    id=new_id("step"),
                    type=step_type,
                    name=event.get("tool_name"),
                    trace_id=trace_id,
                    started_at=ended_at,
                    ended_at=ended_at,
                    data=copy.deepcopy(event),
                )
            )
        return steps
