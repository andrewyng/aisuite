from __future__ import annotations

import copy
from typing import Any, Callable, Optional

from ..client import Client
from ..tracing.sinks import TraceEvent, TraceSink, emit_event, get_configured_sinks
from .context import ActiveRunContext, reset_active_run_context, set_active_run_context
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
        parent_run_id: Optional[str] = None,
        group_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tool_policy: Optional[ToolPolicy | Callable] = None,
        trace_sinks: Optional[list[TraceSink]] = None,
        tracing_disabled: bool = False,
        **kwargs: Any,
    ) -> RunResult:
        return Runner.run_sync(
            agent,
            input,
            client=client,
            max_turns=max_turns,
            run_name=run_name,
            parent_run_id=parent_run_id,
            group_id=group_id,
            tags=tags,
            metadata=metadata,
            tool_policy=tool_policy,
            trace_sinks=trace_sinks,
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
        parent_run_id: Optional[str] = None,
        group_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tool_policy: Optional[ToolPolicy | Callable] = None,
        trace_sinks: Optional[list[TraceSink]] = None,
        tracing_disabled: bool = False,
        **kwargs: Any,
    ) -> RunResult:
        active_client = client or Client()
        trace_id = None if tracing_disabled else new_id("trace")
        active_trace_id = trace_id or ""
        active_sinks = (
            [] if tracing_disabled else (trace_sinks or get_configured_sinks())
        )
        if isinstance(input, RunState):
            messages = copy.deepcopy(input.messages)
            effective_run_name = run_name if run_name is not None else input.run_name
            effective_parent_run_id = (
                parent_run_id if parent_run_id is not None else input.parent_run_id
            )
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
            effective_parent_run_id = parent_run_id
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
                "trace_id": active_trace_id,
                "parent_run_id": effective_parent_run_id,
                "group_id": effective_group_id,
                "tags": effective_tags,
                "metadata": effective_metadata,
                "messages": copy.deepcopy(messages),
            }

        agent_step = RunStep(
            id=new_id("step"),
            type="agent",
            name=agent.name,
            trace_id=active_trace_id,
            started_at=now(),
            data={
                "agent_name": agent.name,
                "model": agent.model,
                "run_name": effective_run_name,
            },
        )

        Runner._emit_trace_event(
            active_sinks,
            "run.started",
            active_trace_id,
            agent.name,
            run_name=effective_run_name,
            parent_run_id=effective_parent_run_id,
            group_id=effective_group_id,
            tags=effective_tags,
            metadata=effective_metadata,
            data={
                "input": copy.deepcopy(
                    input.to_dict() if isinstance(input, RunState) else input
                ),
                "model": agent.model,
            },
        )
        Runner._emit_trace_event(
            active_sinks,
            "model.started",
            active_trace_id,
            agent.name,
            span_id=agent_step.id,
            run_name=effective_run_name,
            parent_run_id=effective_parent_run_id,
            group_id=effective_group_id,
            tags=effective_tags,
            metadata=effective_metadata,
            data={"model": agent.model, "message_count": len(messages)},
        )
        context_token = set_active_run_context(
            ActiveRunContext(
                client=active_client,
                trace_id=active_trace_id,
                group_id=effective_group_id,
                tags=copy.deepcopy(effective_tags),
                metadata=copy.deepcopy(effective_metadata),
                trace_sinks=active_sinks,
                tool_policy=tool_policy,
            )
        )
        try:
            response = active_client.chat.completions.create(
                model=agent.model,
                messages=copy.deepcopy(messages),
                **request_kwargs,
            )
            status: RunStatus = "completed"
        except Exception as exc:
            agent_step.ended_at = now()
            Runner._emit_trace_event(
                active_sinks,
                "run.failed",
                active_trace_id,
                agent.name,
                span_id=agent_step.id,
                run_name=effective_run_name,
                parent_run_id=effective_parent_run_id,
                group_id=effective_group_id,
                tags=effective_tags,
                metadata=effective_metadata,
                data={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise
        finally:
            reset_active_run_context(context_token)

        agent_step.ended_at = now()
        all_messages = extract_response_messages(response, messages)
        raw_responses = [
            *getattr(response, "intermediate_responses", []),
            response,
        ]
        steps = [
            *prior_steps,
            agent_step,
            *Runner._build_response_steps(raw_responses, active_trace_id),
            *Runner._build_tool_steps(response, active_trace_id),
        ]

        result = RunResult(
            final_output=extract_final_output(response),
            status=status,
            agent=agent,
            last_agent=agent,
            input=input,
            messages=all_messages,
            new_items=all_messages[len(messages) :],
            raw_responses=raw_responses,
            run_name=effective_run_name,
            trace_id=active_trace_id,
            parent_run_id=effective_parent_run_id,
            group_id=effective_group_id,
            tags=effective_tags,
            metadata=effective_metadata,
            steps=steps,
            max_turns=effective_max_turns,
            _client=active_client,
        )
        Runner._emit_trace_event(
            active_sinks,
            "model.completed",
            active_trace_id,
            agent.name,
            span_id=agent_step.id,
            run_name=effective_run_name,
            parent_run_id=effective_parent_run_id,
            group_id=effective_group_id,
            tags=effective_tags,
            metadata=effective_metadata,
            data={
                "final_output": copy.deepcopy(result.final_output),
                "message_count": len(result.messages),
            },
        )
        for event in getattr(response, "tool_events", []):
            Runner._emit_tool_event(
                active_sinks,
                event,
                active_trace_id,
                agent.name,
                effective_run_name,
                effective_parent_run_id,
                effective_group_id,
                effective_tags,
                effective_metadata,
            )
        Runner._emit_trace_event(
            active_sinks,
            "run.completed",
            active_trace_id,
            agent.name,
            run_name=effective_run_name,
            parent_run_id=effective_parent_run_id,
            group_id=effective_group_id,
            tags=effective_tags,
            metadata=effective_metadata,
            data={"run": result.trace_to_dict()},
        )
        return result

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

    @staticmethod
    def _emit_trace_event(
        sinks: list[TraceSink],
        event_type: str,
        trace_id: str,
        agent_name: str,
        *,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        group_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        if not sinks or not trace_id:
            return
        emit_event(
            sinks,
            TraceEvent(
                event_type=event_type,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                parent_run_id=parent_run_id,
                group_id=group_id,
                run_name=run_name,
                agent_name=agent_name,
                tags=copy.deepcopy(tags or []),
                metadata=copy.deepcopy(metadata or {}),
                data=copy.deepcopy(data or {}),
            ),
        )

    @staticmethod
    def _emit_tool_event(
        sinks: list[TraceSink],
        tool_event: dict[str, Any],
        trace_id: str,
        agent_name: str,
        run_name: Optional[str],
        parent_run_id: Optional[str],
        group_id: Optional[str],
        tags: list[str],
        metadata: dict[str, Any],
    ) -> None:
        if tool_event.get("type") == "tool_result":
            event_type = "tool.completed"
        elif tool_event.get("allowed") is False:
            event_type = "tool.denied"
        else:
            event_type = "tool.allowed"
        Runner._emit_trace_event(
            sinks,
            event_type,
            trace_id,
            agent_name,
            parent_run_id=parent_run_id,
            group_id=group_id,
            run_name=run_name,
            tags=tags,
            metadata=metadata,
            data=copy.deepcopy(tool_event),
        )
