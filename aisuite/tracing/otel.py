"""OpenTelemetry sink for aisuite traces.

``OtelTraceSink`` translates aisuite ``TraceEvent``s into OpenTelemetry spans,
so traces flow to any OTLP-compatible backend (Langfuse, Grafana, Jaeger,
Datadog, ...) through the standard OTel pipeline the application configures:

    from opentelemetry import trace as otel_trace
    from aisuite.tracing import configure
    from aisuite.tracing.otel import OtelTraceSink

    configure(OtelTraceSink())  # uses the globally configured TracerProvider

The event stream is already span-shaped: ``run.started``/``model.send``/
``tool.started`` open spans and their terminal counterparts close them, with
``span_id``/``trace_id`` correlating the pairs. Point-in-time events
(``tool.allowed``/``tool.denied``) attach as span events on the enclosing run.

Requires the ``opentelemetry-api`` package (``pip install 'aisuite[otel]'``);
the application supplies the SDK/exporter of its choice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from .sinks import TraceEvent

# Terminal event → the prefix of its opening event; used to pair spans.
_SPAN_OPENERS = {
    "run.completed": "run",
    "run.failed": "run",
    "model.response": "model",
    "model.error": "model",
    "tool.completed": "tool",
    "tool.failed": "tool",
}
_ERROR_EVENTS = {"run.failed", "model.error", "tool.failed"}


def _event_time_ns(event: TraceEvent) -> Optional[int]:
    try:
        return int(datetime.fromisoformat(event.timestamp).timestamp() * 1e9)
    except (TypeError, ValueError):
        return None


def _span_key(event: TraceEvent, kind: str) -> tuple:
    # span_id pairs runner events; the client's model events carry no span_id
    # but are emitted send→response sequentially, so the trace-scoped kind key
    # still matches them up.
    return (event.trace_id, kind, event.span_id)


def _attributes(event: TraceEvent) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "aisuite.event_type": event.event_type,
        "aisuite.trace_id": event.trace_id,
    }
    if event.agent_name:
        attributes["aisuite.agent_name"] = event.agent_name
    if event.run_name:
        attributes["aisuite.run_name"] = event.run_name
    if event.tags:
        attributes["aisuite.tags"] = list(event.tags)
    model = event.data.get("model")
    if model:
        attributes["gen_ai.system"] = "aisuite"
        attributes["gen_ai.request.model"] = str(model)
    for key, value in event.data.items():
        if isinstance(value, (str, int, float, bool)) and key != "model":
            attributes[f"aisuite.{key}"] = value
    return attributes


def _usage_attributes(event: TraceEvent) -> dict[str, Any]:
    usage = event.data.get("usage")
    if not isinstance(usage, dict):
        return {}
    attributes = {}
    if usage.get("prompt_tokens") is not None:
        attributes["gen_ai.usage.input_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        attributes["gen_ai.usage.output_tokens"] = usage["completion_tokens"]
    return attributes


class OtelTraceSink:
    """Emit aisuite trace events as OpenTelemetry spans."""

    def __init__(self, tracer_provider: Any = None):
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import Status, StatusCode
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OtelTraceSink requires the 'opentelemetry-api' package. "
                "Install it with: pip install 'aisuite[otel]'"
            ) from exc
        self._otel_trace = otel_trace
        self._status = Status
        self._status_code = StatusCode
        self._tracer = otel_trace.get_tracer("aisuite", tracer_provider=tracer_provider)
        # (trace_id, kind, span_id) → open span; run spans double as parents.
        self._open: dict[tuple, Any] = {}
        self._run_spans: dict[str, Any] = {}

    def emit(self, event: TraceEvent) -> None:
        event_type = event.event_type

        if event_type == "run.started":
            self._open_span(event, "run", self._span_name(event, "run"))
        elif event_type == "model.send":
            self._open_span(event, "model", self._span_name(event, "model"))
        elif event_type == "tool.started":
            self._open_span(event, "tool", self._span_name(event, "tool"))
        elif event_type in _SPAN_OPENERS:
            self._close_span(event, _SPAN_OPENERS[event_type])
        else:  # point-in-time events: tool.allowed / tool.denied
            self._record_point_event(event)

    def close(self) -> None:
        """End any spans left open (e.g. a crashed run)."""
        for span in list(self._open.values()):
            span.end()
        self._open.clear()
        self._run_spans.clear()

    # -- internals ---------------------------------------------------------------
    @staticmethod
    def _span_name(event: TraceEvent, kind: str) -> str:
        if kind == "run":
            return f"run {event.run_name or event.agent_name or event.trace_id}"
        if kind == "tool":
            tool = event.data.get("name") or event.data.get("tool_name") or "tool"
            return f"tool {tool}"
        model = event.data.get("model")
        return f"model {model}" if model else "model call"

    def _open_span(self, event: TraceEvent, kind: str, name: str) -> None:
        parent = self._run_spans.get(event.trace_id)
        context = (
            self._otel_trace.set_span_in_context(parent) if parent is not None else None
        )
        span = self._tracer.start_span(
            name,
            context=context,
            start_time=_event_time_ns(event),
            attributes=_attributes(event),
        )
        self._open[_span_key(event, kind)] = span
        if kind == "run":
            self._run_spans[event.trace_id] = span

    def _close_span(self, event: TraceEvent, kind: str) -> None:
        span = self._open.pop(_span_key(event, kind), None)
        if span is None:
            # Terminal event without a recorded opener — represent it as a
            # point-in-time span so the information is not lost.
            self._record_point_event(event)
            return
        for key, value in {**_attributes(event), **_usage_attributes(event)}.items():
            span.set_attribute(key, value)
        if event.event_type in _ERROR_EVENTS:
            message = str(event.data.get("error") or event.event_type)
            span.set_status(self._status(self._status_code.ERROR, message))
        span.end(end_time=_event_time_ns(event))
        if kind == "run":
            self._run_spans.pop(event.trace_id, None)

    def _record_point_event(self, event: TraceEvent) -> None:
        parent = self._run_spans.get(event.trace_id)
        if parent is not None:
            parent.add_event(event.event_type, attributes=_attributes(event))
            return
        span = self._tracer.start_span(
            event.event_type,
            start_time=_event_time_ns(event),
            attributes=_attributes(event),
        )
        if event.event_type in _ERROR_EVENTS:
            span.set_status(
                self._status(
                    self._status_code.ERROR,
                    str(event.data.get("error") or event.event_type),
                )
            )
        span.end(end_time=_event_time_ns(event))
