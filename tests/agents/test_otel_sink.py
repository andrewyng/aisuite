"""Tests for OtelTraceSink: aisuite trace events → OpenTelemetry spans."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from aisuite.tracing.otel import OtelTraceSink
from aisuite.tracing.sinks import TraceEvent


@pytest.fixture()
def exporter_and_sink():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, OtelTraceSink(tracer_provider=provider)


def _event(event_type, trace_id="trace_1", span_id=None, data=None, run_name="demo"):
    return TraceEvent(
        event_type=event_type,
        trace_id=trace_id,
        agent_name="agent",
        run_name=run_name,
        span_id=span_id,
        data=data or {},
    )


def test_model_send_response_becomes_one_span(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("model.send", data={"model": "openai:gpt-4o"}))
    sink.emit(
        _event(
            "model.response",
            data={
                "model": "openai:gpt-4o",
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "model openai:gpt-4o"
    assert span.attributes["gen_ai.request.model"] == "openai:gpt-4o"
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.status.status_code == StatusCode.UNSET


def test_run_span_parents_model_and_tool_spans(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("run.started"))
    sink.emit(_event("model.send", span_id="step_1", data={"model": "m"}))
    sink.emit(_event("model.response", span_id="step_1", data={"model": "m"}))
    sink.emit(_event("tool.started", span_id="call_1", data={"name": "get_weather"}))
    sink.emit(_event("tool.completed", span_id="call_1", data={"name": "get_weather"}))
    sink.emit(_event("run.completed"))

    spans = {span.name: span for span in exporter.get_finished_spans()}
    run_span = spans["run demo"]
    assert spans["model m"].parent.span_id == run_span.context.span_id
    assert spans["tool get_weather"].parent.span_id == run_span.context.span_id
    # all three share one OTel trace
    assert (
        spans["model m"].context.trace_id
        == spans["tool get_weather"].context.trace_id
        == run_span.context.trace_id
    )


def test_error_events_set_error_status(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("model.send", data={"model": "m"}))
    sink.emit(_event("model.error", data={"model": "m", "error": "rate limited"}))

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert "rate limited" in span.status.description


def test_point_events_attach_to_open_run(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("run.started"))
    sink.emit(_event("tool.denied", data={"name": "rm_rf"}))
    sink.emit(_event("run.completed"))

    (run_span,) = exporter.get_finished_spans()
    assert [e.name for e in run_span.events] == ["tool.denied"]


def test_unpaired_terminal_event_still_recorded(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("model.response", data={"model": "m"}))

    (span,) = exporter.get_finished_spans()
    assert span.attributes["aisuite.event_type"] == "model.response"


def test_close_ends_dangling_spans(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("run.started"))
    sink.emit(_event("model.send", data={"model": "m"}))
    assert exporter.get_finished_spans() == ()

    sink.close()
    assert len(exporter.get_finished_spans()) == 2


def test_separate_traces_do_not_share_parents(exporter_and_sink):
    exporter, sink = exporter_and_sink
    sink.emit(_event("run.started", trace_id="trace_a", run_name="a"))
    sink.emit(_event("run.started", trace_id="trace_b", run_name="b"))
    sink.emit(_event("model.send", trace_id="trace_b", data={"model": "m"}))
    sink.emit(_event("model.response", trace_id="trace_b", data={"model": "m"}))
    sink.emit(_event("run.completed", trace_id="trace_a", run_name="a"))
    sink.emit(_event("run.completed", trace_id="trace_b", run_name="b"))

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["model m"].parent.span_id == spans["run b"].context.span_id
    assert spans["run a"].parent is None
