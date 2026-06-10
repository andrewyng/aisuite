from __future__ import annotations

import argparse
import base64
import copy
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from .sinks import TraceStoreSink
from .store import InMemoryTraceStore, JsonlTraceStore, TraceStore

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_UI_DIST = Path(__file__).resolve().parent / "static" / "viewer"
REPO_UI_DIST = REPO_ROOT / "viewer-ui" / "dist"
DEFAULT_UI_DIST = PACKAGE_UI_DIST if PACKAGE_UI_DIST.exists() else REPO_UI_DIST


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>aisuite runs</title>
  <style>
    :root { color-scheme: light; --bg: #f6f7f9; --surface: #ffffff; --surface-soft: #fafbfc; --line: #dde3ea; --line-soft: #edf1f5; --text: #18212f; --muted: #667385; --blue: #276ef1; --green: #12715b; --red: #b42318; --amber: #9a5b00; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); letter-spacing: 0; }
    header { height: 64px; display: flex; align-items: center; justify-content: space-between; padding: 0 22px; background: rgba(255,255,255,0.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(14px); position: sticky; top: 0; z-index: 2; }
    main { display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: calc(100vh - 64px); }
    aside { border-right: 1px solid var(--line); background: var(--surface); overflow: auto; }
    section { padding: 24px; overflow: auto; }
    .brand { font-weight: 850; font-size: 15px; }
    .top-meta { color: var(--muted); font-size: 13px; }
    .run { padding: 13px 14px; border: 1px solid transparent; border-radius: 8px; cursor: pointer; margin: 7px 10px; transition: background .12s ease, border-color .12s ease, box-shadow .12s ease; }
    .run:hover { background: var(--surface-soft); border-color: var(--line-soft); }
    .run.active { background: #f7faff; border-color: #b9cffb; box-shadow: inset 3px 0 0 var(--blue); }
    .run-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .toolbar { display: grid; grid-template-columns: minmax(0, 1fr) 118px; gap: 8px; padding: 12px 10px 8px; position: sticky; top: 0; background: var(--surface); z-index: 1; border-bottom: 1px solid var(--line-soft); }
    .input, .select { width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; background: var(--surface-soft); color: var(--text); font: inherit; font-size: 13px; }
    .input:focus, .select:focus { outline: 2px solid #bfd1ff; border-color: #9bb8fb; background: white; }
    .group { padding: 18px 18px 7px; color: var(--muted); font-size: 11px; font-weight: 850; text-transform: uppercase; display: flex; justify-content: space-between; }
    .name { font-weight: 780; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .meta { color: var(--muted); font-size: 12px; margin-top: 5px; line-height: 1.35; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .pill { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; background: #eef2f6; color: #354154; font-size: 12px; font-weight: 700; margin: 6px 5px 0 0; }
    .pill.green { background: #e7f5f0; color: var(--green); }
    .pill.blue { background: #eaf1ff; color: #174ea6; }
    .pill.red { background: #fdeceb; color: var(--red); }
    .pill.amber { background: #fff5df; color: var(--amber); }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 16px; box-shadow: 0 12px 26px rgba(24, 33, 47, 0.05); overflow: hidden; }
    .panel h2 { margin: 0; padding: 14px 16px; font-size: 14px; border-bottom: 1px solid var(--line-soft); }
    .panel .content { padding: 16px; }
    .hero { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 16px; align-items: start; }
    .title { font-size: 20px; font-weight: 850; margin-bottom: 6px; }
    .summary { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-top: 18px; }
    .metric { border: 1px solid var(--line-soft); border-radius: 8px; padding: 12px; background: var(--surface-soft); }
    .metric .label { color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .metric .value { font-weight: 850; font-size: 20px; margin-top: 4px; }
    .output { white-space: pre-wrap; word-break: break-word; line-height: 1.52; color: #263244; }
    .tabs { display: flex; gap: 4px; padding: 0 12px; border-bottom: 1px solid var(--line-soft); background: var(--surface-soft); }
    .tab { border: 0; border-bottom: 2px solid transparent; background: transparent; padding: 13px 9px 11px; color: var(--muted); font: inherit; font-weight: 760; cursor: pointer; }
    .tab.active { color: var(--blue); border-bottom-color: var(--blue); }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .message { border: 1px solid var(--line-soft); border-radius: 8px; margin-bottom: 10px; overflow: hidden; background: var(--surface-soft); }
    .message .role { color: var(--muted); font-size: 11px; font-weight: 850; padding: 8px 10px; text-transform: uppercase; }
    .message .body { padding: 0 10px 10px; white-space: pre-wrap; word-break: break-word; line-height: 1.48; }
    .timeline { display: grid; gap: 10px; }
    .timeline-card { border: 1px solid var(--line-soft); border-radius: 8px; background: var(--surface-soft); padding: 13px; }
    .timeline-main { display: grid; grid-template-columns: 18px minmax(0, 1fr) auto; gap: 12px; align-items: start; }
    .timeline-title { font-weight: 800; }
    .tool-panel { margin-top: 12px; display: grid; gap: 10px; }
    .kv { display: grid; grid-template-columns: 140px minmax(0, 1fr); gap: 8px; padding: 7px 0; border-bottom: 1px solid var(--line-soft); }
    .kv:last-child { border-bottom: 0; }
    .kv-key { color: var(--muted); font-size: 12px; font-weight: 800; }
    .kv-value { white-space: pre-wrap; word-break: break-word; font-size: 13px; }
    .codebox { background: #101828; color: #e4eaf7; padding: 12px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; max-height: 240px; overflow: auto; font-size: 12px; line-height: 1.45; }
    .event { display: grid; grid-template-columns: 18px minmax(0, 1fr) auto; gap: 12px; padding: 13px 0; border-bottom: 1px solid var(--line-soft); }
    .event:last-child { border-bottom: 0; }
    .event-dot { width: 10px; height: 10px; border-radius: 999px; background: var(--blue); margin-top: 5px; box-shadow: 0 0 0 4px #eaf1ff; }
    .event-dot.green { background: var(--green); box-shadow: 0 0 0 4px #e7f5f0; }
    .event-dot.red { background: var(--red); box-shadow: 0 0 0 4px #fdeceb; }
    .event-dot.amber { background: var(--amber); box-shadow: 0 0 0 4px #fff5df; }
    .event-type { font-weight: 780; }
    .event-summary { color: var(--muted); font-size: 13px; margin-top: 4px; line-height: 1.42; }
    .event details { margin-top: 8px; }
    .event summary { cursor: pointer; color: var(--blue); font-size: 12px; font-weight: 760; }
    .step { padding: 11px 0; border-bottom: 1px solid var(--line-soft); }
    .step:last-child { border-bottom: 0; }
    pre { white-space: pre-wrap; word-break: break-word; background: #101828; color: #e4eaf7; padding: 12px; border-radius: 8px; overflow: auto; }
    .empty { color: #607086; padding: 22px; }
    @media (max-width: 800px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: 35vh; border-right: 0; border-bottom: 1px solid #d8dee8; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .toolbar { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div><span class="brand">aisuite runs</span><span id="count" class="top-meta" style="margin-left: 10px;"></span></div>
    <span class="pill blue">local viewer</span>
  </header>
  <main>
    <aside id="runs"></aside>
    <section id="detail"><div class="empty">No runs yet.</div></section>
  </main>
  <script>
    let runs = [];
    let selectedTraceId = null;
    let selectedTab = "overview";
    let query = "";
    let statusFilter = "all";

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function displayValue(value) {
      if (value === null || value === undefined) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function compactValue(value, maxLength = 180) {
      let rendered = displayValue(value).replaceAll("\\n", " ");
      if (rendered.length <= maxLength) return rendered;
      return rendered.slice(0, maxLength - 3) + "...";
    }

    function statusPill(status) {
      const normalized = String(status || "running");
      const cls = normalized === "completed" ? "green" : normalized === "failed" ? "red" : "amber";
      return `<span class="pill ${cls}">${escapeHtml(normalized)}</span>`;
    }

    function tonePill(label, tone) {
      return `<span class="pill ${escapeHtml(tone || "")}">${escapeHtml(label)}</span>`;
    }

    function eventTone(event) {
      if (event.event_type === "tool.completed" || event.event_type === "run.completed") return "green";
      if (event.event_type === "tool.denied" || event.event_type === "run.failed") return "red";
      if (event.event_type === "tool.allowed") return "amber";
      return "";
    }

    function shortTime(value) {
      if (!value) return "";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function eventSummary(event, run) {
      const data = event.data || {};
      const name = data.tool_name || event.name || event.agent_name || run.agent_name || "";
      if (event.event_type === "model.response") {
        const response = data.response || {};
        const preview = response.text_preview;
        const calls = response.tool_calls || [];
        const toolPart = calls.length ? ` · ${calls.map(call => {
          const id = call.id ? ` (${call.id})` : "";
          return `${call.name || "tool"}${id}`;
        }).join(", ")}` : "";
        if (preview) return `${compactValue(preview)}${toolPart}`;
        return `${response.kind || "response"}${toolPart}`;
      }
      if (event.event_type === "tool.allowed" || event.event_type === "tool.denied") {
        const args = data.arguments ? compactValue(data.arguments) : "";
        const status = data.allowed === false ? "denied" : "allowed";
        const reason = data.reason ? ` · ${data.reason}` : "";
        return `${name}(${args}) · ${status}${reason}`;
      }
      if (event.event_type === "tool.completed") {
        const status = data.status || "completed";
        const preview = data.result_preview ? ` · ${compactValue(data.result_preview)}` : "";
        return `${name} · ${status}${preview}`;
      }
      const bits = [];
      if (event.agent_name || run.agent_name) bits.push(event.agent_name || run.agent_name);
      if (event.run_name) bits.push(event.run_name);
      if (data.message_count !== undefined) bits.push(`${data.message_count} messages`);
      return bits.join(" · ");
    }

    function latestToolActivity(run) {
      return (run.display && run.display.latest_activity) || "";
    }

    function messageContent(message) {
      if (!message || message.content === undefined) return displayValue(message);
      return displayValue(message.content);
    }

    function renderRuns() {
      document.getElementById("count").textContent = `${runs.length} run${runs.length === 1 ? "" : "s"}`;
      const list = document.getElementById("runs");
      if (!runs.length) {
        list.innerHTML = '<div class="empty">Waiting for traces...</div>';
        return;
      }
      const filteredRuns = runs.filter(matchesFilters);
      const controls = `<div class="toolbar">
        <input class="input" placeholder="Search runs, tools, traces" value="${escapeHtml(query)}" oninput="query = this.value; renderRuns(); renderDetail();" />
        <select class="select" onchange="statusFilter = this.value; renderRuns(); renderDetail();">
          ${["all", "completed", "running", "failed"].map(status =>
            `<option value="${status}" ${status === statusFilter ? "selected" : ""}>${status}</option>`
          ).join("")}
        </select>
      </div>`;
      if (!filteredRuns.length) {
        list.innerHTML = controls + '<div class="empty">No matching runs.</div>';
        return;
      }
      const groups = {};
      for (const run of filteredRuns) {
        const key = run.group_id || "ungrouped";
        if (!groups[key]) groups[key] = [];
        groups[key].push(run);
      }
      list.innerHTML = controls + Object.entries(groups).map(([group, groupRuns]) => `
        <div class="group"><span>${escapeHtml(group)}</span><span>${groupRuns.length} run${groupRuns.length === 1 ? "" : "s"}</span></div>
        ${groupRuns.map(run => {
        const active = run.trace_id === selectedTraceId ? " active" : "";
        const display = run.display || {};
        const title = display.title || run.run_name || run.trace_id || "run";
        const tags = (run.tags || []).map(tag => `<span class="pill">${escapeHtml(tag)}</span>`).join("");
        const counts = `${run.message_count ?? (run.messages || []).length} messages · ${display.tool_count ?? 0} tool events · ${display.duration || "-"}`;
        const latestTool = latestToolActivity(run);
        return `<div class="run${active}" onclick="selectRun('${escapeHtml(run.trace_id)}')">
          <div class="run-head"><div class="name">${escapeHtml(title)}</div>${statusPill(run.status)}</div>
          <div class="meta">${escapeHtml(display.subtitle || run.agent_name || "")}</div>
          <div class="meta">${escapeHtml(counts)}</div>
          ${latestTool ? `<div class="meta">${escapeHtml(latestTool)}</div>` : ""}
          <div>${tags}</div>
        </div>`;
        }).join("")}
      `).join("");
    }

    function matchesFilters(run) {
      if (statusFilter !== "all" && run.status !== statusFilter) return false;
      if (!query.trim()) return true;
      const needle = query.trim().toLowerCase();
      const display = run.display || {};
      const haystack = [
        run.trace_id,
        run.run_name,
        run.agent_name,
        run.group_id,
        run.status,
        display.latest_activity,
        ...(display.tools || []),
      ].join(" ").toLowerCase();
      return haystack.includes(needle);
    }

    function renderKeyValues(value) {
      if (!value || typeof value !== "object") return "";
      return `<div class="tool-panel">${Object.entries(value).map(([key, item]) => `
        <div class="kv"><div class="kv-key">${escapeHtml(key)}</div><div class="kv-value">${escapeHtml(displayValue(item))}</div></div>
      `).join("")}</div>`;
    }

    function renderResult(item) {
      const result = item.result;
      if (result && typeof result === "object" && ("command" in result || "exit_code" in result || "stdout" in result || "stderr" in result)) {
        return `<div class="tool-panel">
          ${result.command ? `<div class="kv"><div class="kv-key">command</div><div class="kv-value mono">${escapeHtml(result.command)}</div></div>` : ""}
          ${"exit_code" in result ? `<div class="kv"><div class="kv-key">exit code</div><div class="kv-value">${escapeHtml(result.exit_code)}</div></div>` : ""}
          ${"timed_out" in result ? `<div class="kv"><div class="kv-key">timed out</div><div class="kv-value">${escapeHtml(result.timed_out)}</div></div>` : ""}
          ${result.stdout ? `<div><div class="kv-key">stdout</div><div class="codebox">${escapeHtml(result.stdout)}</div></div>` : ""}
          ${result.stderr ? `<div><div class="kv-key">stderr</div><div class="codebox">${escapeHtml(result.stderr)}</div></div>` : ""}
        </div>`;
      }
      if (result && typeof result === "object" && ("path" in result || "content" in result)) {
        return `<div class="tool-panel">
          ${renderKeyValues(Object.fromEntries(Object.entries(result).filter(([key]) => key !== "content")))}
          ${result.content !== undefined ? `<div><div class="kv-key">content</div><div class="codebox">${escapeHtml(result.content)}</div></div>` : ""}
        </div>`;
      }
      if (item.result_preview) {
        return `<div class="tool-panel"><div class="codebox">${escapeHtml(item.result_preview)}</div></div>`;
      }
      return "";
    }

    function renderTimelineItem(item) {
      const args = item.arguments ? `<div><div class="kv-key">arguments</div>${renderKeyValues(item.arguments)}</div>` : "";
      return `<div class="timeline-card">
        <div class="timeline-main">
          <div class="event-dot ${escapeHtml(item.tone || "blue")}"></div>
          <div>
            <div class="timeline-title">${escapeHtml(item.title || item.event_type)}</div>
            <div class="event-summary">${escapeHtml(item.summary || "")}</div>
          </div>
          <div class="meta mono">${escapeHtml(item.time || "")}</div>
        </div>
        ${args}
        ${renderResult(item)}
        <details><summary>Details</summary><pre>${escapeHtml(JSON.stringify(item.raw || {}, null, 2))}</pre></details>
      </div>`;
    }

    function setTab(tab) {
      selectedTab = tab;
      renderDetail();
    }

    function renderDetail() {
      const detail = document.getElementById("detail");
      const run = runs.find(item => item.trace_id === selectedTraceId) || runs[0];
      if (!run) {
        detail.innerHTML = '<div class="empty">No runs yet.</div>';
        return;
      }
      selectedTraceId = run.trace_id;
      const display = run.display || {};
      const metadata = Object.entries(run.metadata || {}).map(([key, value]) =>
        `<span class="pill">${escapeHtml(key)}=${escapeHtml(value)}</span>`
      ).join("");
      const tools = (display.tools || []).map(tool => `<span class="pill blue">${escapeHtml(tool)}</span>`).join("");
      const messages = (run.messages || []).map(message => {
        const role = message.role || "message";
        return `<div class="message">
          <div class="role">${escapeHtml(role)}</div>
          <div class="body">${escapeHtml(messageContent(message))}</div>
        </div>`;
      }).join("") || '<div class="empty">No messages.</div>';
      const timeline = (display.timeline || []).map(renderTimelineItem).join("") || '<div class="empty">No timeline events.</div>';
      const steps = (run.steps || []).map(step => {
        const data = step.data || {};
        const bits = [];
        if (data.allowed !== undefined) bits.push(`allowed=${data.allowed}`);
        if (data.status) bits.push(`status=${data.status}`);
        if (data.reason) bits.push(`reason=${data.reason}`);
        if (data.arguments) bits.push(`args=${compactValue(data.arguments)}`);
        if (data.result_preview) bits.push(`result=${compactValue(data.result_preview)}`);
        return `<div class="step">
          <strong>${escapeHtml(step.type)}</strong> ${escapeHtml(step.name || "")}
          <div class="meta">${escapeHtml(bits.join(" · "))}</div>
        </div>`;
      }).join("") || '<div class="empty">No steps.</div>';
      const events = (run.events || []).map(event => {
        const detail = event.data ? JSON.stringify(event.data, null, 2) : "";
        return `<div class="event">
          <div class="event-dot ${eventTone(event)}"></div>
          <div>
            <div class="event-type">${escapeHtml(event.event_type)}</div>
            <div class="event-summary">${escapeHtml(eventSummary(event, run))}</div>
            <details><summary>Details</summary><pre>${escapeHtml(detail)}</pre></details>
          </div>
          <div class="meta mono">${escapeHtml(shortTime(event.timestamp))}</div>
        </div>`;
      }).join("") || '<div class="empty">No events.</div>';
      const finalOutput = run.final_output === undefined || run.final_output === null
        ? '<div class="empty">No final output.</div>'
        : `<div class="output">${escapeHtml(displayValue(run.final_output))}</div>`;
      const active = tab => tab === selectedTab ? " active" : "";
      detail.innerHTML = `
        <div class="panel"><h2>Run</h2><div class="content">
          <div class="hero">
            <div>
              <div class="title">${escapeHtml(display.title || run.run_name || run.trace_id)}</div>
              <div class="meta">${escapeHtml(display.subtitle || "")}</div>
              <div class="meta">model: ${escapeHtml(display.model || "-")}</div>
              <div class="meta mono">trace: ${escapeHtml(run.trace_id)}</div>
              <div class="meta mono">parent: ${escapeHtml(run.parent_run_id || "-")}</div>
            </div>
            <div>${statusPill(run.status)}</div>
          </div>
          <div>${metadata}${tools}</div>
          <div class="summary">
            <div class="metric"><div class="label">Messages</div><div class="value">${escapeHtml(run.message_count ?? (run.messages || []).length)}</div></div>
            <div class="metric"><div class="label">Duration</div><div class="value">${escapeHtml(display.duration || "-")}</div></div>
            <div class="metric"><div class="label">Tools</div><div class="value">${escapeHtml(display.tool_count ?? 0)}</div></div>
            <div class="metric"><div class="label">Approvals</div><div class="value">${escapeHtml(display.approval_count ?? 0)}</div></div>
            <div class="metric"><div class="label">Events</div><div class="value">${escapeHtml(run.event_count ?? (run.events || []).length)}</div></div>
            <div class="metric"><div class="label">Errors</div><div class="value">${escapeHtml(display.error_count ?? 0)}</div></div>
          </div>
        </div></div>
        <div class="panel">
          <div class="tabs">
            <button class="tab${active("overview")}" onclick="setTab('overview')">Overview</button>
            <button class="tab${active("transcript")}" onclick="setTab('transcript')">Transcript</button>
            <button class="tab${active("events")}" onclick="setTab('events')">Events</button>
            <button class="tab${active("raw")}" onclick="setTab('raw')">Raw</button>
          </div>
          <div class="content">
            <div class="tab-panel${active("overview")}">
              <h2 style="padding: 0 0 10px; border: 0;">Final Output</h2>
              ${finalOutput}
              <h2 style="padding: 18px 0 10px; border: 0;">Timeline</h2>
              <div class="timeline">${timeline}</div>
            </div>
            <div class="tab-panel${active("transcript")}">${messages}</div>
            <div class="tab-panel${active("events")}">
              <h2 style="padding: 0 0 10px; border: 0;">Steps</h2>
              ${steps}
              <h2 style="padding: 18px 0 10px; border: 0;">Events</h2>
              ${events}
            </div>
            <div class="tab-panel${active("raw")}"><pre>${escapeHtml(JSON.stringify(run, null, 2))}</pre></div>
          </div>
        </div>
      `;
      renderRuns();
    }

    function selectRun(traceId) {
      selectedTraceId = traceId;
      renderDetail();
    }

    async function refresh() {
      const response = await fetch('/api/runs');
      const payload = await response.json();
      runs = payload.runs || [];
      if (!selectedTraceId && runs.length) selectedTraceId = runs[0].trace_id;
      renderRuns();
      renderDetail();
    }

    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
"""


def read_trace_file(trace_file: str | Path) -> list[dict[str, Any]]:
    return prepare_viewer_runs(JsonlTraceStore(trace_file).list_runs())


class ViewerTraceState:
    def __init__(
        self,
        trace_file: str | Path | None = None,
        *,
        store: TraceStore | None = None,
    ):
        self.trace_file = Path(trace_file) if trace_file else None
        self.store = store or (
            JsonlTraceStore(self.trace_file)
            if self.trace_file
            else InMemoryTraceStore()
        )
        self._lock = threading.Lock()

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return prepare_viewer_run_summaries(self.store.list_runs())

    def get_run(self, trace_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            selected = self.store.get_run(trace_id)
            if selected is None:
                return None
            runs = _include_selected_run(self.store.list_runs(), selected)
        for run in prepare_viewer_runs(runs):
            if run.get("trace_id") == trace_id:
                return _viewer_run_detail(run)
        return None

    def trace_sink(self) -> TraceStoreSink:
        return TraceStoreSink(self.store)

    def list_events(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self.store.list_events(trace_id)

    def import_jsonl(self, content: str) -> int:
        with self._lock:
            return self.store.import_jsonl(content)

    def add_event(self, record: dict[str, Any]) -> None:
        self.add_records([record])

    def add_records(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self.store.append_records(records)


def _include_selected_run(
    runs: list[dict[str, Any]],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    trace_id = selected.get("trace_id")
    if not trace_id:
        return [*runs, selected]
    replaced = False
    merged = []
    for run in runs:
        if run.get("trace_id") == trace_id:
            merged.append(selected)
            replaced = True
        else:
            merged.append(run)
    if not replaced:
        merged.append(selected)
    return merged


def prepare_viewer_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    child_counts: dict[str, int] = {}
    children_by_parent_tool: dict[tuple[str, str], dict[str, Any]] = {}
    titles = {
        run.get("trace_id"): run.get("run_name")
        or run.get("agent_name")
        or run.get("trace_id")
        for run in runs
        if run.get("trace_id")
    }
    for run in runs:
        parent_id = run.get("parent_run_id")
        if parent_id:
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
            run_name = run.get("run_name")
            if run_name:
                children_by_parent_tool[(parent_id, run_name)] = _child_run_summary(run)
    return [
        _prepare_viewer_run(
            run,
            child_count=child_counts.get(run.get("trace_id"), 0),
            parent_title=titles.get(run.get("parent_run_id")),
            child_tools={
                tool_name: summary
                for (parent_id, tool_name), summary in children_by_parent_tool.items()
                if parent_id == run.get("trace_id")
            },
        )
        for run in runs
    ]


def prepare_viewer_run_summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_viewer_run_summary(run) for run in prepare_viewer_runs(runs)]


def _viewer_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "trace_id",
        "parent_run_id",
        "group_id",
        "run_name",
        "agent_name",
        "status",
        "tags",
        "metadata",
        "final_output",
        "message_count",
        "step_count",
        "event_count",
    ]
    summary = {key: copy.deepcopy(run.get(key)) for key in keys if key in run}
    display = copy.deepcopy(run.get("display") or {})
    summary["display"] = {
        key: display.get(key)
        for key in [
            "title",
            "subtitle",
            "status_tone",
            "started_at",
            "ended_at",
            "duration",
            "model",
            "tools",
            "relationship",
            "parent_title",
            "child_count",
            "tool_count",
            "approval_count",
            "denied_count",
            "error_count",
            "usage",
            "latest_activity",
            "latency",
            "artifact_count",
        ]
        if key in display
    }
    return _sanitize_for_viewer(summary, max_string_chars=1000)


def _viewer_run_detail(run: dict[str, Any]) -> dict[str, Any]:
    detail = copy.deepcopy(run)
    detail["messages"] = _sanitize_for_viewer(detail.get("messages", []))
    detail["steps"] = _sanitize_for_viewer(detail.get("steps", []))
    detail["events"] = _sanitize_for_viewer(detail.get("events", []))
    detail["display"] = _sanitize_for_viewer(detail.get("display", {}))
    return detail


def _sanitize_for_viewer(value: Any, *, max_string_chars: int = 4000) -> Any:
    if _is_artifact_ref(value):
        return copy.deepcopy(value)
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        return {
            "type": "text_preview",
            "text_length": len(value),
            "line_count": len(value.splitlines()),
            "preview": value[: max_string_chars - 3] + "...",
            "truncated": True,
        }
    if isinstance(value, list):
        return [
            _sanitize_for_viewer(item, max_string_chars=max_string_chars)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _sanitize_for_viewer(item, max_string_chars=max_string_chars)
            for key, item in value.items()
        }
    return value


def _is_artifact_ref(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "artifact_ref"
        and isinstance(value.get("artifact_ref"), dict)
        and isinstance(value["artifact_ref"].get("artifact_id"), str)
    )


def _collect_artifact_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(item: Any) -> None:
        if _is_artifact_ref(item):
            ref = item.get("artifact_ref") or {}
            artifact_id = ref.get("artifact_id")
            if artifact_id and artifact_id not in seen:
                seen.add(artifact_id)
                refs.append(copy.deepcopy(item))
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return refs


def _prepare_viewer_run(
    run: dict[str, Any],
    *,
    child_count: int = 0,
    parent_title: Optional[str] = None,
    child_tools: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    prepared = dict(run)
    events = prepared.get("events", [])
    steps = prepared.get("steps", [])
    tool_events = [
        event
        for event in events
        if event.get("event_type")
        in {
            "tool.allowed",
            "tool.denied",
            "tool.started",
            "tool.completed",
            "tool.failed",
        }
    ]
    tool_names = sorted(
        {
            event.get("data", {}).get("tool_name")
            for event in tool_events
            if event.get("data", {}).get("tool_name")
        }
    )
    approvals = [
        event
        for event in tool_events
        if event.get("event_type") in {"tool.allowed", "tool.denied"}
    ]
    denied = [
        event for event in tool_events if event.get("event_type") == "tool.denied"
    ]
    failed_events = [
        event
        for event in events
        if event.get("event_type") in {"run.failed", "tool.failed", "model.error"}
    ]
    started_at = _first_timestamp(events) or _first_step_time(steps)
    ended_at = _last_timestamp(events) or _last_step_time(steps)
    activities = _run_activities(
        events,
        prepared,
        child_tools=child_tools or {},
    )
    artifact_refs = _collect_artifact_refs(prepared)
    display = {
        "title": prepared.get("run_name") or prepared.get("trace_id") or "run",
        "subtitle": _run_subtitle(prepared),
        "status_tone": _status_tone(prepared.get("status")),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration": _duration_label(started_at, ended_at),
        "model": _run_model(prepared),
        "tools": tool_names,
        "relationship": "child" if prepared.get("parent_run_id") else "root",
        "parent_title": parent_title,
        "child_count": child_count,
        "tool_count": len(tool_events),
        "approval_count": len(approvals),
        "denied_count": len(denied),
        "error_count": len(failed_events),
        "usage": _run_usage(events),
        "latest_activity": _latest_activity(prepared),
        "timeline": [
            _timeline_item(event, prepared, child_tools=child_tools or {})
            for event in events
        ],
        "activities": activities,
        "latency": _run_latency(activities),
        "artifact_refs": artifact_refs,
        "artifact_count": len(artifact_refs),
    }
    prepared["display"] = display
    return prepared


def _run_subtitle(run: dict[str, Any]) -> str:
    pieces = [
        run.get("agent_name") or "agent",
        run.get("group_id") or "ungrouped",
    ]
    return " · ".join(piece for piece in pieces if piece)


def _status_tone(status: Optional[str]) -> str:
    if status == "completed":
        return "green"
    if status == "failed":
        return "red"
    return "amber"


def _run_model(run: dict[str, Any]) -> str:
    for event in run.get("events", []):
        model = event.get("data", {}).get("model")
        if model:
            return model
    for step in run.get("steps", []):
        model = step.get("data", {}).get("model")
        if model:
            return model
    return "-"


def _latest_activity(run: dict[str, Any]) -> str:
    events = run.get("events", [])
    for event in reversed(events):
        if event.get("event_type") == "run.completed":
            continue
        summary = _event_summary(event, run)
        if summary:
            return summary
    return run.get("final_output") or "Waiting for activity"


def _timeline_item(
    event: dict[str, Any],
    run: dict[str, Any],
    *,
    child_tools: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    data = event.get("data", {}) or {}
    parsed_result = _parse_preview(data.get("result_preview"))
    tool_name = data.get("tool_name")
    artifact_refs = _collect_artifact_refs(data)
    return {
        "event_type": event.get("event_type"),
        "timestamp": event.get("timestamp"),
        "time": _short_time(event.get("timestamp")),
        "tone": _event_tone(event),
        "title": _event_title(event),
        "summary": _event_summary(event, run),
        "tool_name": tool_name,
        "arguments": data.get("arguments"),
        "result_preview": data.get("result_preview"),
        "result": parsed_result,
        "usage": data.get("usage"),
        "artifact_refs": artifact_refs,
        "artifact_count": len(artifact_refs),
        "child_run": child_tools.get(tool_name) if tool_name else None,
        "raw": _display_raw_data(data),
    }


def _display_raw_data(data: dict[str, Any]) -> dict[str, Any]:
    if "run" not in data:
        return data
    run = data.get("run") or {}
    return {
        **{key: value for key, value in data.items() if key != "run"},
        "run": {
            "trace_id": run.get("trace_id"),
            "status": run.get("status"),
            "message_count": run.get("message_count"),
            "step_count": run.get("step_count"),
        },
    }


def _event_title(event: dict[str, Any]) -> str:
    event_type = event.get("event_type") or "event"
    data = event.get("data", {}) or {}
    tool_name = data.get("tool_name")
    if event_type == "tool.allowed":
        return f"Allowed {tool_name}"
    if event_type == "tool.denied":
        return f"Denied {tool_name}"
    if event_type == "tool.completed":
        return f"Completed {tool_name}"
    if event_type == "tool.started":
        return f"Started {tool_name}"
    if event_type == "tool.failed":
        return f"Failed {tool_name}"
    if event_type == "model.send":
        return "Model send"
    if event_type == "model.response":
        return "Model response"
    if event_type == "model.error":
        return "Model error"
    if event_type == "run.started":
        return "Run started"
    if event_type == "run.completed":
        return "Run completed"
    if event_type == "run.failed":
        return "Run failed"
    return event_type


def _event_summary(event: dict[str, Any], run: dict[str, Any]) -> str:
    data = event.get("data", {}) or {}
    event_type = event.get("event_type")
    if event_type in {"tool.allowed", "tool.denied"}:
        args = _tool_arguments_summary(data.get("tool_name"), data.get("arguments"))
        status = "denied" if data.get("allowed") is False else "allowed"
        reason = data.get("reason")
        reason_part = f" · {reason}" if reason else ""
        return f"{data.get('tool_name')}({args}) · {status}{reason_part}"
    if event_type == "tool.started":
        return f"{data.get('tool_name')} started"
    if event_type == "tool.completed":
        preview = _tool_result_summary(
            data.get("tool_name"), data.get("result_preview")
        )
        return (
            f"{data.get('tool_name')} · {data.get('status', 'completed')} · {preview}"
        )
    if event_type == "tool.failed":
        return f"{data.get('tool_name')} · failed · {_compact_value(data.get('error'))}"
    if event_type == "model.send":
        model_input = data.get("input", {}) or {}
        modalities = ", ".join(model_input.get("modalities") or [])
        modality_part = f" · {modalities}" if modalities else ""
        return f"{data.get('model', _run_model(run))} · {model_input.get('message_count', 0)} messages{modality_part}"
    if event_type == "model.response":
        response = data.get("response", {}) or {}
        preview = response.get("text_preview")
        tool_count = response.get("tool_call_count", 0)
        tool_part = _tool_call_summary(response.get("tool_calls") or [], tool_count)
        usage_part = _usage_summary(data.get("usage"))
        text = (
            _compact_value(preview) + tool_part
            if preview
            else f"{response.get('kind', 'response')}{tool_part}"
        )
        return f"{text}{usage_part}"
    if event_type == "model.error":
        return _compact_value(data.get("error")) or "Model request failed"
    if event_type == "run.started":
        return _compact_value(data.get("input")) or "Run accepted"
    if event_type == "run.completed":
        return "Run completed"
    if event_type == "run.failed":
        return _compact_value(data.get("error")) or "Run failed"
    return ""


def _event_tone(event: dict[str, Any]) -> str:
    event_type = event.get("event_type")
    if event_type in {"tool.completed", "run.completed"}:
        return "green"
    if event_type in {"tool.denied", "tool.failed", "model.error", "run.failed"}:
        return "red"
    if event_type in {"tool.allowed", "tool.started"}:
        return "amber"
    return "blue"


def _compact_value(value: Any, max_chars: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, sort_keys=True)
        except TypeError:
            rendered = str(value)
    rendered = rendered.replace("\n", " ")
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."


def _tool_call_summary(tool_calls: list[dict[str, Any]], tool_count: int) -> str:
    if not tool_count:
        return ""
    labels = []
    for call in tool_calls:
        name = call.get("name") or "tool"
        call_id = call.get("id")
        labels.append(f"{name} ({call_id})" if call_id else name)
    if labels:
        return " · " + ", ".join(labels)
    return f" · {tool_count} tool call{'s' if tool_count != 1 else ''}"


def _run_activities(
    events: list[dict[str, Any]],
    run: dict[str, Any],
    *,
    child_tools: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline = [_timeline_item(event, run, child_tools=child_tools) for event in events]
    groups: list[dict[str, Any]] = []
    tool_groups: dict[str, dict[str, Any]] = {}
    tool_order: list[str] = []
    active_tool_keys: dict[str, str] = {}

    for index, item in enumerate(timeline):
        if item.get("event_type", "").startswith("tool."):
            key = _tool_activity_key(item, index, active_tool_keys)
            if key:
                if key not in tool_groups:
                    tool_groups[key] = {
                        "id": key,
                        "type": "tool_call",
                        "events": [],
                    }
                    tool_order.append(key)
                tool_groups[key]["events"].append(item)
                if item.get("event_type") in {
                    "tool.completed",
                    "tool.failed",
                    "tool.denied",
                }:
                    tool_name = item.get("tool_name")
                    if tool_name and active_tool_keys.get(tool_name) == key:
                        active_tool_keys.pop(tool_name, None)
                continue

        groups.append(
            {
                "id": f"{item.get('event_type')}-{index}",
                "type": "event",
                "events": [item],
            }
        )

    groups.extend(tool_groups[key] for key in tool_order)
    groups.sort(key=lambda group: group["events"][0].get("timestamp") or "")

    activities: list[dict[str, Any]] = []
    skip_next = False
    for index, group in enumerate(groups):
        if skip_next:
            skip_next = False
            continue
        item = group["events"][0]
        next_group = groups[index + 1] if index + 1 < len(groups) else None
        if (
            group["type"] == "event"
            and item.get("event_type") == "model.send"
            and next_group
            and next_group["type"] == "event"
            and next_group["events"][0].get("event_type") == "model.response"
        ):
            activities.append(
                _model_activity(
                    {
                        "id": f"model-{item.get('timestamp') or index}",
                        "type": "model_call",
                        "events": [item, next_group["events"][0]],
                    }
                )
            )
            skip_next = True
            continue
        if group["type"] == "tool_call":
            activities.append(_tool_activity(group))
        elif item.get("event_type") == "model.response":
            activities.append(
                _model_activity(
                    {
                        "id": f"model-{item.get('timestamp') or index}",
                        "type": "model_call",
                        "events": [item],
                    }
                )
            )
        else:
            activities.append(_event_activity(group))
    return activities


def _run_latency(activities: list[dict[str, Any]]) -> dict[str, Any]:
    model_calls = [
        activity
        for activity in activities
        if activity.get("type") == "model_call"
        and isinstance(activity.get("duration_ms"), (int, float))
    ]
    tool_calls = [
        activity
        for activity in activities
        if activity.get("type") in {"tool_call", "subagent_call"}
        and isinstance(activity.get("duration_ms"), (int, float))
    ]
    slowest_model = _slowest_activity(model_calls)
    slowest_tool = _slowest_activity(tool_calls)
    return {
        "model_call_count": len(model_calls),
        "tool_call_count": len(tool_calls),
        "slowest_model_ms": slowest_model.get("duration_ms") if slowest_model else None,
        "slowest_model": slowest_model,
        "slowest_tool_ms": slowest_tool.get("duration_ms") if slowest_tool else None,
        "slowest_tool": slowest_tool,
    }


def _slowest_activity(activities: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not activities:
        return None
    activity = max(activities, key=lambda item: item.get("duration_ms") or 0)
    return {
        "id": activity.get("id"),
        "type": activity.get("type"),
        "title": activity.get("title"),
        "summary": activity.get("summary"),
        "duration": activity.get("duration"),
        "duration_ms": activity.get("duration_ms"),
        "status": activity.get("status"),
        "tool_name": activity.get("tool_name"),
        "model": activity.get("model"),
    }


def _tool_activity_key(
    item: dict[str, Any],
    fallback: int,
    active_tool_keys: dict[str, str],
) -> Optional[str]:
    raw = item.get("raw") or {}
    call_id = raw.get("tool_call_id") or item.get("tool_call_id")
    if call_id:
        return f"tool:{call_id}"
    tool_name = item.get("tool_name")
    if tool_name:
        if tool_name in active_tool_keys:
            return active_tool_keys[tool_name]
        key = f"tool:{tool_name}:{fallback}"
        active_tool_keys[tool_name] = key
        return key
    return None


def _activity_base(group: dict[str, Any], activity_type: str) -> dict[str, Any]:
    events = group["events"]
    artifact_refs = _collect_artifact_refs(events)
    return {
        "id": group["id"],
        "type": activity_type,
        "schema_version": 1,
        "event_count": len(events),
        "event_types": [event.get("event_type") for event in events],
        "artifact_refs": artifact_refs,
        "artifact_count": len(artifact_refs),
        "events": events,
    }


def _tool_activity(group: dict[str, Any]) -> dict[str, Any]:
    events = group["events"]
    first = events[0]
    last = events[-1]
    completed = _find_event(events, "tool.completed")
    failed = _find_event(events, "tool.failed")
    allowed = _find_event(events, "tool.allowed")
    denied = _find_event(events, "tool.denied")
    started = _find_event(events, "tool.started")
    terminal = failed or completed or denied or last
    status = (
        "failed"
        if failed
        else "denied" if denied else "completed" if completed else "running"
    )
    tool_name = first.get("tool_name") or terminal.get("tool_name")
    raw = terminal.get("raw") or {}
    approval_event = denied or allowed
    approval_raw = approval_event.get("raw") if approval_event else {}
    approval = {
        "required": bool(
            (approval_raw.get("tool_metadata") or {}).get("requires_approval")
        ),
        "allowed": approval_raw.get("allowed") if approval_event else None,
        "reason": approval_raw.get("reason") if approval_event else None,
    }
    started_at = (started or first).get("timestamp")
    ended_at = terminal.get("timestamp") if terminal is not first else None
    summary_parts = [
        "approved" if allowed else None,
        "denied" if denied else None,
        completed.get("summary") if completed else None,
        failed.get("summary") if failed else None,
    ]
    summary = " · ".join(part for part in summary_parts if part) or terminal.get(
        "summary"
    )
    child_run = terminal.get("child_run") or _first_value(events, "child_run")
    activity = _activity_base(
        group,
        "subagent_call" if child_run else "tool_call",
    )
    activity.update(
        {
            "tool_name": tool_name,
            "tool_call_id": raw.get("tool_call_id"),
            "status": status,
            "tone": "red" if failed or denied else "green" if completed else "amber",
            "title": (
                f"Subagent: {tool_name}"
                if child_run
                else f"Tool: {tool_name or 'tool'}"
            ),
            "summary": summary,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": _duration_ms(started_at, ended_at),
            "duration": _duration_label(started_at, ended_at),
            "time": first.get("time"),
            "approval": approval,
            "arguments": _first_value(events, "arguments"),
            "result": terminal.get("result") if completed else None,
            "result_preview": terminal.get("result_preview") if completed else None,
            "error": raw.get("error") if failed else None,
            "child_run": child_run,
        }
    )
    return activity


def _model_activity(group: dict[str, Any]) -> dict[str, Any]:
    events = group["events"]
    first = events[0]
    response_event = _find_event(events, "model.response")
    send_event = _find_event(events, "model.send")
    event = response_event or send_event or first
    raw = event.get("raw") or {}
    response = raw.get("response") or {}
    started_at = (send_event or first).get("timestamp")
    ended_at = response_event.get("timestamp") if response_event else None
    activity = _activity_base(group, "model_call")
    activity.update(
        {
            "model": raw.get("model"),
            "status": "completed" if response_event else "running",
            "tone": "blue" if response_event else "amber",
            "title": "Model response" if response_event else "Model send",
            "summary": event.get("summary"),
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_ms": _duration_ms(started_at, ended_at),
            "duration": _duration_label(started_at, ended_at),
            "time": first.get("time"),
            "input": (send_event.get("raw") or {}).get("input") if send_event else None,
            "response": response if response_event else None,
            "usage": response_event.get("usage") if response_event else None,
        }
    )
    return activity


def _event_activity(group: dict[str, Any]) -> dict[str, Any]:
    event = group["events"][0]
    activity = _activity_base(group, "event")
    activity.update(
        {
            "event_type": event.get("event_type"),
            "status": event.get("event_type", "").replace(".", " "),
            "tone": event.get("tone"),
            "title": event.get("title"),
            "summary": event.get("summary"),
            "started_at": event.get("timestamp"),
            "ended_at": event.get("timestamp"),
            "duration_ms": 0,
            "duration": _duration_label(event.get("timestamp"), event.get("timestamp")),
            "time": event.get("time"),
        }
    )
    return activity


def _find_event(
    events: list[dict[str, Any]], event_type: str
) -> Optional[dict[str, Any]]:
    return next(
        (event for event in events if event.get("event_type") == event_type), None
    )


def _first_value(events: list[dict[str, Any]], key: str) -> Any:
    for event in events:
        value = event.get(key)
        if value is not None:
            return value
    return None


def _child_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    started_at = _first_timestamp(run.get("events", [])) or _first_step_time(
        run.get("steps", [])
    )
    ended_at = _last_timestamp(run.get("events", [])) or _last_step_time(
        run.get("steps", [])
    )
    return {
        "trace_id": run.get("trace_id"),
        "run_name": run.get("run_name"),
        "agent_name": run.get("agent_name"),
        "status": run.get("status"),
        "duration": _duration_label(started_at, ended_at),
        "duration_ms": _duration_ms(started_at, ended_at),
        "usage": _run_usage(run.get("events", [])),
        "final_output_preview": _compact_value(run.get("final_output"), 140),
    }


def _run_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model_call_count": 0,
    }
    seen_any = False
    for event in events:
        if event.get("event_type") != "model.response":
            continue
        usage = event.get("data", {}).get("usage") or {}
        if not isinstance(usage, dict):
            continue
        totals["model_call_count"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
                seen_any = True
    if not seen_any:
        totals["total_tokens"] = 0
    return totals


def _usage_summary(usage: Any) -> str:
    if not isinstance(usage, dict):
        return ""
    total = usage.get("total_tokens")
    if not isinstance(total, int) or isinstance(total, bool):
        return ""
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    parts = [f"{_format_count(total)} tok"]
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        parts.append(f"{_format_count(input_tokens)} in")
        parts.append(f"{_format_count(output_tokens)} out")
    return " · " + " · ".join(parts)


def _format_count(value: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _tool_arguments_summary(tool_name: Optional[str], arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return _compact_value(arguments)
    if tool_name == "run_shell":
        return _compact_value(arguments.get("command"))
    if tool_name == "write_file":
        return _file_content_summary(
            path=arguments.get("path"),
            content=arguments.get("content"),
            action="write",
        )
    if tool_name == "apply_unified_diff":
        return _text_blob_summary("diff", arguments.get("diff"))
    if tool_name == "apply_patch":
        return _text_blob_summary("patch", arguments.get("patch"))
    return _compact_value(arguments)


def _tool_result_summary(tool_name: Optional[str], result_preview: Any) -> str:
    parsed = _parse_preview(result_preview)
    if tool_name == "run_shell" and isinstance(parsed, dict):
        command = parsed.get("command")
        exit_code = parsed.get("exit_code")
        timed_out = parsed.get("timed_out")
        status = "timed out" if timed_out else f"exit {exit_code}"
        return f"{_compact_value(command)} · {status}"
    if tool_name == "write_file":
        return f"wrote {_compact_value(parsed or result_preview)}"
    if tool_name in {"apply_unified_diff", "apply_patch"} and isinstance(parsed, dict):
        files = parsed.get("changed_files") or []
        deleted = parsed.get("deleted_files") or []
        file_count = parsed.get("file_count", len(files) + len(deleted))
        hunk_count = parsed.get("hunk_count", 0)
        names = ", ".join([*files, *deleted][:3])
        more = "..." if len([*files, *deleted]) > 3 else ""
        return f"{file_count} files · {hunk_count} hunks · {names}{more}".strip(" ·")
    return _compact_value(result_preview)


def _file_content_summary(path: Any, content: Any, action: str) -> str:
    path_part = _compact_value(path) or "file"
    if not isinstance(content, str):
        return path_part
    lines = len(content.splitlines())
    return f"{action} {path_part} · {len(content)} chars · {lines} lines"


def _text_blob_summary(label: str, value: Any) -> str:
    if not isinstance(value, str):
        return _compact_value(value)
    lines = len(value.splitlines())
    return f"{label} · {len(value)} chars · {lines} lines"


def _parse_preview(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _first_timestamp(events: list[dict[str, Any]]) -> Optional[str]:
    return events[0].get("timestamp") if events else None


def _last_timestamp(events: list[dict[str, Any]]) -> Optional[str]:
    return events[-1].get("timestamp") if events else None


def _first_step_time(steps: list[dict[str, Any]]) -> Optional[str]:
    return steps[0].get("started_at") if steps else None


def _last_step_time(steps: list[dict[str, Any]]) -> Optional[str]:
    if not steps:
        return None
    return steps[-1].get("ended_at") or steps[-1].get("started_at")


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_label(start: Optional[str], end: Optional[str]) -> str:
    duration = _duration_ms(start, end)
    if duration is None:
        return "-"
    seconds = duration / 1000
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {int(remainder)}s"


def _duration_ms(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)
    if not start_dt or not end_dt:
        return None
    return max((end_dt - start_dt).total_seconds() * 1000, 0)


def _short_time(value: Optional[str]) -> str:
    parsed = _parse_dt(value)
    if not parsed:
        return value or ""
    return parsed.strftime("%H:%M:%S")


def _artifact_payload(artifact) -> dict[str, Any]:
    payload = {
        "artifact": artifact.ref.to_dict(),
        "created_at": artifact.created_at,
    }
    media_type = artifact.ref.media_type or "application/octet-stream"
    if media_type.startswith("text/") or "charset=utf-8" in media_type:
        payload["text"] = artifact.text()
    else:
        payload["data_base64"] = base64.b64encode(artifact.data).decode("ascii")
    return payload


class ViewerServer:
    def __init__(
        self,
        trace_file: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        ui_dist: str | Path | None = None,
        artifact_store: Any | None = None,
        trace_store: TraceStore | None = None,
    ):
        self.trace_file = Path(trace_file) if trace_file else None
        self.host = host
        self.port = port
        self.ui_dist = Path(ui_dist) if ui_dist else DEFAULT_UI_DIST
        self.artifact_store = artifact_store
        self.state = ViewerTraceState(self.trace_file, store=trace_store)
        self.trace_sink = self.state.trace_sink()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        if self._server:
            host, port = self._server.server_address
            return f"http://{host}:{port}"
        return f"http://{self.host}:{self.port}"

    def start(self) -> "ViewerServer":
        state = self.state
        ui_dist = self.ui_dist
        artifact_store = self.artifact_store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/api/runs":
                    self._send_json({"runs": state.list_runs()})
                    return
                if parsed.path.startswith("/api/runs/"):
                    trace_id = parsed.path.removeprefix("/api/runs/")
                    run = state.get_run(trace_id)
                    if run is None:
                        self.send_error(404)
                        return
                    self._send_json({"run": run})
                    return
                if parsed.path.startswith("/api/events/"):
                    trace_id = parsed.path.removeprefix("/api/events/")
                    self._send_json({"events": state.list_events(trace_id)})
                    return
                if parsed.path.startswith("/api/artifacts/"):
                    if artifact_store is None:
                        self.send_error(404)
                        return
                    artifact_id = parsed.path.removeprefix("/api/artifacts/")
                    try:
                        artifact = artifact_store.get(artifact_id)
                    except KeyError:
                        self.send_error(404)
                        return
                    self._send_json(_artifact_payload(artifact))
                    return
                if parsed.path in {"/", "/index.html"}:
                    index = ui_dist / "index.html"
                    if index.exists():
                        self._send_file(index)
                    else:
                        self._send_html(VIEWER_HTML)
                    return
                asset = self._asset_path(parsed.path)
                if asset is not None:
                    self._send_file(asset)
                    return
                self.send_error(404)

            def do_POST(self):
                parsed = urlparse(self.path)
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                if parsed.path == "/api/import-jsonl":
                    count = state.import_jsonl(body)
                    self._send_json({"imported": count, "runs": state.list_runs()})
                    return
                if parsed.path == "/api/events":
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError:
                        self.send_error(400, "Invalid JSON event")
                        return
                    state.add_event(event)
                    self._send_json({"ok": True})
                    return
                self.send_error(404)

            def log_message(self, format, *args):
                return

            def _send_json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _asset_path(self, path):
                if not ui_dist.exists():
                    return None
                relative = path.lstrip("/") or "index.html"
                candidate = (ui_dist / relative).resolve()
                try:
                    candidate.relative_to(ui_dist.resolve())
                except ValueError:
                    return None
                if candidate.exists() and candidate.is_file():
                    return candidate
                return None

            def _send_file(self, path):
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", self._content_type(path))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _content_type(self, path):
                suffix = path.suffix
                if suffix == ".html":
                    return "text/html; charset=utf-8"
                if suffix == ".js":
                    return "text/javascript; charset=utf-8"
                if suffix == ".css":
                    return "text/css; charset=utf-8"
                if suffix == ".svg":
                    return "image/svg+xml"
                return "application/octet-stream"

        if self.trace_file:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="aisuite-runs-viewer",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None


def start_viewer(
    trace_file: str | Path | None = ".aisuite/runs.jsonl",
    host: str = "127.0.0.1",
    port: int = 8765,
    ui_dist: str | Path | None = None,
    artifact_store: Any | None = None,
    artifact_root: str | Path | None = None,
    trace_store: TraceStore | None = None,
) -> ViewerServer:
    if artifact_store is None and artifact_root is not None:
        from ..agents.artifact_store import FileArtifactStore

        artifact_store = FileArtifactStore(artifact_root)
    return ViewerServer(
        trace_file=trace_file,
        host=host,
        port=port,
        ui_dist=ui_dist,
        artifact_store=artifact_store,
        trace_store=trace_store,
    ).start()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Start the aisuite runs viewer.")
    parser.add_argument("--trace-file", default=".aisuite/runs.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-dist", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create a deterministic demo trace before starting the viewer.",
    )
    parser.add_argument("--demo-cwd", default=".")
    args = parser.parse_args(argv)

    if args.host != "127.0.0.1":
        print("Warning: non-localhost hosts may expose trace data.")

    if args.demo:
        from examples.cli.create_demo_trace import create_demo_trace

        create_demo_trace(
            trace_file=Path(args.trace_file),
            cwd=Path(args.demo_cwd).expanduser().resolve(),
        )

    viewer = start_viewer(
        args.trace_file,
        host=args.host,
        port=args.port,
        ui_dist=args.ui_dist,
        artifact_root=args.artifact_root,
    )
    print(f"aisuite runs viewer: {viewer.url}")
    print(f"Watching {args.trace_file}")
    print(f"Live event endpoint: {viewer.url}/api/events")
    print("Press q then Enter to stop.")
    try:
        while input().strip().lower() != "q":
            print("Press q then Enter to stop.")
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()
