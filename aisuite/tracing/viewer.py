from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from .store import JsonlTraceStore


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>aisuite runs</title>
  <style>
    body { margin: 0; font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18212f; background: #f4f6f8; }
    header { height: 58px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; background: rgba(255,255,255,0.92); border-bottom: 1px solid #dce2e8; backdrop-filter: blur(12px); }
    main { display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 58px); }
    aside { border-right: 1px solid #dce2e8; background: white; overflow: auto; }
    section { padding: 22px; overflow: auto; }
    .brand { font-weight: 800; }
    .top-meta { color: #637083; font-size: 13px; }
    .run { padding: 12px 14px; border: 1px solid transparent; border-radius: 8px; cursor: pointer; margin: 6px 10px; }
    .run:hover { background: #f9fafb; border-color: #edf1f5; }
    .run.active { background: #eaf1ff; border-color: #bfd1ff; }
    .group { padding: 16px 18px 6px; color: #637083; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; display: flex; justify-content: space-between; }
    .name { font-weight: 750; }
    .meta { color: #637083; font-size: 12px; margin-top: 4px; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #edf1f5; color: #354154; font-size: 12px; font-weight: 650; margin: 5px 4px 0 0; }
    .pill.green { background: #e7f5f0; color: #12715b; }
    .pill.blue { background: #eaf1ff; color: #174ea6; }
    .panel { background: white; border: 1px solid #dce2e8; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 10px 24px rgba(24, 33, 47, 0.06); }
    .panel h2 { margin: 0; padding: 14px 16px; font-size: 15px; border-bottom: 1px solid #edf1f5; }
    .panel .content { padding: 16px; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
    .metric { border: 1px solid #edf1f5; border-radius: 8px; padding: 11px; background: #f9fafb; }
    .metric .label { color: #637083; font-size: 12px; font-weight: 700; }
    .metric .value { font-weight: 850; font-size: 18px; margin-top: 4px; }
    .output { white-space: pre-wrap; word-break: break-word; line-height: 1.48; color: #263244; }
    .tabs { display: flex; gap: 6px; padding: 0 12px; border-bottom: 1px solid #edf1f5; }
    .tab { border: 0; border-bottom: 2px solid transparent; background: transparent; padding: 13px 8px 11px; color: #637083; font: inherit; font-weight: 750; cursor: pointer; }
    .tab.active { color: #276ef1; border-bottom-color: #276ef1; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .message { border: 1px solid #edf1f5; border-radius: 8px; margin-bottom: 10px; overflow: hidden; background: #f9fafb; }
    .message .role { color: #637083; font-size: 12px; font-weight: 800; padding: 8px 10px; text-transform: uppercase; letter-spacing: 0.04em; }
    .message .body { padding: 0 10px 10px; white-space: pre-wrap; word-break: break-word; line-height: 1.48; }
    .event { display: grid; grid-template-columns: 12px 1fr auto; gap: 12px; padding: 12px 0; border-bottom: 1px solid #edf1f5; }
    .event:last-child { border-bottom: 0; }
    .event-dot { width: 10px; height: 10px; border-radius: 999px; background: #276ef1; margin-top: 5px; }
    .event-type { font-weight: 750; }
    .event-summary { color: #637083; font-size: 13px; margin-top: 3px; }
    .event details { margin-top: 8px; }
    .event summary { cursor: pointer; color: #276ef1; font-size: 12px; font-weight: 750; }
    .step { padding: 9px 0; border-bottom: 1px solid #edf0f5; }
    .step:last-child { border-bottom: 0; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0b1020; color: #d5e0ff; padding: 12px; border-radius: 6px; overflow: auto; }
    .empty { color: #607086; padding: 20px; }
    @media (max-width: 800px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: 35vh; border-right: 0; border-bottom: 1px solid #d8dee8; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header><div><span class="brand">aisuite runs</span><span id="count" class="top-meta" style="margin-left: 10px;"></span></div><span class="top-meta">local viewer</span></header>
  <main>
    <aside id="runs"></aside>
    <section id="detail"><div class="empty">No runs yet.</div></section>
  </main>
  <script>
    let runs = [];
    let selectedTraceId = null;
    let selectedTab = "overview";

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
      const groups = {};
      for (const run of runs) {
        const key = run.group_id || "ungrouped";
        if (!groups[key]) groups[key] = [];
        groups[key].push(run);
      }
      list.innerHTML = Object.entries(groups).map(([group, groupRuns]) => `
        <div class="group"><span>${escapeHtml(group)}</span><span>${groupRuns.length} run${groupRuns.length === 1 ? "" : "s"}</span></div>
        ${groupRuns.map(run => {
        const active = run.trace_id === selectedTraceId ? " active" : "";
        const title = run.run_name || run.trace_id || "run";
        const tags = (run.tags || []).map(tag => `<span class="pill">${escapeHtml(tag)}</span>`).join("");
        const counts = `${run.message_count ?? (run.messages || []).length} messages · ${run.step_count ?? (run.steps || []).length} steps`;
        return `<div class="run${active}" onclick="selectRun('${escapeHtml(run.trace_id)}')">
          <div class="name">${escapeHtml(title)}</div>
          <div class="meta">${escapeHtml(run.agent_name)} · ${escapeHtml(run.status)}</div>
          <div class="meta">${escapeHtml(counts)}</div>
          ${run.parent_run_id ? `<div class="meta">parent: ${escapeHtml(run.parent_run_id.slice(0, 12))}</div>` : ""}
          <div class="meta">${escapeHtml(run.group_id || "")}</div>
          <div>${tags}</div>
        </div>`;
        }).join("")}
      `).join("");
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
      const metadata = Object.entries(run.metadata || {}).map(([key, value]) =>
        `<span class="pill">${escapeHtml(key)}=${escapeHtml(value)}</span>`
      ).join("");
      const messages = (run.messages || []).map(message => {
        const role = message.role || "message";
        return `<div class="message">
          <div class="role">${escapeHtml(role)}</div>
          <div class="body">${escapeHtml(messageContent(message))}</div>
        </div>`;
      }).join("") || '<div class="empty">No messages.</div>';
      const steps = (run.steps || []).map(step => {
        const data = step.data || {};
        const bits = [];
        if (data.allowed !== undefined) bits.push(`allowed=${data.allowed}`);
        if (data.status) bits.push(`status=${data.status}`);
        if (data.reason) bits.push(`reason=${data.reason}`);
        return `<div class="step">
          <strong>${escapeHtml(step.type)}</strong> ${escapeHtml(step.name || "")}
          <div class="meta">${escapeHtml(bits.join(" · "))}</div>
        </div>`;
      }).join("") || '<div class="empty">No steps.</div>';
      const events = (run.events || []).map(event => {
        const detail = event.data ? JSON.stringify(event.data, null, 2) : "";
        return `<div class="event">
          <div class="event-dot"></div>
          <div>
            <div class="event-type">${escapeHtml(event.event_type)}</div>
            <div class="event-summary">${escapeHtml(event.agent_name || run.agent_name || "")} ${event.run_name ? "· " + escapeHtml(event.run_name) : ""}</div>
            <details><summary>Details</summary><pre>${escapeHtml(detail)}</pre></details>
          </div>
          <div class="meta">${escapeHtml(event.timestamp || "")}</div>
        </div>`;
      }).join("") || '<div class="empty">No events.</div>';
      const finalOutput = run.final_output === undefined || run.final_output === null
        ? '<div class="empty">No final output.</div>'
        : `<div class="output">${escapeHtml(displayValue(run.final_output))}</div>`;
      const active = tab => tab === selectedTab ? " active" : "";
      detail.innerHTML = `
        <div class="panel"><h2>Run</h2><div class="content">
          <div><strong>${escapeHtml(run.run_name || run.trace_id)}</strong></div>
          <div class="meta">trace: ${escapeHtml(run.trace_id)}</div>
          <div class="meta">agent: ${escapeHtml(run.agent_name)} · status: ${escapeHtml(run.status)}</div>
          <div class="meta">group: ${escapeHtml(run.group_id || "-")}</div>
          <div class="meta">parent: ${escapeHtml(run.parent_run_id || "-")}</div>
          <div>${metadata}</div>
          <div class="summary">
            <div class="metric"><div class="label">Messages</div><div class="value">${escapeHtml(run.message_count ?? (run.messages || []).length)}</div></div>
            <div class="metric"><div class="label">Steps</div><div class="value">${escapeHtml(run.step_count ?? (run.steps || []).length)}</div></div>
            <div class="metric"><div class="label">Events</div><div class="value">${escapeHtml(run.event_count ?? (run.events || []).length)}</div></div>
            <div class="metric"><div class="label">Trace</div><div class="value">${escapeHtml((run.trace_id || "").slice(0, 12))}</div></div>
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
              <h2 style="padding: 18px 0 10px; border: 0;">Steps</h2>
              ${steps}
            </div>
            <div class="tab-panel${active("transcript")}">${messages}</div>
            <div class="tab-panel${active("events")}">${events}</div>
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
    return JsonlTraceStore(trace_file).list_runs()


class ViewerServer:
    def __init__(
        self,
        trace_file: str | Path,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.trace_file = Path(trace_file)
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        if self._server:
            host, port = self._server.server_address
            return f"http://{host}:{port}"
        return f"http://{self.host}:{self.port}"

    def start(self) -> "ViewerServer":
        trace_file = self.trace_file

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/api/runs":
                    self._send_json({"runs": read_trace_file(trace_file)})
                    return
                if parsed.path in {"/", "/index.html"}:
                    self._send_html(VIEWER_HTML)
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
    trace_file: str | Path = ".aisuite/runs.jsonl",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ViewerServer:
    return ViewerServer(trace_file=trace_file, host=host, port=port).start()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Start the aisuite runs viewer.")
    parser.add_argument("--trace-file", default=".aisuite/runs.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.host != "127.0.0.1":
        print("Warning: non-localhost hosts may expose trace data.")

    viewer = start_viewer(args.trace_file, host=args.host, port=args.port)
    print(f"aisuite runs viewer: {viewer.url}")
    print(f"Watching {args.trace_file}")
    print("Press q then Enter to stop.")
    try:
        while input().strip().lower() != "q":
            print("Press q then Enter to stop.")
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()
