import {
  Activity,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  CircleAlert,
  Clipboard,
  Clock3,
  FileJson,
  Filter,
  Moon,
  Search,
  Sun,
  TerminalSquare,
  Upload,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

function cx(...values) {
  return values.filter(Boolean).join(" ");
}

function compact(value, max = 180) {
  if (value == null) return "";
  const text =
    typeof value === "string"
      ? value
      : value?.type === "text_preview"
        ? value.preview
        : JSON.stringify(value, null, 2);
  const flat = text.replace(/\n/g, " ");
  return flat.length > max ? `${flat.slice(0, max - 3)}...` : flat;
}

function formatCount(value) {
  if (typeof value !== "number") return "-";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function formatDurationMs(value) {
  if (typeof value !== "number") return "-";
  if (value < 1000) return `${Math.round(value)} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(1)} s`;
  const minutes = Math.floor(value / 60_000);
  const seconds = Math.floor((value % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function usageLabel(usage) {
  if (!usage || typeof usage.total_tokens !== "number") return "";
  const parts = [`${formatCount(usage.total_tokens)} tok`];
  if (
    typeof usage.input_tokens === "number" &&
    typeof usage.output_tokens === "number"
  ) {
    parts.push(`${formatCount(usage.input_tokens)} in`);
    parts.push(`${formatCount(usage.output_tokens)} out`);
  }
  return parts.join(" · ");
}

function textStats(value) {
  const text = typeof value === "string" ? value : value?.type === "text_preview" ? value.preview || "" : "";
  const lines = text
    ? text.split(/\r?\n/).filter((line, index, all) => index < all.length - 1 || line !== "").length
    : 0;
  return {
    chars: text.length,
    lines,
    preview: text.length > 1200 ? `${text.slice(0, 1200)}...` : text,
  };
}

function toneClasses(tone) {
  if (tone === "green") return "status-pill status-green";
  if (tone === "red") return "status-pill status-red";
  if (tone === "amber") return "status-pill status-amber";
  return "status-pill status-blue";
}

function Dot({ tone }) {
  return (
    <span
      className={cx(
        "mt-1 h-2.5 w-2.5 rounded-full ring-4",
        tone === "green" && "bg-emerald-500 ring-emerald-500/15",
        tone === "red" && "bg-rose-500 ring-rose-500/15",
        tone === "amber" && "bg-amber-500 ring-amber-500/15",
        (!tone || tone === "blue") && "bg-cyan-500 ring-cyan-500/15",
      )}
    />
  );
}

function Pill({ children, tone = "blue" }) {
  return (
    <span
      className={cx(
        "inline-flex h-6 items-center rounded-full px-2.5 text-xs font-semibold ring-1",
        toneClasses(tone),
      )}
    >
      {children}
    </span>
  );
}

async function fetchRuns() {
  const response = await fetch(`${API_BASE}/api/runs`);
  if (!response.ok) throw new Error("Unable to fetch runs");
  return response.json();
}

async function fetchRun(traceId) {
  const response = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(traceId)}`);
  if (!response.ok) throw new Error("Unable to fetch run detail");
  return response.json();
}

async function importJsonl(text) {
  const response = await fetch(`${API_BASE}/api/import-jsonl`, {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: text,
  });
  if (!response.ok) throw new Error("Unable to import JSONL");
  return response.json();
}

function isArtifactValue(value) {
  return (
    value &&
    typeof value === "object" &&
    value.type === "artifact_ref" &&
    value.artifact_ref &&
    typeof value.artifact_ref.artifact_id === "string"
  );
}

async function fetchArtifact(artifactId) {
  const response = await fetch(`${API_BASE}/api/artifacts/${artifactId}`);
  if (!response.ok) throw new Error("Unable to fetch artifact");
  return response.json();
}

function runGroups(runs) {
  const groups = new Map();
  for (const run of runs) {
    const key = run.group_id || "ungrouped";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(run);
  }
  return Array.from(groups.entries()).map(([groupId, groupRuns]) => {
    const ordered = [...groupRuns].sort((left, right) => {
      const leftChild = left.parent_run_id ? 1 : 0;
      const rightChild = right.parent_run_id ? 1 : 0;
      if (leftChild !== rightChild) return leftChild - rightChild;
      return (left.display?.started_at || "").localeCompare(right.display?.started_at || "");
    });
    return [groupId, ordered];
  });
}

function ArtifactChip({ count }) {
  if (!count) return null;
  return <span className="timeline-status">{count} artifact{count === 1 ? "" : "s"}</span>;
}

function RunList({ runs, selectedId, onSelect }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return runs.filter((run) => {
      if (status !== "all" && run.status !== status) return false;
      if (!needle) return true;
      const display = run.display || {};
      const text = [
        run.trace_id,
        run.run_name,
        run.agent_name,
        run.group_id,
        run.status,
        display.latest_activity,
        ...(display.tools || []),
      ]
        .join(" ")
        .toLowerCase();
      return text.includes(needle);
    });
  }, [runs, query, status]);

  return (
    <aside className="sidebar-panel flex min-h-[calc(100vh-68px)] w-[360px] shrink-0 flex-col">
      <div className="sticky top-0 z-10 border-b border-[var(--line)] bg-[var(--panel-translucent)] p-3 backdrop-blur-xl">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <div className="text-xs font-semibold uppercase text-[var(--muted)]">Runs</div>
            <div className="text-xs text-[var(--subtle)]">
              {filtered.length} of {runs.length}
            </div>
          </div>
        </div>
        <div className="grid grid-cols-[1fr_126px] gap-2">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-[var(--muted)]" />
            <input
              className="control h-9 w-full pl-9 pr-3"
              placeholder="Search runs"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <label className="relative">
            <Filter className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-[var(--muted)]" />
            <select
              className="control h-9 w-full pl-9 pr-3"
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="all">all</option>
              <option value="completed">completed</option>
              <option value="running">running</option>
              <option value="failed">failed</option>
            </select>
          </label>
        </div>
      </div>
      <div className="flex-1 overflow-auto px-2 py-3">
        {!filtered.length ? (
          <div className="p-5 text-sm text-[var(--muted)]">No matching runs.</div>
        ) : (
          runGroups(filtered).map(([groupId, groupRuns]) => (
            <div key={groupId} className="mb-3">
              <div className="px-3 pb-1 pt-2 text-[11px] font-bold uppercase tracking-[0.12em] text-[var(--subtle)]">
                {groupId}
              </div>
              {groupRuns.map((run) => {
                const display = run.display || {};
                const isChild = !!run.parent_run_id;
                return (
                  <button
                    key={run.trace_id}
                    type="button"
                    onClick={() => onSelect(run.trace_id)}
                    className={cx(
                      "run-row w-full px-3 py-2.5 text-left transition",
                      isChild && "ml-4 w-[calc(100%-1rem)]",
                      selectedId === run.trace_id
                        ? "run-row-selected"
                        : "run-row-idle",
                    )}
                  >
                    <div className="grid grid-cols-[12px_1fr_auto] items-center gap-3">
                      <span className={cx("run-status-dot", `run-status-${display.status_tone || "blue"}`)} />
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold">
                          {display.title || run.run_name || run.trace_id}
                        </div>
                        <div className="mt-0.5 truncate text-xs text-[var(--muted)]">
                          {isChild ? `subagent${display.parent_title ? ` of ${display.parent_title}` : ""}` : run.agent_name || display.subtitle || "agent"}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {display.child_count ? <span className="timeline-status">{display.child_count} child</span> : null}
                          <ArtifactChip count={display.artifact_count} />
                        </div>
                      </div>
                      <div className="text-xs text-[var(--subtle)]">{display.duration || "-"}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          ))
        )}
      </div>
    </aside>
  );
}

function Metric({ label, value, icon: Icon }) {
  return (
    <div className="metric-card p-3">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase text-[var(--muted)]">
        {Icon ? <Icon className="h-3.5 w-3.5" /> : null}
        {label}
      </div>
      <div className="mt-1 truncate text-xl font-bold text-[var(--ink)]">{value}</div>
    </div>
  );
}

function KeyValues({ value }) {
  if (!value || typeof value !== "object") return null;
  return (
    <div className="kv-panel mt-3 divide-y">
      {Object.entries(value).map(([key, item]) => (
        <div key={key} className="grid grid-cols-[150px_1fr] gap-3 px-3 py-2">
          <div className="text-xs font-semibold text-[var(--muted)]">{key}</div>
          <pre className="whitespace-pre-wrap break-words text-xs text-[var(--ink)]">
            {typeof item === "string" ? item : JSON.stringify(item, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  );
}

function ArgumentPanel({ toolName, value }) {
  if (!value || typeof value !== "object") return null;
  if (toolName === "write_file" && (typeof value.content === "string" || isArtifactValue(value.content))) {
    const stats = textStats(value.content);
    return (
      <div className="mt-3 space-y-3">
        <KeyValues
          value={{
            path: value.path,
            overwrite: value.overwrite,
            chars: isArtifactValue(value.content) ? value.content.artifact_ref.size_bytes : stats.chars,
            lines: stats.lines,
          }}
        />
        <CodeBlock title="content preview" value={isArtifactValue(value.content) ? value.content : stats.preview} />
      </div>
    );
  }
  if (toolName === "apply_unified_diff" && (typeof value.diff === "string" || isArtifactValue(value.diff))) {
    const stats = textStats(value.diff);
    return (
      <div className="mt-3 space-y-3">
        <KeyValues value={{ chars: isArtifactValue(value.diff) ? value.diff.artifact_ref.size_bytes : stats.chars, lines: stats.lines }} />
        <CodeBlock title="diff preview" value={isArtifactValue(value.diff) ? value.diff : stats.preview} />
      </div>
    );
  }
  if (toolName === "apply_patch" && (typeof value.patch === "string" || isArtifactValue(value.patch))) {
    const stats = textStats(value.patch);
    return (
      <div className="mt-3 space-y-3">
        <KeyValues value={{ chars: isArtifactValue(value.patch) ? value.patch.artifact_ref.size_bytes : stats.chars, lines: stats.lines }} />
        <CodeBlock title="patch preview" value={isArtifactValue(value.patch) ? value.patch : stats.preview} />
      </div>
    );
  }
  if (toolName === "run_shell") {
    return (
      <KeyValues
        value={{
          command: value.command,
          timeout_seconds: value.timeout_seconds,
        }}
      />
    );
  }
  return <KeyValues value={value} />;
}

function ResultPanel({ item }) {
  const result = item.result;
  if (!result && !item.result_preview) return null;
  if (
    result &&
    typeof result === "object" &&
    ("command" in result || "stdout" in result || "stderr" in result)
  ) {
    return (
      <div className="mt-3 space-y-3">
        <KeyValues
          value={{
            command: result.command,
            exit_code: result.exit_code,
            timed_out: result.timed_out,
          }}
        />
        {result.stdout ? <CodeBlock title="stdout" value={result.stdout} /> : null}
        {result.stderr ? <CodeBlock title="stderr" value={result.stderr} /> : null}
      </div>
    );
  }
  if (result && typeof result === "object" && "content" in result) {
    const { content, ...rest } = result;
    return (
      <div className="mt-3 space-y-3">
        <KeyValues value={rest} />
        <CodeBlock title="content" value={content} />
      </div>
    );
  }
  if (
    ["apply_unified_diff", "apply_patch"].includes(item.tool_name) &&
    result &&
    typeof result === "object"
  ) {
    return <KeyValues value={result} />;
  }
  return <CodeBlock title="result" value={item.result_preview || compact(result)} />;
}

function ArtifactBlock({ title, value }) {
  const [content, setContent] = useState(null);
  const [status, setStatus] = useState("");
  const ref = value.artifact_ref || {};

  async function load() {
    setStatus("Loading...");
    try {
      const payload = await fetchArtifact(ref.artifact_id);
      setContent(payload.text ?? payload.data_base64 ?? "");
      setStatus("");
    } catch (error) {
      setStatus(error.message);
    }
  }

  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 text-xs font-semibold text-[var(--muted)]">
        <span>{title}</span>
        <span className="font-mono text-[11px] text-[var(--subtle)]">
          {formatCount(ref.size_bytes)} bytes · {ref.artifact_id}
        </span>
      </div>
      <pre className="code-block max-h-64 overflow-auto p-3 text-xs leading-relaxed">
        {content ?? value.preview ?? "Artifact preview unavailable."}
      </pre>
      <div className="mt-2 flex items-center gap-3">
        <button type="button" className="action-button h-8 px-3 text-xs font-semibold" onClick={load}>
          Load full artifact
        </button>
        {status ? <span className="text-xs text-[var(--muted)]">{status}</span> : null}
      </div>
    </div>
  );
}

function CodeBlock({ title, value }) {
  if (isArtifactValue(value)) return <ArtifactBlock title={title} value={value} />;
  return (
    <div>
      <div className="mb-1 text-xs font-semibold text-[var(--muted)]">{title}</div>
      <pre className="code-block max-h-64 overflow-auto p-3 text-xs leading-relaxed">
        {value}
      </pre>
    </div>
  );
}

function isToolEvent(item) {
  return item.event_type?.startsWith("tool.");
}

function toolGroupKey(item, fallback) {
  const raw = item.raw || {};
  const callId = raw.tool_call_id || raw.tool_call?.id || item.tool_call_id;
  if (callId) return `tool:${callId}`;
  if (item.tool_name) return `tool:${item.tool_name}:${fallback}`;
  return null;
}

function activityToGroup(activity) {
  const type =
    activity.type === "tool_call" || activity.type === "subagent_call"
      ? "tool"
      : activity.type === "model_call"
        ? "model"
        : "event";
  return {
    id: activity.id,
    type,
    events: activity.events || [],
    activity,
  };
}

function buildTimelineGroups(timeline, activities = []) {
  if (activities.length) return activities.map(activityToGroup);

  const groups = [];
  const toolGroups = new Map();
  const toolOrder = [];

  timeline.forEach((item, index) => {
    if (isToolEvent(item)) {
      const key = toolGroupKey(item, index);
      if (key) {
        if (!toolGroups.has(key)) {
          toolGroups.set(key, {
            id: key,
            type: "tool",
            events: [],
          });
          toolOrder.push(key);
        }
        toolGroups.get(key).events.push(item);
        return;
      }
    }

    groups.push({
      id: `${item.event_type}-${index}`,
      type: "event",
      events: [item],
    });
  });

  for (const key of toolOrder) {
    groups.push(toolGroups.get(key));
  }

  return groups.sort((left, right) => {
    const a = left.events[0]?.timestamp || "";
    const b = right.events[0]?.timestamp || "";
    return a.localeCompare(b);
  }).reduce((acc, group, index, sorted) => {
    const item = group.events[0];
    if (
      group.type === "event" &&
      item?.event_type === "model.send" &&
      sorted[index + 1]?.type === "event" &&
      sorted[index + 1].events[0]?.event_type === "model.response"
    ) {
      acc.push({
        id: `model-${item.timestamp || index}`,
        type: "model",
        events: [item, sorted[index + 1].events[0]],
      });
      sorted[index + 1].merged = true;
      return acc;
    }
    if (!group.merged) acc.push(group);
    return acc;
  }, []);
}

function groupFilterType(group) {
  if (group.type === "model") return "model";
  if (group.type === "tool") {
    const activity = group.activity;
    const childRun = activity?.child_run || group.events?.find((event) => event.child_run)?.child_run;
    if (activity?.type === "subagent_call" || childRun) return "subagent";
    if ((activity?.approval && activity.approval.reason !== "low risk") || group.events?.some((event) => ["tool.allowed", "tool.denied"].includes(event.event_type))) {
      return "approval";
    }
    return "tool";
  }
  if (group.events?.some((event) => event.tone === "red" || event.event_type?.includes("failed") || event.event_type?.includes("error"))) return "error";
  return "event";
}

function groupMatchesFilter(group, filter) {
  if (filter === "all") return true;
  if (filter === "error") {
    const summary = summarizeGroup(group);
    return summary.tone === "red" || group.events?.some((event) => event.tone === "red" || event.event_type?.includes("failed") || event.event_type?.includes("error"));
  }
  return groupFilterType(group) === filter;
}

const TIMELINE_FILTERS = [
  { id: "all", label: "All" },
  { id: "model", label: "Model" },
  { id: "tool", label: "Tools" },
  { id: "approval", label: "Approvals" },
  { id: "error", label: "Errors" },
  { id: "subagent", label: "Subagents" },
];

function summarizeGroup(group) {
  if (group.activity) {
    return {
      tone: group.activity.tone,
      time: group.activity.time,
      title: group.activity.title,
      summary: group.activity.summary,
      status: group.activity.status,
      usage: group.activity.usage,
      childRun: group.activity.child_run,
      duration: group.activity.duration,
      durationMs: group.activity.duration_ms,
      artifactCount: group.activity.artifact_count,
    };
  }
  const events = group.events;
  const first = events[0] || {};
  const last = events[events.length - 1] || first;
  if (group.type === "tool") {
    const completed = events.find((event) => event.event_type === "tool.completed");
    const failed = events.find((event) => event.event_type === "tool.failed");
    const allowed = events.find((event) => event.event_type === "tool.allowed");
    const denied = events.find((event) => event.event_type === "tool.denied");
    const terminal = failed || completed || denied || last;
    const status = failed ? "failed" : denied ? "denied" : completed ? "completed" : "pending";
    const summaryParts = [
      allowed ? "approved" : null,
      denied ? "denied" : null,
      completed?.summary || failed?.summary || terminal.summary,
    ].filter(Boolean);
    return {
      tone: failed || denied ? "red" : completed ? "green" : "amber",
      time: first.time,
      title: `Tool: ${first.tool_name || terminal.tool_name || "tool"}`,
      summary: summaryParts.join(" · ") || status,
      status,
      childRun: terminal.child_run || events.find((event) => event.child_run)?.child_run,
      artifactCount: events.reduce((total, event) => total + (event.artifact_count || 0), 0),
    };
  }
  if (group.type === "model") {
    const response = events.find((event) => event.event_type === "model.response");
    return {
      tone: response ? "blue" : "amber",
      time: first.time,
      title: "Model response",
      summary: response?.summary || first.summary || "Model call",
      status: response?.raw?.response?.kind || (response ? "response" : "send"),
      usage: response?.usage,
      artifactCount: events.reduce((total, event) => total + (event.artifact_count || 0), 0),
    };
  }
  return {
    tone: first.tone,
    time: first.time,
    title: first.title,
    summary: first.summary,
    status: first.event_type?.replace(".", " "),
    artifactCount: first.artifact_count || 0,
  };
}

function TimelineDetails({ group, onSelectRun }) {
  const events = group.events;
  const activity = group.activity;
  const args = activity?.arguments || events.find((event) => event.arguments)?.arguments;
  const toolName =
    activity?.tool_name || events.find((event) => event.tool_name)?.tool_name;
  const childRun =
    activity?.child_run || events.find((event) => event.child_run)?.child_run;
  const results =
    activity?.result || activity?.result_preview
      ? [
          {
            event_type: activity.type,
            tool_name: activity.tool_name,
            result: activity.result,
            result_preview: activity.result_preview,
          },
        ]
      : events.filter((event) => event.result || event.result_preview);
  return (
    <div className="timeline-details">
      {childRun ? (
        <div className="mb-3 rounded-xl border border-[var(--line)] bg-[var(--panel-muted)] p-3">
          <div className="text-xs font-semibold uppercase text-[var(--muted)]">
            subagent run
          </div>
          <div className="mt-1 text-sm font-semibold">
            {childRun.agent_name || childRun.run_name || "subagent"}
          </div>
          <div className="mt-1 text-xs text-[var(--muted)]">
            {childRun.status || "running"} · {childRun.duration || "-"}
          </div>
          {childRun.final_output_preview ? (
            <div className="mt-2 text-sm text-[var(--muted)]">
              {childRun.final_output_preview}
            </div>
          ) : null}
          {childRun.trace_id ? (
            <button
              type="button"
              className="action-button mt-3 h-8 px-3 text-xs font-semibold"
              onClick={() => onSelectRun?.(childRun.trace_id)}
            >
              Open child run
            </button>
          ) : null}
        </div>
      ) : null}
      {args ? (
        <div>
          <div className="text-xs font-semibold text-[var(--muted)]">arguments</div>
          <ArgumentPanel toolName={toolName} value={args} />
        </div>
      ) : null}
      {results.map((event, index) => (
        <ResultPanel key={`${event.event_type}-${index}`} item={event} />
      ))}
      <div>
        <div className="mb-1 mt-3 text-xs font-semibold text-[var(--muted)]">events</div>
        <div className="event-list">
          {events.map((event, index) => (
            <div key={`${event.event_type}-${index}`} className="event-list-row">
              <span className={cx("run-status-dot", `run-status-${event.tone || "blue"}`)} />
              <span className="font-mono text-[11px] text-[var(--subtle)]">{event.time}</span>
              <span className="text-xs text-[var(--muted)]">{event.event_type}</span>
              <span className="truncate text-xs">{event.summary}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Timeline({ run, onSelectRun, filter = "all", onFilterChange }) {
  const timeline = run.display?.timeline || [];
  const activities = run.display?.activities || [];
  const groups = useMemo(
    () => buildTimelineGroups(timeline, activities),
    [timeline, activities],
  );
  const [expanded, setExpanded] = useState({});

  function toggle(id) {
    setExpanded((current) => ({ ...current, [id]: !current[id] }));
  }

  const visibleGroups = groups.filter((group) => groupMatchesFilter(group, filter));

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {TIMELINE_FILTERS.map((item) => {
          const count = item.id === "all" ? groups.length : groups.filter((group) => groupMatchesFilter(group, item.id)).length;
          return (
            <button
              key={item.id}
              type="button"
              className={cx("filter-chip", filter === item.id && "filter-chip-active")}
              onClick={() => onFilterChange?.(item.id)}
            >
              {item.label} <span>{count}</span>
            </button>
          );
        })}
      </div>
      {!visibleGroups.length ? (
        <div className="output-panel p-3 text-sm text-[var(--muted)]">No timeline events match this filter.</div>
      ) : null}
      <div className="timeline-spine">
      {visibleGroups.map((group) => {
        const summary = summarizeGroup(group);
        const isOpen = !!expanded[group.id];
        const Chevron = isOpen ? ChevronDown : ChevronRight;
        return (
          <div key={group.id} className="timeline-entry">
            <div className="timeline-marker">
              <Dot tone={summary.tone} />
            </div>
            <div className="min-w-0 flex-1">
              <button
                type="button"
                className="timeline-row w-full"
                onClick={() => toggle(group.id)}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <div className="truncate font-semibold">{summary.title}</div>
                    <span className="timeline-status">{summary.status}</span>
                    {summary.duration && summary.duration !== "-" ? (
                      <span className="timeline-status">{summary.duration}</span>
                    ) : null}
                    <ArtifactChip count={summary.artifactCount} />
                  </div>
                  <div className="mt-1 truncate text-sm text-[var(--muted)]">
                    {summary.summary}
                  </div>
                  {summary.usage ? (
                    <div className="mt-1 text-xs text-[var(--subtle)]">
                      {usageLabel(summary.usage)}
                    </div>
                  ) : null}
                  {summary.childRun ? (
                    <div className="mt-1 text-xs text-[var(--subtle)]">
                      subagent: {summary.childRun.agent_name || summary.childRun.run_name}
                      {summary.childRun.duration ? ` · ${summary.childRun.duration}` : ""}
                    </div>
                  ) : null}
                </div>
                <div className="flex items-center gap-3">
                  <div className="font-mono text-xs text-[var(--muted)]">{summary.time}</div>
                  <Chevron className="h-4 w-4 text-[var(--subtle)]" />
                </div>
              </button>
              {isOpen ? <TimelineDetails group={group} onSelectRun={onSelectRun} /> : null}
            </div>
          </div>
        );
      })}
      </div>
    </div>
  );
}

function StatStrip({ run, display }) {
  const usage = display.usage || {};
  const latency = display.latency || {};
  const stats = [
    { label: "Messages", value: run.message_count || 0, icon: Activity },
    { label: "Duration", value: display.duration || "-", icon: Clock3 },
    { label: "Tokens", value: usage.total_tokens ? formatCount(usage.total_tokens) : "-", icon: Activity },
    {
      label: "Model Latency",
      value: formatDurationMs(latency.slowest_model_ms),
      icon: Clock3,
    },
    {
      label: "Tool Latency",
      value: formatDurationMs(latency.slowest_tool_ms),
      icon: Clock3,
    },
    { label: "Tools", value: display.tool_count || 0, icon: Wrench },
    { label: "Approvals", value: display.approval_count || 0, icon: CheckCircle2 },
    { label: "Errors", value: display.error_count || 0, icon: CircleAlert },
    { label: "Model", value: display.model || "-", icon: TerminalSquare },
  ];
  return (
    <div className="stat-strip">
      {stats.map(({ label, value, icon: Icon }) => (
        <div key={label} className="stat-strip-item">
          <Icon className="h-3.5 w-3.5 text-[var(--subtle)]" />
          <span className="text-xs text-[var(--muted)]">{label}</span>
          <span className="truncate text-xs font-semibold text-[var(--ink)]">{value}</span>
        </div>
      ))}
    </div>
  );
}

function renderMessageBody(message) {
  if (!message) return "";
  if (typeof message.content === "string") return message.content;
  if (message.content?.type === "text_preview") return message.content.preview;
  if (Array.isArray(message.tool_calls)) {
    return message.tool_calls
      .map((call) => `${call.function?.name || call.name || "tool"} (${call.id || "no id"})`)
      .join("\n");
  }
  return JSON.stringify(message, null, 2);
}

function Transcript({ run }) {
  return (
    <div className="space-y-3">
      {(run.messages || []).map((message, index) => (
        <div key={index} className="timeline-card p-3">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs font-bold uppercase text-[var(--muted)]">
              {message.role || "message"}
            </div>
            {message.tool_call_id ? (
              <div className="font-mono text-[11px] text-[var(--subtle)]">{message.tool_call_id}</div>
            ) : null}
          </div>
          <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words text-sm">
            {renderMessageBody(message)}
          </pre>
        </div>
      ))}
    </div>
  );
}

function OperationalSummary({ run, display }) {
  const usage = display.usage || {};
  const latency = display.latency || {};
  const items = [
    ["model", display.model || "-"],
    ["tokens", usage.total_tokens ? formatCount(usage.total_tokens) : "-"],
    ["duration", display.duration || "-"],
    ["tools", display.tool_count || 0],
    ["approvals", display.approval_count || 0],
    ["errors", display.error_count || 0],
    ["slowest model", formatDurationMs(latency.slowest_model_ms)],
    ["slowest tool", formatDurationMs(latency.slowest_tool_ms)],
  ];
  return (
    <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label} className="ops-summary-item">
          <div className="text-[11px] font-semibold uppercase text-[var(--subtle)]">{label}</div>
          <div className="mt-1 truncate text-sm font-semibold text-[var(--ink)]">{value}</div>
        </div>
      ))}
    </div>
  );
}

function focusedRunUrl(traceId) {
  if (typeof window === "undefined" || !traceId) return "";
  const url = new URL(window.location.href);
  url.searchParams.set("embed", "1");
  url.searchParams.set("trace_id", traceId);
  return url.toString();
}

function CopyFocusedLinkButton({ traceId }) {
  const [status, setStatus] = useState("");
  async function copy() {
    const url = focusedRunUrl(traceId);
    if (!url) return;
    try {
      await navigator.clipboard.writeText(url);
      setStatus("Copied");
    } catch (_error) {
      setStatus("Copy failed");
    }
    window.setTimeout(() => setStatus(""), 1600);
  }
  return (
    <div className="flex items-center gap-2">
      <button type="button" className="action-button inline-flex h-8 items-center gap-2 px-3 text-xs font-semibold" onClick={copy}>
        <Clipboard className="h-3.5 w-3.5" />
        Copy focused link
      </button>
      {status ? <span className="text-xs text-[var(--muted)]">{status}</span> : null}
    </div>
  );
}

function Detail({ run, onSelectRun, embedded = false, requestedTraceId = null }) {
  const [tab, setTab] = useState("timeline");
  const [timelineFilter, setTimelineFilter] = useState("all");
  if (!run) {
    return (
      <main className={cx("flex flex-1 items-center justify-center px-6 text-[var(--muted)]", embedded && "embed-detail-main")}>
        <div className="empty-state max-w-xl p-8 text-center">
          <div className="text-sm font-semibold text-[var(--ink)]">
            {requestedTraceId ? "Waiting for selected run" : "No runs yet"}
          </div>
          <div className="mt-2 text-sm leading-relaxed">
            {requestedTraceId
              ? "Run the notebook or CLI cell that writes this trace, then this focused view will update automatically."
              : "Stream events to this viewer, import a JSONL file, or run an agent with viewer.trace_sink."}
          </div>
          {requestedTraceId ? (
            <div className="mt-3 font-mono text-xs text-[var(--subtle)]">{requestedTraceId}</div>
          ) : null}
        </div>
      </main>
    );
  }
  const display = run.display || {};
  const tabs = ["timeline", "transcript", "raw"];
  return (
    <main className={cx("min-w-0 flex-1 overflow-auto px-8 py-7", embedded && "embed-detail-main")}>
      <section className={cx("detail-panel", embedded && "embed-detail-panel")}>
        <div className="detail-hero border-b border-[var(--line)] px-6 py-5">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="truncate text-2xl font-bold">
                {display.title || run.run_name || run.trace_id}
              </div>
              <div className="mt-1 text-sm text-[var(--muted)]">{display.subtitle}</div>
              {display.parent_title ? (
                <div className="mt-1 text-xs text-[var(--subtle)]">
                  Subagent of {display.parent_title}
                </div>
              ) : display.child_count ? (
                <div className="mt-1 text-xs text-[var(--subtle)]">
                  {display.child_count} child run{display.child_count === 1 ? "" : "s"}
                </div>
              ) : null}
              <div className="mt-1 font-mono text-xs text-[var(--subtle)]">
                {run.trace_id}
              </div>
            </div>
            <div className="flex shrink-0 flex-col items-end gap-3">
              <Pill tone={display.status_tone}>{run.status || "running"}</Pill>
              {!embedded ? <CopyFocusedLinkButton traceId={run.trace_id} /> : null}
            </div>
          </div>
          <OperationalSummary run={run} display={display} />
          <StatStrip run={run} display={display} />
        </div>
        <div className="border-b border-[var(--line)] bg-[var(--panel-muted)] px-4">
          <div className="flex gap-1">
            {tabs.map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => setTab(name)}
                className={cx(
                  "border-b-2 px-3 py-3 text-sm font-semibold capitalize",
                  tab === name
                    ? "border-cyan-500 text-[var(--ink)]"
                    : "border-transparent text-[var(--muted)] hover:text-[var(--ink)]",
                )}
              >
                {name}
              </button>
            ))}
          </div>
        </div>
        <div className="px-6 py-5">
          {tab === "timeline" ? (
            <div>
              <div className="mb-5">
                <div className="mb-2 text-sm font-semibold">Final output</div>
                <div className="output-panel p-3 text-sm leading-relaxed">
                  {run.final_output || "No final output yet."}
                </div>
              </div>
              <Timeline run={run} onSelectRun={onSelectRun} filter={timelineFilter} onFilterChange={setTimelineFilter} />
            </div>
          ) : null}
          {tab === "transcript" ? <Transcript run={run} /> : null}
          {tab === "raw" ? (
            <pre className="code-block overflow-auto p-4 text-xs">
              {JSON.stringify(run, null, 2)}
            </pre>
          ) : null}
        </div>
      </section>
    </main>
  );
}

function ThemeToggle({ theme, onToggle }) {
  const Icon = theme === "dark" ? Sun : Moon;
  return (
    <button
      type="button"
      className="icon-button"
      onClick={onToggle}
      title={theme === "dark" ? "Use light theme" : "Use dark theme"}
    >
      <Icon className="h-4 w-4" />
    </button>
  );
}

function ImportBar({ onImported }) {
  const [status, setStatus] = useState("");

  async function handleFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setStatus("Importing...");
    try {
      const text = await file.text();
      const payload = await importJsonl(text);
      setStatus(`Imported ${payload.imported} records`);
      onImported(payload.runs || []);
    } catch (error) {
      setStatus(error.message);
    } finally {
      event.target.value = "";
    }
  }

  return (
    <div className="flex items-center gap-3">
      <label className="action-button inline-flex h-9 cursor-pointer items-center gap-2 px-3 text-sm font-semibold">
        <Upload className="h-4 w-4" />
        Import JSONL
        <input className="hidden" type="file" accept=".jsonl,.json,.txt" onChange={handleFile} />
      </label>
      {status ? <span className="text-sm text-[var(--muted)]">{status}</span> : null}
    </div>
  );
}

function getViewerParams() {
  if (typeof window === "undefined") {
    return { embed: false, traceId: null, theme: null };
  }
  const params = new URLSearchParams(window.location.search);
  const embedValue = params.get("embed");
  return {
    embed: embedValue === "1" || embedValue === "true",
    traceId: params.get("trace_id") || params.get("run"),
    theme: params.get("theme"),
  };
}

function getInitialTheme() {
  if (typeof window === "undefined") return "light";
  const params = getViewerParams();
  if (params.theme === "dark" || params.theme === "light") return params.theme;
  const stored = window.localStorage.getItem("aisuite-viewer-theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const viewerParams = useMemo(() => getViewerParams(), []);
  const [runs, setRuns] = useState([]);
  const [selectedId, setSelectedId] = useState(viewerParams.traceId);
  const [runDetails, setRunDetails] = useState({});
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(getInitialTheme);

  async function refresh() {
    try {
      const payload = await fetchRuns();
      const nextRuns = payload.runs || [];
      setRuns(nextRuns);
      setError("");
      setSelectedId((current) => {
        if (current && nextRuns.some((run) => run.trace_id === current)) {
          return current;
        }
        if (viewerParams.traceId) return viewerParams.traceId;
        return nextRuns[0]?.trace_id || null;
      });
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 1500);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("aisuite-viewer-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!selectedId) return undefined;
    let cancelled = false;
    fetchRun(selectedId)
      .then((payload) => {
        if (cancelled) return;
        setRunDetails((current) => ({
          ...current,
          [selectedId]: payload.run,
        }));
        setError("");
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, runs]);

  const selectedSummary = runs.find((run) => run.trace_id === selectedId) || null;
  const selectedRun = runDetails[selectedId] || selectedSummary;

  if (viewerParams.embed) {
    return (
      <div className="app-shell min-h-screen text-[var(--ink)]">
        {error ? (
          <div className="border-b border-rose-500/20 bg-rose-500/10 px-5 py-2 text-sm text-rose-600">
            {error}
          </div>
        ) : null}
        <Detail
          run={selectedRun}
          onSelectRun={setSelectedId}
          embedded
          requestedTraceId={viewerParams.traceId}
        />
      </div>
    );
  }

  return (
    <div className="app-shell min-h-screen text-[var(--ink)]">
      <header className="topbar sticky top-0 z-20 flex h-[68px] items-center justify-between px-5">
        <div className="flex items-center gap-3">
          <div className="brand-mark">
            <FileJson className="h-5 w-5" />
          </div>
          <div>
            <div className="text-[15px] font-bold tracking-[0.01em]">aisuite runs</div>
            <div className="text-xs text-[var(--muted)]">
              {runs.length} run{runs.length === 1 ? "" : "s"} · local viewer
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ImportBar
            onImported={(nextRuns) => {
              setRuns(nextRuns);
              if (nextRuns.length) setSelectedId(nextRuns[0].trace_id);
            }}
          />
          <ThemeToggle
            theme={theme}
            onToggle={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          />
        </div>
      </header>
      {error ? (
        <div className="border-b border-rose-500/20 bg-rose-500/10 px-5 py-2 text-sm text-rose-600">
          {error}
        </div>
      ) : null}
      <div className="flex">
        <RunList runs={runs} selectedId={selectedRun?.trace_id} onSelect={setSelectedId} />
        <Detail run={selectedRun} onSelectRun={setSelectedId} />
      </div>
    </div>
  );
}
