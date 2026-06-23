import { useEffect, useState } from "react";
import {
  createAutomation,
  deleteAutomation,
  getAutomation,
  getAutomations,
  updateAutomation,
  type Automation,
  type AutomationRun,
} from "../api";
import { Icon } from "./Icon";

// Parse a simple "min hour * * dow" cron back into the time + frequency the editor uses.
// Falls back to 09:00 / daily for anything it doesn't recognize (e.g. agent-written crons).
function fromCron(cron?: string | null): { time: string; freq: string } {
  const parts = (cron || "").trim().split(/\s+/);
  if (parts.length !== 5) return { time: "09:00", freq: "daily" };
  const [m, h, , , dow] = parts;
  const hh = String(Math.min(23, Math.max(0, parseInt(h, 10) || 9))).padStart(2, "0");
  const mm = String(Math.min(59, Math.max(0, parseInt(m, 10) || 0))).padStart(2, "0");
  const freq = dow === "1-5" ? "weekdays" : dow === "0,6" || dow === "6,0" ? "weekends" : "daily";
  return { time: `${hh}:${mm}`, freq };
}

const fmt = (t: number | null) =>
  t ? new Date(t * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

// One-click prebuilt templates. Each maps directly to a createAutomation payload.
interface Template {
  key: string;
  title: string;
  blurb: string;
  instructions: string;
  cron: string;
  when: string;
}

const TEMPLATES: Template[] = [
  {
    key: "news",
    title: "Morning news briefing",
    blurb: "A 5-bullet tech & world news digest, saved as markdown.",
    instructions:
      "Search the web for the most important technology and world news from the last 24 hours and write a concise 5-bullet briefing, saved as a markdown file.",
    cron: "0 8 * * *",
    when: "Every day · 8:00 AM",
  },
  {
    key: "inbox",
    title: "Inbox digest",
    blurb: "One short digest of your unread email.",
    instructions: "Summarize my unread email into one short digest note.",
    cron: "0 9 * * 1-5",
    when: "Weekdays · 9:00 AM",
  },
  {
    key: "cleanup",
    title: "Folder cleanup",
    blurb: "Sort recent Downloads into tidy folders by type.",
    instructions: "Sort my recent Downloads into tidy folders by file type.",
    cron: "30 17 * * 5",
    when: "Fridays · 5:30 PM",
  },
];

// Map a simple time-of-day + frequency selection to a 5-field cron string.
function toCron(time: string, freq: string): string {
  const [h, m] = (time || "09:00").split(":").map((x) => parseInt(x, 10) || 0);
  const dow = freq === "weekdays" ? "1-5" : freq === "weekends" ? "0,6" : "*";
  return `${m} ${h} * * ${dow}`;
}

interface Props {
  onOpenRun: (sessionId: string, workspace: string, agent: string) => void;
  onRunNow: (taskId: string) => void;
}

export function ScheduledView({ onOpenRun, onRunNow }: Props) {
  const [tasks, setTasks] = useState<Automation[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = () => getAutomations().then(setTasks).catch(() => setTasks([]));
  useEffect(() => {
    refresh();
    const h = setInterval(refresh, 5000);
    return () => clearInterval(h);
  }, []);

  // Create from a payload, refresh the list, and open the new task's detail.
  const create = async (payload: {
    title: string;
    instructions: string;
    cron?: string;
  }) => {
    setBusy(payload.title);
    try {
      const res = await createAutomation(payload);
      await refresh();
      if (res.ok && res.task) {
        setShowForm(false);
        setOpenId(res.task.id);
      } else if (res.error) {
        alert(res.error);
      }
    } finally {
      setBusy(null);
    }
  };

  if (openId) {
    return (
      <TaskDetail
        id={openId}
        onBack={() => { setOpenId(null); refresh(); }}
        onOpenRun={onOpenRun}
        onRunNow={onRunNow}
      />
    );
  }

  const empty = tasks.length === 0;

  return (
    <div className="main page-view">
      <div className="page-col">
      <div className="sa-view-head">
        <div className="sa-view-heading">
          <div className="sa-view-title"><Icon name="clock" size={21} /> Automations</div>
          <div className="sa-view-sub">Recurring tasks OpenCoworker runs on a schedule.</div>
        </div>
        <button className="btn new-action" onClick={() => setShowForm((v) => !v)}>
          + New automation
        </button>
      </div>
      <div className="main-scroll">
        <div className="sched-banner">
          <span className="ico">ⓘ</span>
          <span>
            Scheduled tasks only run while <strong>coworker-server</strong> is running. If it's off at
            the scheduled time, the task runs once when the server next starts (catch-up).
          </span>
        </div>

        {showForm && (
          <NewAutomationForm
            busy={busy !== null}
            onCancel={() => setShowForm(false)}
            onCreate={create}
          />
        )}

        {(empty || showForm) && (
          <div className="tmpl-wrap">
            <div className="sa-sub tmpl-head">Start from a template</div>
            <div className="tmpl-grid">
              {TEMPLATES.map((t) => (
                <div className="tmpl-card" key={t.key}>
                  <div className="tmpl-title">{t.title}</div>
                  <div className="tmpl-blurb">{t.blurb}</div>
                  <div className="tmpl-when"><Icon name="clock" size={12} /> {t.when}</div>
                  <button
                    className="btn sm tmpl-add"
                    disabled={busy !== null}
                    onClick={() =>
                      create({ title: t.title, instructions: t.instructions, cron: t.cron })
                    }
                  >
                    {busy === t.title ? "Creating…" : "Use this"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {empty ? (
          !showForm && (
            <div className="hero tmpl-empty">
              <div className="suggest-head">
                No scheduled tasks yet — use a template above, click <strong>+ New automation</strong>,
                or just ask OpenCoworker in a session.
              </div>
            </div>
          )
        ) : (
          <div className="sched-list">
            {tasks.map((t) => (
              <div className="sched-card" key={t.id} onClick={() => setOpenId(t.id)}>
                <div className="sched-card-top">
                  <span className="conn-name">{t.title}</span>
                  <button
                    className="sched-card-del"
                    title="Delete automation"
                    aria-label={`Delete ${t.title}`}
                    onClick={async (e) => {
                      e.stopPropagation();
                      await deleteAutomation(t.id);
                      refresh();
                    }}
                  >
                    <Icon name="trash" size={14} />
                  </button>
                </div>
                <div className="sched-card-meta">
                  <Icon name="clock" size={13} className="sched-clock" />
                  {t.enabled ? t.schedule : "Paused"} · next {fmt(t.next_run)} · {t.run_count} run{t.run_count === 1 ? "" : "s"}
                  {t.last_status ? ` · last ${t.last_status}` : ""}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      </div>
    </div>
  );
}

function NewAutomationForm({
  busy,
  onCancel,
  onCreate,
}: {
  busy: boolean;
  onCancel: () => void;
  onCreate: (p: { title: string; instructions: string; cron?: string }) => void;
}) {
  const [title, setTitle] = useState("");
  const [instructions, setInstructions] = useState("");
  const [time, setTime] = useState("09:00");
  const [freq, setFreq] = useState("daily");

  const valid = title.trim() && instructions.trim();

  return (
    <div className="tmpl-form">
      <div className="sa-sub tmpl-head">New automation</div>
      <input
        className="tmpl-input"
        placeholder="Title (e.g. Daily standup notes)"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <textarea
        className="tmpl-input tmpl-textarea"
        placeholder="What should it do each run? (e.g. Summarize today's calendar and open tasks.)"
        value={instructions}
        onChange={(e) => setInstructions(e.target.value)}
      />
      <div className="tmpl-sched">
        <label className="tmpl-field">
          <span>At</span>
          <input
            type="time"
            className="tmpl-input tmpl-time"
            value={time}
            onChange={(e) => setTime(e.target.value)}
          />
        </label>
        <label className="tmpl-field">
          <span>Repeat</span>
          <select
            className="tmpl-input tmpl-select"
            value={freq}
            onChange={(e) => setFreq(e.target.value)}
          >
            <option value="daily">Every day</option>
            <option value="weekdays">Weekdays</option>
            <option value="weekends">Weekends</option>
          </select>
        </label>
      </div>
      <div className="tmpl-form-actions">
        <button
          className="btn-primary sm"
          disabled={!valid || busy}
          onClick={() =>
            onCreate({
              title: title.trim(),
              instructions: instructions.trim(),
              cron: toCron(time, freq),
            })
          }
        >
          {busy ? "Creating…" : "Create automation"}
        </button>
        <button className="link" onClick={onCancel}>cancel</button>
      </div>
    </div>
  );
}

function TaskDetail({
  id,
  onBack,
  onOpenRun,
  onRunNow,
}: {
  id: string;
  onBack: () => void;
  onOpenRun: (sessionId: string, workspace: string, agent: string) => void;
  onRunNow: (taskId: string) => void;
}) {
  const [task, setTask] = useState<Automation | null>(null);
  const [runs, setRuns] = useState<AutomationRun[]>([]);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState("");
  const [instructions, setInstructions] = useState("");
  const [time, setTime] = useState("09:00");
  const [freq, setFreq] = useState("daily");
  const [saving, setSaving] = useState(false);

  const refresh = () =>
    getAutomation(id)
      .then((d) => {
        setTask(d.task);
        setRuns(d.runs || []);
      })
      .catch(() => {});
  useEffect(() => {
    refresh();
  }, [id]);

  if (!task) return <div className="main"><div className="main-scroll"><div className="manage-empty">Loading…</div></div></div>;

  const startEdit = () => {
    setTitle(task.title);
    setInstructions(task.instructions);
    const { time: t, freq: f } = fromCron(task.schedule_raw?.cron);
    setTime(t);
    setFreq(f);
    setEditing(true);
  };
  const saveEdit = async () => {
    setSaving(true);
    try {
      await updateAutomation(id, {
        title: title.trim(),
        instructions: instructions.trim(),
        cron: toCron(time, freq),
      });
      await refresh();
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };
  const toggle = async () => {
    await updateAutomation(id, { enabled: !task.enabled });
    refresh();
  };
  const remove = async () => {
    await deleteAutomation(id);
    onBack();
  };

  return (
    <div className="main page-view">
      <div className="page-col">
      <div className="sa-view-head">
        <button className="sa-back" onClick={onBack}>← Automations</button>
      </div>
      <div className="main-scroll">
        <div className="sched-detail">
          <div className="sched-detail-head">
            {editing ? (
              <input
                className="tmpl-input sched-edit-title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Title"
              />
            ) : (
              <h2>{task.title}</h2>
            )}
            <div className="sched-actions">
              {editing ? (
                <>
                  <button className="btn-primary sm" disabled={saving || !title.trim() || !instructions.trim()} onClick={saveEdit}>
                    {saving ? "Saving…" : "Save"}
                  </button>
                  <button className="link" onClick={() => setEditing(false)}>cancel</button>
                </>
              ) : (
                <>
                  <button className="btn-primary sm" onClick={() => onRunNow(id)}>
                    ▶ Run now
                  </button>
                  <button className="btn sm" onClick={startEdit}>Edit</button>
                  <button className="btn sm danger-btn" onClick={remove}>
                    <Icon name="trash" size={14} /> Delete
                  </button>
                </>
              )}
            </div>
          </div>

          {editing ? (
            <div className="tmpl-sched sched-edit-sched">
              <label className="tmpl-field">
                <span>At</span>
                <input type="time" className="tmpl-input tmpl-time" value={time} onChange={(e) => setTime(e.target.value)} />
              </label>
              <label className="tmpl-field">
                <span>Repeat</span>
                <select className="tmpl-input tmpl-select" value={freq} onChange={(e) => setFreq(e.target.value)}>
                  <option value="daily">Every day</option>
                  <option value="weekdays">Weekdays</option>
                  <option value="weekends">Weekends</option>
                </select>
              </label>
            </div>
          ) : (
            <div className="conn-meta">
              <label className="switch">
                <input type="checkbox" checked={task.enabled} onChange={toggle} />
                <span className="slider" />
              </label>{" "}
              {task.enabled ? `Active · next ${fmt(task.next_run)}` : "Paused"} · {task.schedule}
            </div>
          )}

          <div className="sa-sub">Instructions</div>
          {editing ? (
            <textarea
              className="tmpl-input tmpl-textarea sched-edit-instr"
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
            />
          ) : (
            <div className="sched-instructions">{task.instructions}</div>
          )}

          <div className="sa-sub">Runs</div>
          <div className="dim" style={{ marginBottom: 8, fontSize: 12.5 }}>
            Each run is a live conversation — open one to see what the agent did and ask a follow-up.
          </div>
          {runs.length === 0 && <div className="dim">No runs yet.</div>}
          {runs.map((r) => (
            <div
              className="sched-run open"
              key={r.run_id}
              onClick={() => r.session_id && onOpenRun(r.session_id, task.workspace, task.agent)}
              title="Open this run's conversation"
            >
              <div className="sched-run-row">
                <span>
                  {fmt(r.started_at)} · <span className={"run-" + r.status}>{r.status}</span> · {r.trigger}
                  {r.artifacts.length > 0 && <span className="dim"> · {r.artifacts.length} file(s)</span>}
                </span>
                <span className="sched-run-go" aria-hidden>
                  Open ›
                </span>
              </div>
              {r.result_text && <div className="sched-run-peek">{r.result_text}</div>}
              {r.error && <div className="mcp-error">{r.error}</div>}
            </div>
          ))}
        </div>
      </div>
      </div>
    </div>
  );
}
