import { useEffect, useState } from "react";
import {
  deleteAutomation,
  getAutomation,
  getAutomations,
  updateAutomation,
  type Automation,
  type AutomationRun,
} from "../api";

const fmt = (t: number | null) =>
  t ? new Date(t * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

interface Props {
  onOpenRun: (sessionId: string, workspace: string, agent: string) => void;
  onRunNow: (taskId: string) => void;
}

export function ScheduledView({ onOpenRun, onRunNow }: Props) {
  const [tasks, setTasks] = useState<Automation[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);

  const refresh = () => getAutomations().then(setTasks).catch(() => setTasks([]));
  useEffect(() => {
    refresh();
    const h = setInterval(refresh, 5000);
    return () => clearInterval(h);
  }, []);

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

  return (
    <div className="main">
      <div className="sa-view-head">
        <div className="sa-view-title">
          <span className="mark">⏰</span> Scheduled tasks
        </div>
        <div className="sa-view-sub">
          Run tasks on a schedule. Ask OpenCoworker or MyHelper to "set up an automation…".
        </div>
      </div>
      <div className="main-scroll">
        <div className="sched-banner">
          <span className="ico">ⓘ</span>
          <span>
            Scheduled tasks only run while <strong>coworker-server</strong> is running. If it's off at
            the scheduled time, the task runs once when the server next starts (catch-up).
          </span>
        </div>
        {tasks.length === 0 ? (
          <div className="hero">
            <h1 className="greeting"><span className="mark">⏰</span> No scheduled tasks yet.</h1>
            <div className="suggest-head">
              In an OpenCoworker session, try: "Search the web and give me a news briefing every day at 7:10pm."
            </div>
          </div>
        ) : (
          <div className="sched-list">
            {tasks.map((t) => (
              <div className="sched-card" key={t.id} onClick={() => setOpenId(t.id)}>
                <div className="sched-card-top">
                  <span className="conn-name">{t.title}</span>
                  <span className={"sched-pill" + (t.enabled ? " on" : "")}>
                    🕐 {t.enabled ? t.schedule : "paused"}
                  </span>
                </div>
                <div className="conn-meta">
                  next {fmt(t.next_run)} · {t.run_count} run{t.run_count === 1 ? "" : "s"}
                  {t.last_status ? ` · last ${t.last_status}` : ""}
                </div>
              </div>
            ))}
          </div>
        )}
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

  const toggle = async () => {
    await updateAutomation(id, { enabled: !task.enabled });
    refresh();
  };
  const remove = async () => {
    await deleteAutomation(id);
    onBack();
  };

  return (
    <div className="main">
      <div className="sa-view-head">
        <div className="sa-view-title">
          <button className="link" onClick={onBack}>← Scheduled</button>
        </div>
      </div>
      <div className="main-scroll">
        <div className="sched-detail">
          <div className="sched-detail-head">
            <h2>{task.title}</h2>
            <div className="sched-actions">
              <button className="btn-primary sm" onClick={() => onRunNow(id)}>
                ▶ Run now
              </button>
              <button className="link danger" onClick={remove}>delete</button>
            </div>
          </div>
          <div className="conn-meta">
            <label className="switch">
              <input type="checkbox" checked={task.enabled} onChange={toggle} />
              <span className="slider" />
            </label>{" "}
            {task.enabled ? `Active · next ${fmt(task.next_run)}` : "Paused"} · {task.schedule}
          </div>

          <div className="sa-sub">Instructions</div>
          <div className="sched-instructions">{task.instructions}</div>

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
  );
}
