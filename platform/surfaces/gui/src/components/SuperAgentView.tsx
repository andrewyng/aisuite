import { useEffect, useRef, useState } from "react";
import { getSuperagent, SuperagentSession, type SuperagentStatus } from "../api";
import type { ApprovalDecision, Item, WsEvent } from "../types";
import { Transcript } from "./Transcript";

const newId = () =>
  (crypto as any).randomUUID ? crypto.randomUUID().slice(0, 12) : Math.random().toString(36).slice(2, 14);

function itemsFromMessages(messages: any[]): Item[] {
  const items: Item[] = [];
  for (const m of messages || []) {
    if (m.role === "user" && typeof m.content === "string") {
      items.push({ kind: "user", text: m.content });
    } else if (m.role === "assistant") {
      if (m.content) items.push({ kind: "assistant", text: m.content });
      for (const tc of m.tool_calls || []) {
        let args: any = {};
        try {
          args = JSON.parse(tc.function?.arguments || "{}");
        } catch {
          args = {};
        }
        items.push({ kind: "tool", id: tc.id, name: tc.function?.name, args, status: "ok" });
      }
    }
  }
  return items;
}

function updateLastTool(items: Item[], name: string, status: string, preview?: string): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "tool" && it.name === name && it.status === "…") {
      copy[i] = { ...it, status, preview };
      break;
    }
  }
  return copy;
}

function resolveLastApproval(items: Item[], decision: ApprovalDecision): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "approval" && !it.resolved) {
      copy[i] = { ...it, resolved: decision };
      break;
    }
  }
  return copy;
}

export function SuperAgentView() {
  const [items, setItems] = useState<Item[]>([]);
  const [streaming, setStreaming] = useState("");
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<SuperagentStatus | null>(null);
  const [text, setText] = useState("");

  const sessionRef = useRef<SuperagentSession | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    getSuperagent().then(setStatus).catch(() => {});
  }, []);

  useEffect(() => {
    const handle = (ev: WsEvent) => {
      const d = ev.data || {};
      switch (ev.type) {
        case "ready":
          setConnected(true);
          setItems(itemsFromMessages(d.transcript || []));
          break;
        case "inbound": // a message arrived from a connected platform (Telegram/Slack)
          setItems((p) => [...p, { kind: "user", text: `📨 ${d.source}: ${d.text}` }]);
          break;
        case "turn_start":
          setRunning(true);
          setStreaming("");
          break;
        case "assistant_delta":
          setStreaming((s) => s + (d.text || ""));
          break;
        case "assistant_message":
          if (d.text) setItems((p) => [...p, { kind: "assistant", text: d.text }]);
          setStreaming("");
          break;
        case "tool_proposed":
          setItems((p) => [...p, { kind: "tool", id: newId(), name: d.name, args: d.arguments, status: "…" }]);
          break;
        case "permission_required":
          setItems((p) => [...p, { kind: "approval", name: d.name, args: d.arguments, reason: d.reason }]);
          break;
        case "tool_finished":
          setItems((p) => updateLastTool(p, d.name, d.status, d.result_preview || d.reason));
          break;
        case "turn_end":
          setRunning(false);
          break;
        case "interrupted":
          setRunning(false);
          setItems((p) => [...p, { kind: "notice", tone: "warn", text: "Interrupted." }]);
          break;
        case "error":
          setItems((p) => [...p, { kind: "notice", tone: "warn", text: "Error: " + (d.error || "unknown") }]);
          break;
      }
    };

    const session = new SuperagentSession({
      onEvent: handle,
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
    });
    sessionRef.current = session;
    return () => session.close();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [items, streaming]);

  const send = () => {
    const t = text.trim();
    if (!t) return;
    setItems((p) => [...p, { kind: "user", text: t }]);
    sessionRef.current?.userMessage(t);
    setText("");
  };
  const approve = (decision: ApprovalDecision) => {
    setItems((p) => resolveLastApproval(p, decision));
    sessionRef.current?.approve(decision);
  };

  const listening = (status?.connectors || []).filter((c) => c.listening).map((c) => c.name);

  return (
    <div className="main">
      <div className="sa-view-head">
        <div className="sa-view-title">
          <span className="mark">✳</span> {status?.name || "MyHelper"}
          <span className={"sa-status-dot" + (connected ? " on" : "")} />
        </div>
        <div className="sa-view-sub">
          {listening.length ? `listening on ${listening.join(", ")}` : "no bots connected"} ·{" "}
          {status?.workspace || ""}
        </div>
      </div>

      <div className="main-scroll" ref={scrollRef}>
        {items.length === 0 && !streaming ? (
          <div className="hero">
            <h1 className="greeting">
              <span className="mark">✳</span> Your always-on coworker.
            </h1>
            <div className="suggest-head">
              Message it here, or DM your connected bot. It runs on one continuous thread.
            </div>
          </div>
        ) : (
          <>
            <Transcript items={items} onApprove={approve} />
            {streaming && (
              <div className="transcript">
                <div className="bubble-assistant">
                  <div className="who">assistant</div>
                  {streaming}
                  <span className="stream-cursor">▍</span>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <div className="sa-composer">
        <textarea
          value={text}
          placeholder="Message your super-agent…"
          rows={1}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        {running ? (
          <button className="btn danger" onClick={() => sessionRef.current?.interrupt()}>
            Stop
          </button>
        ) : (
          <button className="btn primary" onClick={send} disabled={!connected || !text.trim()}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}
