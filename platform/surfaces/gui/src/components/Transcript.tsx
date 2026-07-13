import { useState } from "react";
import type { ApprovalDecision, Item } from "../types";
import { shortArgs } from "./ApprovalCard";
import { Markdown } from "./Markdown";
import { ConnectorMessageCard } from "./ConnectorMessageCard";

type ToolItem = Extract<Item, { kind: "tool" }>;
type ApprovalItem = Extract<Item, { kind: "approval" }>;
type ActivityItem = ToolItem | ApprovalItem;

// StepGroup (§7 / UX §2b): a collapsed, expandable disclosure that wraps a run of consecutive
// tool/approval items into one summary line — "N actions · M approvals ✓". Pure frontend grouping of
// EXISTING items (no data change); a <details>-style disclosure driven by controlled state so the
// open/closed render is deterministic (jsdom's native <details> toggle is unreliable in tests).
function StepGroup({ items }: { items: ActivityItem[] }) {
  const [open, setOpen] = useState(false);
  const tools = items.filter((item): item is ToolItem => item.kind === "tool");
  const approvals = items.filter((item): item is ApprovalItem => item.kind === "approval");
  const running = tools.some((t) => t.status === "…");
  const nActions = tools.length;
  const mApprovals = approvals.length;
  // Filter-hidden results surface on the collapsed line too — out-of-band info
  // for the user must not hide behind a disclosure.
  const hiddenTotal = tools.reduce((n, t) => n + (t.hidden || 0), 0);
  const actionsLabel = `${nActions} action${nActions === 1 ? "" : "s"}`;
  const approvalsLabel =
    mApprovals > 0 ? `${mApprovals} approval${mApprovals === 1 ? "" : "s"} ✓` : "";

  return (
    <details className="stepgroup rounded-lg border border-line bg-panel overflow-hidden" open={open}>
      <summary
        className="stepgroup-head flex items-center gap-2 px-3 py-2 cursor-pointer select-none text-[12.5px] text-muted"
        onClick={(e) => {
          e.preventDefault(); // drive open/closed from state, not the native toggle
          setOpen((v) => !v);
        }}
      >
        <span className={"chev inline-block text-faint transition-transform" + (open ? " rotate-90" : "")}>›</span>
        <span>
          <span>{running ? `Running ${actionsLabel}…` : actionsLabel}</span>
          {approvalsLabel && (
            <>
              {" · "}
              <span className="text-ok font-medium">{approvalsLabel}</span>
            </>
          )}
          {hiddenTotal > 0 && (
            <>
              {" · "}
              <span className="text-warnInk" data-testid="stepgroup-hidden">
                {hiddenTotal} hidden by your filters
              </span>
            </>
          )}
        </span>
        <span className="ml-auto text-[11px] text-faint">{open ? "hide details" : "show details"}</span>
      </summary>
      {open && (
        <div className="px-2 pb-2 pt-1 space-y-2 border-t border-line">
          {items.map((item, i) =>
            item.kind === "approval" ? (
              <div className="rounded-lg border border-tealLine bg-tealSoft" key={i}>
                <div className="flex items-center gap-2 px-3 py-2">
                  <span className="inline-flex items-center gap-1 text-[11.5px] text-ok font-medium">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                      <path d="M5 13l4 4L19 7" />
                    </svg>
                    approved
                  </span>
                  <span className="text-[13px]">
                    <span className="font-mono">{item.name}</span> approval
                  </span>
                  <span className="text-[11.5px] text-muted">· {item.resolved?.replace("_", " ")}</span>
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-line bg-panel" key={i}>
                <div className="flex items-center gap-2 px-3 py-2">
                  <span className="w-5 h-5 rounded grid place-items-center bg-tealSoft text-tealInk">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M4 7h16M4 12h16M4 17h10" />
                    </svg>
                  </span>
                  <span className="font-mono text-[12.5px]">{item.name}</span>
                  <span className="font-mono text-[11.5px] text-faint truncate">{shortArgs(item.args)}</span>
                  {!!item.hidden && (
                    <span
                      className="text-[11px] text-warnInk shrink-0"
                      data-testid="tool-hidden-count"
                      title="Removed by your privacy filters before the agent saw the results — agents get no trace of these."
                    >
                      {item.hidden} hidden
                    </span>
                  )}
                  <span
                    className={
                      "ml-auto text-[11px] px-1.5 py-0.5 rounded border " +
                      (item.status === "ok"
                        ? "bg-okSoft text-ok border-okLine"
                        : "bg-paper text-faint border-line")
                    }
                  >
                    {item.status === "ok" ? "done" : item.status === "…" ? "running…" : item.status}
                  </span>
                </div>
                {item.preview && (
                  <pre className="mx-3 mb-2 px-2.5 py-1.5 rounded border border-line bg-paper font-mono text-[11.5px] leading-relaxed text-muted whitespace-pre-wrap break-words max-h-56 overflow-auto">
                    {item.preview.length > 1500 ? item.preview.slice(0, 1500) + "\n…" : item.preview}
                  </pre>
                )}
              </div>
            ),
          )}
        </div>
      )}
    </details>
  );
}

function ApprovalOneLine({ item }: { item: ApprovalItem }) {
  return (
    <div className="approval-inline">
      <span className="status ok">✓</span>
      <span>Approved {item.name}</span>
      <span className="dim">{item.resolved?.replace("_", " ")}</span>
    </div>
  );
}

interface Props {
  items: Item[];
  onApprove: (decision: ApprovalDecision) => void;
}

export function Transcript({ items }: Props) {
  // Group tool calls and resolved approvals into one collapsible activity block.
  const blocks: Array<{ activities: ActivityItem[] } | { item: Item; i: number }> = [];
  let run: ActivityItem[] = [];
  const flush = () => {
    if (run.length) {
      blocks.push({ activities: run });
      run = [];
    }
  };
  items.forEach((item, i) => {
    if (item.kind === "tool" || (item.kind === "approval" && item.resolved)) run.push(item);
    else {
      flush();
      blocks.push({ item, i });
    }
  });
  flush();

  return (
    <div className="transcript">
      {blocks.map((block, bi) => {
        if ("activities" in block) return <StepGroup items={block.activities} key={bi} />;
        const { item } = block;
        switch (item.kind) {
          case "connector":
            return <ConnectorMessageCard source={item.source} key={bi} />;
          case "user":
            return (
              <div
                className="bubble-user self-end max-w-[78%] px-3.5 py-2.5 rounded-[14px_14px_4px_14px] bg-solid text-onSolid text-[14.5px] leading-relaxed whitespace-pre-wrap"
                key={bi}
              >
                {item.attachments && item.attachments.length > 0 && (
                  <div className="bubble-attachments">
                    {item.attachments.map((a, i) =>
                      a.kind === "image" ? (
                        <img key={i} className="msg-img" src={a.data_url} alt={a.name} />
                      ) : (
                        <span key={i} className="msg-file">📄 {a.name}</span>
                      ),
                    )}
                  </div>
                )}
                {item.text}
              </div>
            );
          case "assistant":
            return (
              <div className="bubble-assistant" key={bi}>
                <div className="who">assistant</div>
                <Markdown text={item.text} />
              </div>
            );
          case "approval":
            if (!item.resolved) return null;
            return <ApprovalOneLine item={item} key={bi} />;
          case "dirreq":
            if (!item.resolved) return null;
            return (
              <div className="approval-inline" key={bi}>
                <span className={"status " + (item.resolved === "granted" ? "ok" : "denied")}>
                  {item.resolved === "granted" ? "✓" : "✕"}
                </span>
                <span>{item.resolved === "granted" ? "Granted folder access" : "Declined folder access"}</span>
                {item.path && <span className="dim">{item.path}</span>}
              </div>
            );
          case "planreq":
            if (!item.resolved) return null; // pending plan renders in the composer head
            return (
              <div className="bubble-assistant" key={bi}>
                <div className="who">proposed plan</div>
                <Markdown text={item.plan} />
                <div className="approval-inline">
                  <span className={"status " + (item.resolved === "approved" ? "ok" : "denied")}>
                    {item.resolved === "approved" ? "✓" : "✕"}
                  </span>
                  <span>{item.resolved === "approved" ? "Plan approved" : "Sent back with feedback"}</span>
                </div>
              </div>
            );
          case "notice":
            return (
              <div className={"notice " + (item.tone === "warn" ? "warn" : "")} key={bi}>
                {item.text}
              </div>
            );
          default:
            return null;
        }
      })}
    </div>
  );
}
