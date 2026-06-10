import { useState } from "react";
import type { ApprovalDecision, Item } from "../types";
import { shortArgs } from "./ApprovalCard";
import { Markdown } from "./Markdown";

type ToolItem = Extract<Item, { kind: "tool" }>;
type ApprovalItem = Extract<Item, { kind: "approval" }>;
type ActivityItem = ToolItem | ApprovalItem;

function ToolGroup({ items }: { items: ActivityItem[] }) {
  const [open, setOpen] = useState(false);
  const tools = items.filter((item): item is ToolItem => item.kind === "tool");
  const running = tools.some((t) => t.status === "…");
  const label = running ? `Running ${tools.length} tool${tools.length > 1 ? "s" : ""}…` : `Ran ${tools.length} tool${tools.length > 1 ? "s" : ""}`;
  return (
    <div className="toolgroup">
      <div className="toolgroup-head" onClick={() => setOpen(!open)}>
        <span className="chev">{open ? "⌄" : "›"}</span> {label}
      </div>
      {open && (
        <div className="toolgroup-body">
          {items.map((item, i) =>
            item.kind === "approval" ? (
              <div className="toolrow approval-row" key={i}>
                <span className="status ok">✓</span>
                <span className="name">approved</span>
                <span className="rowargs">
                  {item.name} · {item.resolved?.replace("_", " ")}
                </span>
              </div>
            ) : (
              <div className="toolrow" key={i}>
                <div className="toolrow-head">
                  <span className={"status " + item.status}>
                    {item.status === "ok" ? "✓" : item.status === "…" ? "…" : "•"}
                  </span>
                  <span className="name">{item.name}</span>
                  <span className="rowargs">{shortArgs(item.args)}</span>
                </div>
                {item.preview && (
                  <pre className="toolresult">
                    {item.preview.length > 1500 ? item.preview.slice(0, 1500) + "\n…" : item.preview}
                  </pre>
                )}
              </div>
            ),
          )}
        </div>
      )}
    </div>
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
        if ("activities" in block) return <ToolGroup items={block.activities} key={bi} />;
        const { item } = block;
        switch (item.kind) {
          case "user":
            return (
              <div className="bubble-user" key={bi}>
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
