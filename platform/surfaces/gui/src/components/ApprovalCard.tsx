import type { ApprovalDecision, Item } from "../types";

export function shortArgs(args: any): string {
  if (!args || typeof args !== "object") return "";
  return Object.entries(args)
    .map(([k, v]) => {
      let s = typeof v === "string" ? v : JSON.stringify(v);
      if (s.length > 96) s = s.slice(0, 95) + "...";
      return `${k}=${s.replace(/\n/g, " ")}`;
    })
    .join("  ");
}

type ApprovalItem = Extract<Item, { kind: "approval" }>;

export function ApprovalCard({
  item,
  onApprove,
  compact = false,
}: {
  item: ApprovalItem;
  onApprove: (decision: ApprovalDecision) => void;
  compact?: boolean;
}) {
  const connector = item.category === "connector";
  return (
    <div className={"approval" + (compact ? " approval-dock" : "")}>
      <div className="approval-top">
        <div>
          <div className="title">Permission required</div>
          <div className="approval-tool">{item.name}</div>
        </div>
        <span className={"approval-badge" + (connector ? " connector" : "")}>
          {connector ? "connector" : "local action"}
        </span>
      </div>
      <div className="approval-grid">
        <span className="k">args</span>
        <span className="v">{shortArgs(item.args) || "-"}</span>
        <span className="k">reason</span>
        <span className="v">{item.reason || "requires approval"}</span>
      </div>
      {item.resolved ? (
        <div className="resolved">Approved: {item.resolved.replace("_", " ")}</div>
      ) : (
        <div className="approval-btns">
          <button className="btn primary" onClick={() => onApprove("once")}>
            Approve once
          </button>
          <button className="btn danger" onClick={() => onApprove("deny")}>
            Deny
          </button>
          {!connector && (
            <>
              <button className="btn" onClick={() => onApprove("always_tool")}>
                Always this tool
              </button>
              <button className="btn" onClick={() => onApprove("always_command")}>
                Always this command
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
