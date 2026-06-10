import type { ApprovalDecision, Item } from "../types";
import { Icon } from "./Icon";

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

// Human verbs for the common tools, so the card reads as an action, not an identifier.
const TOOL_VERBS: Record<string, string> = {
  write_file: "Write a file",
  replace_in_file: "Edit a file",
  apply_patch: "Apply a patch",
  apply_unified_diff: "Apply a patch",
  run_shell: "Run a command",
  send_message: "Send a message",
};

// The one argument worth showing prominently (path for file tools, command for shell).
function primaryDetail(args: any): { key: string; value: string } | null {
  if (!args || typeof args !== "object") return null;
  if (typeof args.path === "string" && args.path) return { key: "path", value: args.path };
  if (typeof args.command === "string" && args.command) return { key: "command", value: args.command };
  return null;
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
  const verb = TOOL_VERBS[item.name];
  const detail = primaryDetail(item.args);
  const rest = shortArgs(
    Object.fromEntries(Object.entries(item.args || {}).filter(([k]) => k !== detail?.key)),
  );
  // "requires approval" is the engine's default boilerplate — only surface a real reason.
  const reason = item.reason && item.reason !== "requires approval" ? item.reason : "";

  return (
    <div className={"approval" + (compact ? " approval-dock" : "")}>
      <div className="approval-top">
        <div className="approval-heading">
          <span className="approval-ico">
            <Icon name="shield" size={17} />
          </span>
          <div>
            <div className="title">Permission required</div>
            <div className="approval-verb">
              {verb || "Use a tool"} <code className="approval-tool">{item.name}</code>
            </div>
          </div>
        </div>
        <span className={"approval-badge" + (connector ? " connector" : "")}>
          {connector ? "connector" : "local action"}
        </span>
      </div>

      {detail && (
        <div className="approval-detail" title={detail.value}>
          {detail.value}
        </div>
      )}
      {rest && <div className="approval-rest">{rest}</div>}
      {reason && <div className="approval-reason">{reason}</div>}

      {item.resolved ? (
        <div className="resolved">Approved: {item.resolved.replace("_", " ")}</div>
      ) : (
        <div className="approval-btns">
          <button className="btn primary" onClick={() => onApprove("once")}>
            Allow once
          </button>
          {!connector && (
            <button className="btn" onClick={() => onApprove("always_tool")}>
              Always allow
            </button>
          )}
          {item.name === "run_shell" && (
            <button className="btn" onClick={() => onApprove("always_command")}>
              Always this command
            </button>
          )}
          <span className="spacer" />
          <button className="btn danger" onClick={() => onApprove("deny")}>
            Deny
          </button>
        </div>
      )}
    </div>
  );
}
