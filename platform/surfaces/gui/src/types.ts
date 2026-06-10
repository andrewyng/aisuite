export type EventType =
  | "ready"
  | "inbound"
  | "turn_start"
  | "assistant_delta"
  | "assistant_message"
  | "tool_proposed"
  | "permission_required"
  | "directory_requested"
  | "tool_started"
  | "tool_finished"
  | "iteration_end"
  | "turn_end"
  | "error"
  | "interrupted"
  | "turn_done";

export interface WsEvent {
  type: EventType;
  data: any;
}

export type ApprovalDecision = "once" | "deny" | "always_tool" | "always_command";

export interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "done";
}

export interface SessionInfo {
  session_id: string;
  title?: string;
  workspace: string;
  agent: string;
  model: string;
  mode: string;
  updated_at: string | null;
  messages: number;
  pinned?: boolean;
  archived?: boolean;
}

// Attachments (images, text files) sent with a user message.
export interface Attachment {
  kind: "image" | "text";
  name: string;
  mime?: string;
  data_url?: string; // images
  text?: string; // text files
}

// Transcript items
export type Item =
  | { kind: "user"; text: string; attachments?: Attachment[] }
  | { kind: "assistant"; text: string }
  | { kind: "tool"; id: string; name: string; args: any; status: string; preview?: string }
  | {
      kind: "approval";
      name: string;
      args: any;
      reason: string;
      category?: string;
      resolved?: ApprovalDecision;
    }
  | {
      kind: "dirreq";
      reason: string;
      path?: string;
      writable?: boolean;
      resolved?: "granted" | "denied";
    }
  | { kind: "notice"; tone: "info" | "warn"; text: string };
