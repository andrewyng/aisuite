import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Transcript } from "./Transcript";
import type { Item } from "../types";

// A run of consecutive tool calls + a resolved approval — grouped into one StepGroup disclosure.
const ITEMS: Item[] = [
  { kind: "tool", id: "t1", name: "send_message", args: { target: "slack:#ocw-test" }, status: "ok" },
  { kind: "tool", id: "t2", name: "read_file", args: { path: "runbook.md" }, status: "ok" },
  { kind: "approval", name: "write_file", args: { path: "notes.md" }, reason: "", resolved: "once" },
];

afterEach(cleanup);

describe("StepGroup (Transcript)", () => {
  it("collapses consecutive tool/approval items into one summary and expands on click", () => {
    const { container } = render(<Transcript items={ITEMS} onApprove={vi.fn()} />);

    // One summary line: "N actions · M approvals ✓".
    expect(screen.getByText("2 actions")).toBeTruthy();
    expect(screen.getByText(/1 approval/)).toBeTruthy();

    // Collapsed by default: the individual tool rows are not rendered, and the <details> is closed.
    const details = container.querySelector("details.stepgroup") as HTMLDetailsElement;
    expect(details).toBeTruthy();
    expect(details.open).toBe(false);
    expect(screen.queryByText("send_message")).toBeNull();

    // Expand: clicking the summary reveals the grouped items.
    fireEvent.click(container.querySelector("summary.stepgroup-head")!);
    expect(details.open).toBe(true);
    expect(screen.getByText("send_message")).toBeTruthy();
    expect(screen.getByText("read_file")).toBeTruthy();
  });
});
