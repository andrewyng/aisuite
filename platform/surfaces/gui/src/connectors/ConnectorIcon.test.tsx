import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render } from "@testing-library/react";
import { ConnectorBadge, ConnectorIcon } from "./ConnectorIcon";

afterEach(cleanup);

describe("ConnectorIcon", () => {
  it("renders the registry SVG and applies the brand color for a known logo id", () => {
    const { container } = render(
      <ConnectorIcon connector={{ logo: "slack", brand_color: "#611f69" }} />,
    );
    const el = container.querySelector("[data-logo]") as HTMLElement;
    expect(el).not.toBeNull();
    expect(el.getAttribute("data-logo")).toBe("slack");
    expect(el.querySelector("svg")).not.toBeNull();
    // Brand color comes from the prop, set inline as the --brand custom property.
    expect(el.style.getPropertyValue("--brand")).toBe("#611f69");
  });

  it("falls back to the plug glyph for an unknown id while keeping the provided color", () => {
    const { container } = render(
      <ConnectorIcon connector={{ logo: "does-not-exist", brand_color: "#123456" }} />,
    );
    const el = container.querySelector("[data-logo]") as HTMLElement;
    expect(el.getAttribute("data-logo")).toBe("fallback");
    expect(el.querySelector("svg")).not.toBeNull();
    expect(el.style.getPropertyValue("--brand")).toBe("#123456");
  });

  it("uses the neutral fallback color when none is provided", () => {
    const { container } = render(<ConnectorIcon connector={{ logo: "" }} />);
    const el = container.querySelector("[data-logo]") as HTMLElement;
    expect(el.getAttribute("data-logo")).toBe("fallback");
    expect(el.style.getPropertyValue("--brand")).toBe("#6b7280");
  });
});

describe("ConnectorBadge", () => {
  it("renders a brand-tinted badge with the registry SVG for a known id", () => {
    const { container } = render(
      <ConnectorBadge connector={{ logo: "github", brand_color: "#1f2328" }} />,
    );
    const badge = container.querySelector(".connector-badge") as HTMLElement;
    expect(badge).not.toBeNull();
    expect(badge.getAttribute("data-logo")).toBe("github");
    expect(badge.style.getPropertyValue("--brand")).toBe("#1f2328");
    expect(badge.querySelector("svg")).not.toBeNull();
  });
});
