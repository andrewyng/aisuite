/** Tailwind config — mirrors platform/ui-mocks/redesign.html so the app can use the mock's
 *  exact utility classes. Colors map to the CSS custom properties already defined in styles.css
 *  (so light/dark theming flows through one source of truth). */
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "var(--paper)",
        panel: "var(--panel)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        faint: "var(--faint)",
        line: "var(--line)",
        lineStrong: "var(--line-strong)",
        accent: "var(--accent)",
        accentSoft: "var(--accent-soft)",
        ok: "var(--ok)",
        okSoft: "var(--ok-soft)",
        okLine: "var(--ok-line)",
        warnInk: "var(--warn-ink)",
        warnSoft: "var(--warn-soft)",
        danger: "var(--danger)",
        tealInk: "var(--teal-ink)",
        tealSoft: "var(--teal-soft)",
        tealLine: "var(--teal-line)",
        solid: "var(--solid)",
        onSolid: "var(--on-solid)",
      },
      fontFamily: {
        sans: ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Inter", "system-ui", "sans-serif"],
        mono: ["SF Mono", "JetBrains Mono", "Menlo", "monospace"],
      },
      borderRadius: { xl2: "14px" },
    },
  },
  plugins: [],
};
