// A persona is "project-scoped" when its workspace is a real folder the user picks: a git repo
// (`git`, like Code) or a project directory (`project`, like Ops). Project-scoped personas prompt
// for a folder on New session AND group their sessions by project in the sidebar. Scratch/flat
// personas (`deliverable`, `none` — Cowork, Chat) don't. The `family === "code"` fallback keeps
// Code behaving correctly in the window before the persona list has loaded.
export function isProjectScoped(p?: { workspace?: string; family?: string }): boolean {
  return p?.workspace === "git" || p?.workspace === "project" || p?.family === "code";
}

// Persona naming: the product is "OpenCoworker"; the personas are a "Coworker" family — Coworker
// (general), Code Coworker, Ops Coworker. In lists/chrome we use the SHORT label (Coworker / Code /
// Ops); the persona detail page uses the FULL family name. Backend names are left untouched (the
// API + tests keep "OpenCoworker" / "Ops Coworker"); this is purely the display layer.

// Short label for the sidebar + top bar: "Coworker" / "Code" / "Ops" / "Chat".
export function shortPersonaName(name?: string, id?: string): string {
  if (id === "cowork") return "Coworker";
  const n = (name || id || "").trim();
  return n.replace(/\s*coworker$/i, "").trim() || n;
}

// Full family name for the persona detail page: "Coworker" / "Code Coworker" / "Ops Coworker".
// Chat isn't a coworker — left as-is.
export function fullPersonaName(name?: string, id?: string): string {
  if (id === "cowork") return "Coworker";
  const n = (name || id || "").trim();
  if (id === "chat" || !n) return n;
  return /coworker$/i.test(n) ? n : `${n} Coworker`;
}
