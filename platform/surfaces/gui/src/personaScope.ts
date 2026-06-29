// A persona is "project-scoped" when its workspace is a real folder the user picks: a git repo
// (`git`, like Code) or a project directory (`project`, like Ops). Project-scoped personas prompt
// for a folder on New session AND group their sessions by project in the sidebar. Scratch/flat
// personas (`deliverable`, `none` — Cowork, Chat) don't. The `family === "code"` fallback keeps
// Code behaving correctly in the window before the persona list has loaded.
export function isProjectScoped(p?: { workspace?: string; family?: string }): boolean {
  return p?.workspace === "git" || p?.workspace === "project" || p?.family === "code";
}
