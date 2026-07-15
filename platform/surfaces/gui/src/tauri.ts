// Thin bridge to the Tauri desktop shell. In the browser these are inert (isTauri() === false),
// so the SPA stays a single codebase. We use the injected `window.__TAURI__` global (the shell
// sets `withGlobalTauri`) instead of the @tauri-apps npm packages, so the browser build needs
// no Tauri dependencies.

export const isTauri = (): boolean =>
  typeof (globalThis as any).__TAURI__ !== "undefined";

export type DictationStatus = {
  recording: boolean;
  model_installed: boolean;
  model_name: string;
};

const invoke = async <T>(cmd: string, args?: Record<string, unknown>): Promise<T | null> => {
  const tauri = (globalThis as any).__TAURI__;
  if (!tauri?.core?.invoke) return null;
  try {
    return (await tauri.core.invoke(cmd, args)) as T;
  } catch {
    return null;
  }
};

/** Open the native macOS folder picker (Tauri only). Returns the chosen path, or null. */
export async function pickFolder(): Promise<string | null> {
  const path = await invoke<string>("pick_folder");
  return typeof path === "string" && path ? path : null;
}

/** The folder picker that works EVERYWHERE: Tauri's native dialog in the desktop shell, else the
 * sidecar-opened OS dialog (the sidecar is local, so the browser GUI still gets a real picker —
 * owner report 2026-07-04: "Browse" was desktop-only and the browser had paste-a-path only). */
export async function chooseFolder(): Promise<string | null> {
  if (isTauri()) return pickFolder();
  const { pickFolderViaServer } = await import("./api");
  return pickFolderViaServer();
}

/** Open-at-login (macOS LaunchAgent). */
export const getAutostart = () => invoke<boolean>("get_autostart");
export const setAutostart = (enabled: boolean) => invoke<boolean>("set_autostart", { enabled });

/** Keep this system awake so scheduled tasks fire while idle (caffeinate on macOS,
 * SetThreadExecutionState on Windows). Persists across restarts. */
export const getKeepAwake = () => invoke<boolean>("get_keep_awake");
export const setKeepAwake = (enabled: boolean) => invoke<boolean>("set_keep_awake", { enabled });

/** Begin native window dragging from a custom title/header region. */
export const startWindowDrag = () => invoke<boolean>("start_window_drag");

// Local dictation is native-only. The browser build deliberately keeps this unavailable rather
// than silently sending microphone audio to a server.
export const getDictationStatus = () => invoke<DictationStatus>("get_dictation_status");
export const startDictation = () => invoke<DictationStatus>("start_dictation");
export const stopDictation = () => invoke<string>("stop_dictation");
export const cancelDictation = () => invoke<void>("cancel_dictation");
export const downloadDictationModel = () => invoke<DictationStatus>("download_dictation_model");

/** Best-effort open a URL in the user's browser. Uses the Tauri opener plugin if present, else
 * `window.open`. The caller should also render the raw URL so it stays copyable if both no-op
 * (the desktop webview has no opener plugin wired yet). */
export function openExternal(url: string): void {
  const opener = (globalThis as any).__TAURI__?.opener;
  if (opener?.openUrl) {
    opener.openUrl(url).catch(() => window.open(url, "_blank", "noopener,noreferrer"));
    return;
  }
  window.open(url, "_blank", "noopener,noreferrer");
}
