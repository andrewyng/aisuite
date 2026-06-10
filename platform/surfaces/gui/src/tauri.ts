// Thin bridge to the Tauri desktop shell. In the browser these are inert (isTauri() === false),
// so the SPA stays a single codebase. We use the injected `window.__TAURI__` global (the shell
// sets `withGlobalTauri`) instead of the @tauri-apps npm packages, so the browser build needs
// no Tauri dependencies.

export const isTauri = (): boolean =>
  typeof (globalThis as any).__TAURI__ !== "undefined";

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

/** Open-at-login (macOS LaunchAgent). */
export const getAutostart = () => invoke<boolean>("get_autostart");
export const setAutostart = (enabled: boolean) => invoke<boolean>("set_autostart", { enabled });

/** Keep this system awake so scheduled tasks fire while idle (caffeinate on macOS,
 * SetThreadExecutionState on Windows). Persists across restarts. */
export const getKeepAwake = () => invoke<boolean>("get_keep_awake");
export const setKeepAwake = (enabled: boolean) => invoke<boolean>("set_keep_awake", { enabled });

/** Begin native window dragging from a custom title/header region. */
export const startWindowDrag = () => invoke<boolean>("start_window_drag");
