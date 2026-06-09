//! Coworker desktop shell.
//!
//! Tauri is a thin native window over the existing React SPA. It:
//!   1. picks a free localhost port and starts the Python `coworker-server` as a managed
//!      sidecar on that port (so it never clashes with a hand-run server on 8765);
//!   2. injects `window.__COWORKER_HTTP__` / `__COWORKER_WS__` before the SPA loads, so
//!      `api.ts` talks to the sidecar (single codebase — the browser build still hits 8765);
//!   3. lives in the system tray: closing the window hides it (keeps MyHelper + the scheduler
//!      running); only tray → Quit stops the sidecar;
//!   4. exposes native commands: folder picker, autostart (open-at-login), and keep-awake
//!      (caffeinate, so scheduled tasks fire while the Mac is idle).
//!
//! The sidecar inherits this process's environment, so a shell-launched `npm run tauri dev`
//! passes `OPENAI_API_KEY` through. A Finder-launched app has no shell env — there the key
//! comes from the SecretStore (Settings tab), see `coworker.providers.resolve_api_key`.

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_autostart::ManagerExt;

/// The sidecar server child — killed on exit (orphaned servers have bitten us before).
struct ServerProcess(Mutex<Option<Child>>);
/// The `caffeinate` child while keep-awake is on (None when off).
struct KeepAwake(Mutex<Option<Child>>);

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8765)
}

/// Path to the server entrypoint. Resolution order:
///   1. `COWORKER_SERVER_BIN` env override.
///   2. The bundled sidecar next to the app executable (production — Tauri externalBin drops
///      `coworker-server` into Contents/MacOS).
///   3. Dev fallback: the repo venv, relative to this crate (`src-tauri` → `platform/.venv`).
fn server_bin() -> PathBuf {
    if let Ok(p) = std::env::var("COWORKER_SERVER_BIN") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let bundled = dir.join("coworker-server");
            if bundled.exists() {
                return bundled;
            }
        }
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../../.venv/bin/coworker-server");
    p
}

/// Mirror of `coworker.secrets.state_dir()` so the shell and server agree on `desktop.json`.
fn state_dir() -> PathBuf {
    if let Ok(d) = std::env::var("COWORKER_STATE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config").join("coworker")
}

fn desktop_prefs_path() -> PathBuf {
    state_dir().join("desktop.json")
}

fn read_keep_awake_pref() -> bool {
    std::fs::read_to_string(desktop_prefs_path())
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("keep_awake").and_then(|b| b.as_bool()))
        .unwrap_or(false)
}

fn write_keep_awake_pref(enabled: bool) {
    let path = desktop_prefs_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, serde_json::json!({ "keep_awake": enabled }).to_string());
}

/// Hold off idle + system sleep so the scheduler keeps firing. macOS `caffeinate` is built-in.
fn start_caffeinate() -> Option<Child> {
    Command::new("caffeinate").args(["-i", "-s"]).spawn().ok()
}

// -- native commands (invoked from the SPA via window.__TAURI__.core.invoke) -----------------

/// Native macOS folder picker for the workspace gate.
#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog().file().pick_folder(move |p| {
        let _ = tx.send(p);
    });
    rx.recv().ok().flatten().map(|fp| fp.to_string())
}

#[tauri::command]
fn get_autostart(app: tauri::AppHandle) -> bool {
    app.autolaunch().is_enabled().unwrap_or(false)
}

#[tauri::command]
fn set_autostart(app: tauri::AppHandle, enabled: bool) -> bool {
    let m = app.autolaunch();
    let _ = if enabled { m.enable() } else { m.disable() };
    m.is_enabled().unwrap_or(false)
}

#[tauri::command]
fn get_keep_awake(state: tauri::State<KeepAwake>) -> bool {
    state.0.lock().unwrap().is_some()
}

#[tauri::command]
fn set_keep_awake(state: tauri::State<KeepAwake>, enabled: bool) -> bool {
    let mut guard = state.0.lock().unwrap();
    if enabled {
        if guard.is_none() {
            *guard = start_caffeinate();
        }
    } else if let Some(mut child) = guard.take() {
        let _ = child.kill();
    }
    let on = guard.is_some();
    write_keep_awake_pref(on);
    on
}

#[tauri::command]
fn start_window_drag(window: tauri::WebviewWindow) -> bool {
    window.start_dragging().is_ok()
}

fn show_main(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.set_focus();
    }
}

pub fn run() {
    let port = free_port();
    let http = format!("http://127.0.0.1:{port}");
    let ws = format!("ws://127.0.0.1:{port}");
    // Debug-format yields a quoted JS string literal.
    let inject = format!("window.__COWORKER_HTTP__={http:?};window.__COWORKER_WS__={ws:?};");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .invoke_handler(tauri::generate_handler![
            pick_folder,
            get_autostart,
            set_autostart,
            get_keep_awake,
            set_keep_awake,
            start_window_drag
        ])
        .setup(move |app| {
            // 1. Start the Python server sidecar on the chosen port (inherits our env).
            let child = match Command::new(server_bin())
                .args(["--host", "127.0.0.1", "--port", &port.to_string()])
                // The sidecar self-exits if we die abruptly (dev-watcher restart, crash) —
                // belt-and-suspenders alongside the RunEvent::ExitRequested kill below.
                .env("COWORKER_EXIT_WITH_PARENT", "1")
                .spawn()
            {
                Ok(child) => Some(child),
                Err(e) => {
                    eprintln!("[coworker] failed to start server sidecar: {e}");
                    None
                }
            };
            app.manage(ServerProcess(Mutex::new(child)));

            // Restore keep-awake from the last session.
            let ka = if read_keep_awake_pref() {
                start_caffeinate()
            } else {
                None
            };
            app.manage(KeepAwake(Mutex::new(ka)));

            // 2. Build the window, injecting the sidecar endpoints before the SPA loads.
            //    Overlay title bar (macOS): traffic lights float over the edge-to-edge UI.
            let mut builder =
                WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                    .title("Coworker")
                    .inner_size(1360.0, 900.0)
                    .min_inner_size(980.0, 640.0)
                    .initialization_script(&inject);
            #[cfg(target_os = "macos")]
            {
                builder = builder
                    .title_bar_style(tauri::TitleBarStyle::Overlay)
                    .hidden_title(true);
            }
            let win = builder.build()?;

            // Close-to-tray: hide instead of quitting so the sidecar keeps running.
            let w = win.clone();
            win.on_window_event(move |event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    let _ = w.hide();
                    api.prevent_close();
                }
            });

            // 3. System tray: Open / Settings / Quit.
            let open_i = MenuItem::with_id(app, "open", "Open Coworker", true, None::<&str>)?;
            let settings_i = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_i, &settings_i, &quit_i])?;

            // A monochrome template icon (black + alpha, raw RGBA 44×44) so the menu bar tints
            // it for light/dark automatically — not the full-color app icon.
            let tray_icon = tauri::image::Image::new(include_bytes!("../icons/tray.rgba"), 44, 44);
            TrayIconBuilder::new()
                .tooltip("Coworker")
                .icon(tray_icon)
                .icon_as_template(true)
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main(app),
                    "settings" => {
                        show_main(app);
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.eval(
                                "window.dispatchEvent(new CustomEvent('coworker:open-settings'))",
                            );
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Coworker desktop app")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app.try_state::<ServerProcess>() {
                    if let Some(mut child) = state.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
                if let Some(state) = app.try_state::<KeepAwake>() {
                    if let Some(mut child) = state.0.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        });
}
