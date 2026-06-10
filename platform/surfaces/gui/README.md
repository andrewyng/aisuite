# coworker GUI (React + Tauri)

A thin client of the coworker server (OpenAI-compatible API + WS event/approval stream).
Same codebase runs in a browser (dev) and as a Tauri desktop app.

## Run it (browser, two terminals)

1. **Start the server** (needs `OPENAI_API_KEY` in the environment):
   ```bash
   cd platform
   export OPENAI_API_KEY=sk-...
   ./.venv/bin/coworker-server --cwd /path/to/your/project --port 8765
   ```
2. **Start the UI:**
   ```bash
   cd platform/surfaces/gui
   npm install      # first time
   npm run dev      # → http://localhost:5173
   ```

Open http://localhost:5173. The UI talks to `http://127.0.0.1:8765` (override with
`VITE_COWORKER_HTTP` / `VITE_COWORKER_WS`).

## Desktop (Tauri) — later

The Tauri shell wraps this same app and supervises the Python server as a sidecar. It
requires the Rust toolchain (`cargo`), which isn't installed yet:
```bash
curl https://sh.rustup.rs -sSf | sh   # install Rust, then add the Tauri scaffold
```
Until then, use the browser flow above — it's the identical UI.
