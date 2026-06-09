"""Launch the server with uvicorn. Used by the desktop GUI sidecar and `coworker-server`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..config import load_config
from ..permissions import Mode
from .app import create_app
from .manager import SessionManager


def _exit_when_orphaned() -> None:
    """When launched as a desktop sidecar (`COWORKER_EXIT_WITH_PARENT=1`), exit if the parent
    process dies — even on an abrupt SIGKILL (e.g. the Tauri dev watcher restarting the app),
    which skips the shell's graceful child-kill. Detected via re-parenting: when our parent
    dies, getppid() flips to 1 (launchd). Standalone `coworker-server` runs are unaffected."""
    if os.environ.get("COWORKER_EXIT_WITH_PARENT") != "1":
        return
    import threading
    import time

    original = os.getppid()

    def watch() -> None:
        while True:
            time.sleep(1.5)
            if os.getppid() != original:
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def build_app(workspace: str | None, model: str, mode: str):
    manager = SessionManager(
        workspace=Path(workspace).expanduser().resolve() if workspace else None,
        data_dir=Path.home() / ".config" / "coworker",
        model=model,
        mode=Mode(mode),
    )
    return create_app(manager)


def main(argv=None) -> None:
    cfg = load_config()  # global config supplies defaults
    parser = argparse.ArgumentParser(prog="coworker-server")
    parser.add_argument("--cwd", default=None, help="optional seed/default workspace")
    parser.add_argument("--model", default=cfg.model)
    parser.add_argument("--mode", default=cfg.mode, choices=["plan", "interactive", "auto"])
    parser.add_argument("--host", default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    args = parser.parse_args(argv)

    import uvicorn

    _exit_when_orphaned()
    app = build_app(args.cwd, args.model, args.mode)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
