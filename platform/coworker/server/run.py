"""Launch the server with uvicorn. Used by the desktop GUI sidecar and `coworker-server`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..config import load_config
from ..permissions import Mode
from ..secrets import state_dir
from .app import create_app
from .manager import SessionManager


def _exit_when_orphaned() -> None:
    """When launched as a desktop sidecar (`COWORKER_EXIT_WITH_PARENT=1`), exit if the parent
    process dies — even on an abrupt kill (e.g. the Tauri dev watcher restarting the app, or a
    crash) that skips the shell's graceful child-kill. Standalone `coworker-server` runs are
    unaffected.

    POSIX: detected via re-parenting — when our parent dies, getppid() flips to 1 (init/launchd).
    Windows: there is no re-parenting and getppid() keeps returning the stale PID, so instead we
    block on a handle to the parent process and exit the moment it signals (i.e. the parent
    exited)."""
    if os.environ.get("COWORKER_EXIT_WITH_PARENT") != "1":
        return
    import threading

    if sys.platform == "win32":
        _watch_parent_windows()
        return

    import time

    original = os.getppid()

    def watch() -> None:
        while True:
            time.sleep(1.5)
            if os.getppid() != original:
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def _watch_parent_windows() -> None:
    """Block on a handle to the parent process; exit when it dies. Best-effort — any failure
    leaves the parent's RunEvent::ExitRequested kill as the primary cleanup path."""
    import ctypes
    import threading

    SYNCHRONIZE = 0x0010_0000
    INFINITE = 0xFFFF_FFFF
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, os.getppid())
    if not handle:
        return

    def watch() -> None:
        kernel32.WaitForSingleObject(handle, INFINITE)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def build_app(workspace: str | None, model: str, mode: str):
    manager = SessionManager(
        workspace=Path(workspace).expanduser().resolve() if workspace else None,
        data_dir=state_dir(),
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
