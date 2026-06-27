# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the bundled `coworker-server` (desktop sidecar).

One-file binary so it drops into Tauri's externalBin slot. The wrinkles handled here:
  - aisuite isn't pip-installed — it lives at <repo>/aisuite on sys.path via a `.pth`. We add
    both the repo root and platform/ to `pathex` and collect coworker + aisuite submodules.
  - uvicorn loads its protocol/lifespan impls dynamically → collect_all.
  - certifi's CA bundle must ship for TLS (OpenAI, web search, Telegram/Slack).
  - messaging extras (slack_bolt, telegram) are optional; collected if importable.

Cross-platform: paths are derived from this spec's own location (SPECPATH), never hardcoded,
so the same spec builds native binaries on macOS, Windows, and Linux. On Windows PyInstaller
appends `.exe` to `name`. The binary is built as a normal console app on every OS — a windowed
(console=False) build leaves sys.stdout/stderr as None, which breaks uvicorn's startup logging
and hangs the server. To avoid a console window flashing in the desktop app, the Tauri shell
spawns this sidecar with the Windows CREATE_NO_WINDOW flag (see src-tauri/src/lib.rs), which
hides the window while keeping stdio intact.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH is injected by PyInstaller and points at this file's directory
# (<repo>/platform/packaging). Derive everything else from it — no hardcoded paths.
PACKAGING = SPECPATH
PLATFORM = os.path.dirname(PACKAGING)
ROOT = os.path.dirname(PLATFORM)

IS_WINDOWS = sys.platform == "win32"

# Experimental (use-at-your-own-risk) connectors are excluded from official builds: the code
# is stripped, not just disabled. Self-builders opt in with COWORKER_EXPERIMENTAL=1; the
# loader in coworker/connectors/descriptors.py treats the missing package as a no-op.
INCLUDE_EXPERIMENTAL = os.environ.get("COWORKER_EXPERIMENTAL") == "1"

hiddenimports = []
datas = []
binaries = []

for pkg in ("coworker", "aisuite", "mcp", "ddgs", "croniter", "docstring_parser"):
    hiddenimports += collect_submodules(pkg)

if not INCLUDE_EXPERIMENTAL:
    hiddenimports = [
        m for m in hiddenimports if not m.startswith("coworker.connectors.experimental")
    ]
else:
    # experimental builds also need the WhatsApp bridge sources as data files
    _bridge = os.path.join(
        PLATFORM, "coworker", "connectors", "experimental", "whatsapp_bridge"
    )
    for _f in ("bridge.js", "package.json"):
        datas.append(
            (
                os.path.join(_bridge, _f),
                "coworker/connectors/experimental/whatsapp_bridge",
            )
        )

for pkg in ("uvicorn", "certifi", "anyio"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Windows has no system tz database; tzdata ships the zoneinfo files the scheduler needs.
if IS_WINDOWS:
    try:
        d, b, h = collect_all("tzdata")
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

for pkg in ("slack_bolt", "telegram"):  # [messaging] extra — optional
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

a = Analysis(
    [os.path.join(PACKAGING, "server_entry.py")],
    pathex=[ROOT, PLATFORM],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "PyQt5", "PySide6"]
    + ([] if INCLUDE_EXPERIMENTAL else ["coworker.connectors.experimental"]),
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="coworker-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Console on every OS: a windowed build nulls stdout/stderr and hangs uvicorn. The Tauri
    # shell hides the window on Windows via CREATE_NO_WINDOW when spawning the sidecar.
    console=True,
    # target_arch left unset → PyInstaller builds for the host architecture.
)
