# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the bundled `coworker-server` (desktop sidecar).

One-file binary so it drops into Tauri's externalBin slot. The wrinkles handled here:
  - aisuite isn't pip-installed — it lives at <repo>/aisuite on sys.path via a `.pth`. We add
    both the repo root and platform/ to `pathex` and collect coworker + aisuite submodules.
  - uvicorn loads its protocol/lifespan impls dynamically → collect_all.
  - certifi's CA bundle must ship for TLS (OpenAI, web search, Telegram/Slack).
  - messaging extras (slack_bolt, telegram) are optional; collected if importable.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = "/Users/rohit/fleet/ro4d/agent-platform"
PLATFORM = ROOT + "/platform"

hiddenimports = []
datas = []
binaries = []

for pkg in ("coworker", "aisuite", "mcp", "ddgs", "croniter", "docstring_parser"):
    hiddenimports += collect_submodules(pkg)

for pkg in ("uvicorn", "certifi", "anyio"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in ("slack_bolt", "telegram"):  # [messaging] extra — optional
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

a = Analysis(
    [PLATFORM + "/packaging/server_entry.py"],
    pathex=[ROOT, PLATFORM],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "PyQt5", "PySide6"],
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
    console=True,
    target_arch="arm64",
)
