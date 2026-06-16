#!/usr/bin/env bash
# Build the Linux desktop app + .deb and .AppImage installers.
#
#   1. PyInstaller-bundle the server into a standalone binary (no venv needed at runtime).
#   2. Drop it into Tauri's externalBin slot (binaries/coworker-server-<triple>).
#   3. `tauri build` → .deb and .AppImage bundles (externalBin is copied in).
#
# Prerequisites (Ubuntu/Debian — adapt package names for other distros):
#   sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
#     libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev \
#     nodejs npm
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#   source ~/.cargo/env
#
# A Python venv at platform/.venv with this package installed editable, plus
# pyinstaller and typer:
#   python3 -m venv .venv
#   .venv/bin/pip install -e . pyinstaller 'mcp[cli]'
#
# `typer` (pulled in by mcp[cli]) is needed only at build time: PyInstaller
# walks the `mcp` package and `mcp.cli` calls sys.exit() at import if typer
# is absent, which aborts the freeze.
#
# Experimental (use-at-your-own-risk) connectors are EXCLUDED from this build
# by default — the spec strips coworker.connectors.experimental. Self-builders
# can opt in with:
#   COWORKER_EXPERIMENTAL=1 ./build_linux.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="$(cd "$HERE/.." && pwd)"
GUI="$PLATFORM/surfaces/gui"

# Single source of truth for the version: tauri.conf.json (also stamps the bundle).
VERSION="$(node -p "require('$GUI/src-tauri/tauri.conf.json').version")"
TRIPLE="$(rustc -vV | sed -n 's/host: //p')"   # e.g. x86_64-unknown-linux-gnu

echo "==> [1/3] PyInstaller: bundling coworker-server ($TRIPLE)"
"$PLATFORM/.venv/bin/pyinstaller" --noconfirm --clean \
  --distpath "$HERE/dist" --workpath "$HERE/build" "$HERE/coworker-server.spec"

echo "==> [2/3] staging externalBin"
mkdir -p "$GUI/src-tauri/binaries"
cp "$HERE/dist/coworker-server" "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"
chmod +x "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"

echo "==> [3/3] tauri build (.deb / .AppImage)"
( cd "$GUI" && npm run tauri build )

BUNDLE="$GUI/src-tauri/target/release/bundle"
echo ""
echo "Done. Installers under: $BUNDLE"
find "$BUNDLE" \( -name "*.deb" -o -name "*.AppImage" -o -name "*.rpm" \) \
  2>/dev/null | while read -r f; do echo "  $f"; done
