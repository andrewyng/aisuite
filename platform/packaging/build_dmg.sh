#!/usr/bin/env bash
# Build the macOS desktop app + a drag-to-install .dmg.
#
#   1. PyInstaller-bundle the server into a standalone binary (no venv needed at runtime).
#   2. Drop it into Tauri's externalBin slot (binaries/coworker-server-<triple>).
#   3. `tauri build --bundles app` → OpenCoworker.app (the externalBin is copied in).
#   4. Wrap the .app in a compressed .dmg via hdiutil (reliable + headless; Tauri's own
#      bundle_dmg.sh uses Finder AppleScript and fails in non-interactive sessions).
#
# The result is UNSIGNED — first launch needs right-click → Open (Gatekeeper). Real
# code-signing + notarization is a later step.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="$(cd "$HERE/.." && pwd)"
GUI="$PLATFORM/surfaces/gui"
APP="OpenCoworker"
# Single source of truth for the version: tauri.conf.json (also stamps the bundle).
VERSION="$(node -p "require('$GUI/src-tauri/tauri.conf.json').version")"
TRIPLE="$(rustc -vV | sed -n 's/host: //p')"   # e.g. aarch64-apple-darwin
ARCH="${TRIPLE%%-*}"

echo "==> [1/4] PyInstaller: bundling coworker-server ($TRIPLE)"
"$PLATFORM/.venv/bin/pyinstaller" --noconfirm --clean \
  --distpath "$HERE/dist" --workpath "$HERE/build" "$HERE/coworker-server.spec"

echo "==> [2/4] staging externalBin"
mkdir -p "$GUI/src-tauri/binaries"
cp "$HERE/dist/coworker-server" "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"
chmod +x "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"

echo "==> [3/4] tauri build (.app)"
( cd "$GUI" && npm run tauri build -- --bundles app )

echo "==> [4/4] hdiutil: wrapping into .dmg"
BUNDLE="$GUI/src-tauri/target/release/bundle"
STAGING="$(mktemp -d)"
cp -R "$BUNDLE/macos/$APP.app" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
DMG="$BUNDLE/dmg/${APP}_${VERSION}_${ARCH}.dmg"
mkdir -p "$(dirname "$DMG")"
rm -f "$DMG"
hdiutil create -volname "$APP" -srcfolder "$STAGING" -ov -format UDZO "$DMG"
rm -rf "$STAGING"

echo ""
echo "Done → $DMG"
