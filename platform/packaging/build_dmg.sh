#!/usr/bin/env bash
# Build the macOS desktop app + a drag-to-install .dmg.
#
#   1. PyInstaller-bundle the server into a standalone binary (no venv needed at runtime).
#   2. Drop it into Tauri's externalBin slot (binaries/coworker-server-<triple>).
#   3. `tauri build --bundles app` → OpenCoworker.app (the externalBin is copied in).
#   4. Wrap the .app in a compressed .dmg via hdiutil (reliable + headless; Tauri's own
#      bundle_dmg.sh uses Finder AppleScript and fails in non-interactive sessions).
#
# Prerequisites (mirrors build_windows.ps1's header):
#   - Rust (rustup) + Node/npm, and the GUI deps installed (npm ci in surfaces/gui).
#   - A Python venv at platform/.venv with this package installed editable, plus the
#     build-only deps:
#       python3 -m venv platform/.venv
#       platform/.venv/bin/pip install -e . pyinstaller tzdata typer
#     `typer` is needed only at BUILD time: PyInstaller walks the `mcp` package and
#     `mcp.cli` calls sys.exit() at import if typer is absent, which aborts the freeze.
#     (aisuite is not pip-installed — the spec adds <repo> to pathex so PyInstaller finds it.)
#
# SIGNING: set APPLE_SIGNING_IDENTITY to a "Developer ID Application: … (TEAMID)" identity and
# `tauri build` signs the .app + the bundled sidecar with it. Left unset → UNSIGNED (first launch
# needs right-click → Open). Either way this script does NOT notarize; to finish a public release:
#   xcrun notarytool submit "<dmg>" --keychain-profile <profile> --wait   # profile via store-credentials
#   xcrun stapler staple "<dmg>"
#
# Experimental (use-at-your-own-risk) connectors are EXCLUDED from this build by default —
# the spec strips coworker.connectors.experimental. Self-builders can opt in with:
#   COWORKER_EXPERIMENTAL=1 ./build_dmg.sh
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
# Background art (arrow + "drag to Applications") — hidden folder Finder reads for the window.
# A HiDPI TIFF (1x + native 2x reps) so text/arrow stay crisp on Retina; a plain 1x PNG would
# be upscaled and look hazy/pixelated.
mkdir "$STAGING/.background"
cp "$HERE/dmg-background.tiff" "$STAGING/.background/bg.tiff"
DMG="$BUNDLE/dmg/${APP}_${VERSION}_${ARCH}.dmg"
mkdir -p "$(dirname "$DMG")"
rm -f "$DMG"

# A styled install window (fixed size, icons in place, arrow background) instead of Finder's
# default oversized bare window. Needs Finder (AppleScript); if it isn't available (headless CI),
# fall back to the plain compressed image so the build still produces a working .dmg.
style_dmg() {
  local rw; rw="$(mktemp -u).dmg"
  hdiutil create -volname "$APP" -srcfolder "$STAGING" -fs HFS+ -format UDRW -ov "$rw" >/dev/null
  local info dev mnt
  info="$(hdiutil attach -readwrite -noverify -noautoopen "$rw")"
  dev="$(echo "$info" | grep -Eo '^/dev/disk[0-9]+' | head -1)"
  mnt="$(echo "$info" | grep -Eo '/Volumes/.*$' | head -1)"
  [ -n "$dev" ] && [ -n "$mnt" ] || return 1
  sleep 1
  # Icons at y≈205 to sit on the background's arrow: app left of it, Applications right. The
  # background is set via its POSIX path (the HFS `file ".background:bg.png"` form errors -10006).
  osascript <<OSA || { hdiutil detach "$dev" >/dev/null 2>&1 || true; return 1; }
tell application "Finder"
  tell disk "$APP"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 120, 840, 543}
    set opts to the icon view options of container window
    set arrangement of opts to not arranged
    set icon size of opts to 96
    set text size of opts to 12
    set background picture of opts to POSIX file "$mnt/.background/bg.tiff"
    set position of item "$APP.app" of container window to {172, 190}
    set position of item "Applications" of container window to {468, 190}
    update without registering applications
    delay 1
    close
  end tell
end tell
OSA
  sync
  hdiutil detach "$dev" >/dev/null
  hdiutil convert "$rw" -format UDZO -imagekey zlib-level=9 -o "$DMG" >/dev/null
  rm -f "$rw"
}

if ! style_dmg; then
  echo "    (Finder styling unavailable — writing a plain .dmg)"
  hdiutil create -volname "$APP" -srcfolder "$STAGING" -ov -format UDZO "$DMG" >/dev/null
fi
rm -rf "$STAGING"

echo ""
echo "Done → $DMG"
