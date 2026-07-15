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
# needs right-click → Open).
#
# NOTARIZATION (step 5, runs only when the identity is set): signs the .dmg CONTAINER, submits
# to Apple's notary service, staples the ticket, and verifies with spctl. Signing alone is NOT
# enough for public downloads — un-notarized apps get macOS's "Apple could not verify… Move to
# Trash?" dialog. Auth is an App Store Connect API key via NOTARYTOOL_API_KEY_PATH /
# NOTARYTOOL_API_KEY_ID / NOTARYTOOL_API_ISSUER_ID — exported, or in $OCW_NOTARY_ENV, or in
# `.ocw-notary.env` one directory ABOVE the repo (shared by every clone/worktree on a machine,
# never committed). Vars missing → the DMG is still produced, with a loud warning.
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

echo "==> [1/5] PyInstaller: bundling coworker-server ($TRIPLE)"
"$PLATFORM/.venv/bin/pyinstaller" --noconfirm --clean \
  --distpath "$HERE/dist" --workpath "$HERE/build" "$HERE/coworker-server.spec"

echo "==> [2/5] staging externalBin"
mkdir -p "$GUI/src-tauri/binaries"
# rm first: cp WRITES THROUGH a symlink at the destination. A dev-convenience symlink left in
# this slot once routed the fresh binary into another worktree's venv, clobbering its console
# script (caught 2026-07-11) — the bundle stayed correct only by accident.
rm -f "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"
cp "$HERE/dist/coworker-server" "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"
chmod +x "$GUI/src-tauri/binaries/coworker-server-$TRIPLE"

echo "==> [3/5] tauri build (.app)"
( cd "$GUI" && npm run tauri build -- --bundles app )

echo "==> [4/5] hdiutil: wrapping into .dmg"
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
#
# Two hard-won correctness points (both caused a *silently* unstyled .dmg before):
#   1. A stale "$APP" volume already mounted → our RW image mounts as "$APP 1", and a hardcoded
#      `tell disk "$APP"` then styles the WRONG (stale) volume, so our image never gets a
#      .DS_Store. Detach any pre-existing mount first, and target the ACTUAL mounted name.
#   2. Finder writes .DS_Store asynchronously — detaching too soon drops it. Poll until it lands.
style_dmg() {
  # Clear any earlier mount of this volume so we don't collide into "$APP 1".
  [ -d "/Volumes/$APP" ] && hdiutil detach "/Volumes/$APP" -force >/dev/null 2>&1 || true
  local rw; rw="$(mktemp -u).dmg"
  hdiutil create -volname "$APP" -srcfolder "$STAGING" -fs HFS+ -format UDRW -ov "$rw" >/dev/null
  local info dev mnt vol
  info="$(hdiutil attach -readwrite -noverify -noautoopen "$rw")"
  dev="$(echo "$info" | grep -Eo '^/dev/disk[0-9]+' | head -1)"
  mnt="$(echo "$info" | grep -Eo '/Volumes/.*$' | head -1)"
  [ -n "$dev" ] && [ -n "$mnt" ] || return 1
  vol="$(basename "$mnt")"   # the real mounted name — what `tell disk` must target
  sleep 1
  # Icons at y≈190 to sit on the background's arrow: app left of it, Applications right. Background
  # via the relative HFS path (`file ".background:bg.tiff"`) so the alias survives a rename; the
  # close→open→update dance forces Finder to actually write the .DS_Store.
  osascript <<OSA || { hdiutil detach "$dev" -force >/dev/null 2>&1 || true; return 1; }
tell application "Finder"
  tell disk "$vol"
    open
    delay 1
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 120, 840, 543}
    set opts to the icon view options of container window
    set arrangement of opts to not arranged
    set icon size of opts to 96
    set text size of opts to 12
    set background picture of opts to file ".background:bg.tiff"
    set position of item "$APP.app" of container window to {172, 190}
    set position of item "Applications" of container window to {468, 190}
    close
    open
    update without registering applications
    delay 3
  end tell
end tell
OSA
  # Wait for Finder to flush .DS_Store into the image (else the layout is lost).
  local i; for i in $(seq 1 15); do [ -f "$mnt/.DS_Store" ] && break; sleep 1; done
  [ -f "$mnt/.DS_Store" ] || { hdiutil detach "$dev" -force >/dev/null 2>&1 || true; return 1; }
  sync; sync
  hdiutil detach "$dev" -force >/dev/null
  hdiutil convert "$rw" -format UDZO -imagekey zlib-level=9 -o "$DMG" >/dev/null
  rm -f "$rw"
}

if ! style_dmg; then
  echo "    (Finder styling unavailable — writing a plain .dmg)"
  hdiutil create -volname "$APP" -srcfolder "$STAGING" -ov -format UDZO "$DMG" >/dev/null
fi
rm -rf "$STAGING"

if [ -n "${APPLE_SIGNING_IDENTITY:-}" ]; then
  echo "==> [5/5] release finishing: sign container → notarize → staple"
  codesign --sign "$APPLE_SIGNING_IDENTITY" --timestamp "$DMG"

  REPO="$(cd "$PLATFORM/.." && pwd)"
  NOTARY_ENV="${OCW_NOTARY_ENV:-$REPO/../.ocw-notary.env}"
  if [ -z "${NOTARYTOOL_API_KEY_PATH:-}" ] && [ -f "$NOTARY_ENV" ]; then
    set -a; # shellcheck disable=SC1090
    source "$NOTARY_ENV"; set +a
  fi
  if [ -n "${NOTARYTOOL_API_KEY_PATH:-}" ] && [ -n "${NOTARYTOOL_API_KEY_ID:-}" ] \
     && [ -n "${NOTARYTOOL_API_ISSUER_ID:-}" ]; then
    xcrun notarytool submit "$DMG" \
      --key "$NOTARYTOOL_API_KEY_PATH" \
      --key-id "$NOTARYTOOL_API_KEY_ID" \
      --issuer "$NOTARYTOOL_API_ISSUER_ID" \
      --wait
    xcrun stapler staple "$DMG"
    # The same check Gatekeeper runs on download — fail the build rather than ship a
    # DMG that greets users with the "Move to Trash" malware dialog.
    spctl -a -t open --context context:primary-signature "$DMG"
    echo "    Gatekeeper: accepted (notarized + stapled)"
  else
    echo "    WARNING: DMG is signed but NOT notarized — public downloads will see the"
    echo "    'Move to Trash' dialog. Provide NOTARYTOOL_API_KEY_PATH/_KEY_ID/_ISSUER_ID"
    echo "    (env, \$OCW_NOTARY_ENV, or $NOTARY_ENV)."
  fi
else
  echo "    (unsigned dev build — set APPLE_SIGNING_IDENTITY for a distributable DMG)"
fi

echo ""
echo "Done → $DMG"
