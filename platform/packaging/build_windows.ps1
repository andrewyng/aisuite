#requires -Version 5.1
<#
.SYNOPSIS
  Build the Coworker Windows desktop app + NSIS (.exe) and MSI installers.

.DESCRIPTION
  The Windows counterpart to build_dmg.sh:
    1. PyInstaller-bundle the server into a standalone coworker-server.exe (no venv at runtime).
    2. Drop it into Tauri's externalBin slot (binaries\coworker-server-<triple>.exe).
    3. `tauri build --bundles nsis,msi` -> Coworker NSIS setup .exe + .msi (externalBin copied in).

  Prerequisites (see the toolchain notes in the PR/plan):
    - Rust (rustup) with the x86_64-pc-windows-msvc target + the MSVC C++ build tools (link.exe).
    - Node + npm (frontend build).
    - A Python venv at platform\.venv with this package installed editable, plus pyinstaller.
      `typer` is needed only at build time: PyInstaller walks the `mcp` package and `mcp.cli`
      calls sys.exit() at import if typer is absent, which aborts the freeze.
        py -m venv .venv ; .\.venv\Scripts\pip install -e . pyinstaller tzdata typer

  The result is UNSIGNED — first launch shows a SmartScreen warning ("More info" -> "Run anyway").
  Authenticode signing is a later step.
#>
[CmdletBinding()]
param(
    # Which installer bundles to produce. Both by default.
    [string]$Bundles = "nsis,msi"
)
$ErrorActionPreference = "Stop"

$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Platform = Split-Path -Parent $Here
$Gui      = Join-Path $Platform "surfaces\gui"
$Venv     = Join-Path $Platform ".venv"
$PyInst   = Join-Path $Venv "Scripts\pyinstaller.exe"

function Require-Cmd($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required tool '$name' not found on PATH. See the prerequisites in this script's header."
    }
}

Require-Cmd rustc
Require-Cmd npm
if (-not (Test-Path $PyInst)) {
    throw "PyInstaller not found at $PyInst. Create the venv and install deps (see header)."
}

# Host target triple, e.g. x86_64-pc-windows-msvc — Tauri's externalBin suffix.
$Triple = (& rustc -vV | Select-String '^host:').ToString().Split()[-1]
$Arch   = $Triple.Split('-')[0]

# A running coworker-server.exe (e.g. a prior sidecar/smoke test) locks the output exe and
# makes PyInstaller's overwrite fail with Access-is-denied. Stop any before bundling.
$running = Get-Process -Name "coworker-server" -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "==> stopping $($running.Count) running coworker-server process(es) holding the output exe"
    $running | Stop-Process -Force
    Start-Sleep -Seconds 1
}

Write-Host "==> [1/3] PyInstaller: bundling coworker-server ($Triple)" -ForegroundColor Cyan
& $PyInst --noconfirm --clean `
    --distpath (Join-Path $Here "dist") --workpath (Join-Path $Here "build") `
    (Join-Path $Here "coworker-server.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

Write-Host "==> [2/3] staging externalBin" -ForegroundColor Cyan
$BinDir = Join-Path $Gui "src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Src = Join-Path $Here "dist\coworker-server.exe"
$Dst = Join-Path $BinDir "coworker-server-$Triple.exe"
Copy-Item -Force $Src $Dst
Write-Host "    -> $Dst"

Write-Host "==> [3/3] tauri build (--bundles $Bundles)" -ForegroundColor Cyan
Push-Location $Gui
try {
    & npm run tauri build -- --bundles $Bundles
    if ($LASTEXITCODE -ne 0) { throw "tauri build failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

$BundleDir = Join-Path $Gui "src-tauri\target\release\bundle"
Write-Host ""
Write-Host "Done. Installers under: $BundleDir" -ForegroundColor Green
Get-ChildItem -Path $BundleDir -Recurse -Include *.exe, *.msi -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  $($_.FullName)" }
