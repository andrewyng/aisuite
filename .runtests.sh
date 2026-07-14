#!/bin/zsh
# Local test runner: agent-platform venv interpreter + this worktree on PYTHONPATH.
ROOT=/Users/rohit/fleet/ro4d/aisuite-personas
VENV=/Users/rohit/fleet/ro4d/agent-platform/platform/.venv/bin/python
cd "$ROOT"
PYTHONPATH="$ROOT/platform:$ROOT" "$VENV" -m pytest "$@"
