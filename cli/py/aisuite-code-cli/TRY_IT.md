# Try aisuite-code locally

From the repository root:

```bash
./scripts/aisuite-code --cwd /tmp/aisuite-cli-play --viewer
```

If you prefer running from the package directory:

```bash
cd cli/py/aisuite-code-cli
python3 -m poetry install
python3 -m poetry run aisuite-code --cwd /tmp/aisuite-cli-play --viewer
```

Make sure `OPENAI_API_KEY` is set first. If the repo `.env` contains it:

```bash
set -a
source .env
set +a
```

Prompts to try:

```text
List files in this directory and tell me what you see.
```

```text
Create app.py with an add(a, b) function and a small main block that prints add(2, 3). Then run it.
```

```text
Show git status and git diff.
```

```text
Ask the reviewer subagent to review the current changes.
```

Expected behavior:

- File reads and git status/diff are read-only and should not ask for approval.
- File writes and shell commands ask for approval before execution.
- Shell approvals support allowing once, denying, always allowing the tool, or always allowing the exact command for the session.
- `/viewer start` starts the local trace viewer if it was not started at launch.
- `/status` shows model, cwd, trace file, artifact root, and shell command policy.
