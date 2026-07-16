# aisuite-code

`aisuite-code` is a local coding agent CLI built on the aisuite agent framework.
It is intentionally small for now: a conversation loop, cwd-scoped file tools,
approval-gated shell/write tools, and trace output that can be inspected in the
aisuite runs viewer.

Install and run from this package directory:

```bash
cd cli/py/aisuite-code-cli
python3 -m poetry install
python3 -m poetry run aisuite-code --cwd /path/to/project
```

From the repository root, use the convenience shim:

```bash
./scripts/aisuite-code --cwd /path/to/project
```

The CLI package depends on the OpenAI provider by default because the default
model is `openai:gpt-4o-mini`. Set `OPENAI_API_KEY` in your shell before running
the CLI, or source the repository `.env` file first.

Useful commands inside the CLI:

```text
/help
/viewer
/viewer start
/status
/clear
/exit
```

By default the CLI can read and write within `--cwd`, but write operations still
require approval. Shell commands are limited to common coding commands unless
`--allow-shell-all` is supplied.

The main agent also has a read-only reviewer subagent available as
`review_changes(input: str)`. It can call the reviewer when the user asks for a
review or when a second opinion would help after substantial edits. The reviewer
subagent can inspect files, but it cannot edit files or run commands. Use
`--no-reviewer` to disable it.

For a short hands-on flow, see [TRY_IT.md](TRY_IT.md).
