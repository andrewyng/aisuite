from __future__ import annotations

import aisuite as ai

from .config import CliConfig


def build_agent(config: CliConfig) -> ai.Agent:
    tools = _coding_tools(config)
    if config.enable_reviewer:
        tools.append(
            ai.agent_tool(
                build_reviewer_agent(config),
                name="review_changes",
                description=(
                    "Ask the reviewer subagent to inspect the current work for "
                    "bugs, regressions, missing tests, and maintainability risks."
                ),
            )
        )
    return ai.Agent(
        name="aisuite_code",
        model=config.model,
        instructions=_main_agent_instructions(config),
        tools=tools,
        tags=["cli", "code"],
        metadata={"app": "aisuite_code_cli"},
    )


def build_reviewer_agent(config: CliConfig) -> ai.Agent:
    return ai.Agent(
        name="reviewer",
        model=config.model,
        instructions=(
            "You are a code reviewer subagent for aisuite-code. Review the "
            "user-described work and available project files for correctness, "
            "regressions, missing tests, security risks, and maintainability "
            "issues. Do not edit files. Do not run commands. Prefer specific "
            "findings with file paths and line references when available. If "
            "you find no material issues, say that clearly and mention any "
            "residual test gaps. Keep the response concise."
        ),
        tools=ai.toolkits.files(root=config.cwd, allow_write=False),
        tags=["cli", "code", "reviewer"],
        metadata={"app": "aisuite_code_cli", "role": "reviewer"},
    )


def _coding_tools(config: CliConfig) -> list:
    return [
        *ai.toolkits.files(
            root=config.cwd,
            allow_write=config.allow_write,
        ),
        *ai.toolkits.git(root=config.cwd),
        *ai.toolkits.shell(
            cwd=config.cwd,
            allowed_commands=config.allowed_commands,
            allow_all=config.allow_shell_all,
        ),
    ]


def _main_agent_instructions(config: CliConfig) -> str:
    reviewer_instruction = (
        " When the user asks for a review, or after substantial edits when a "
        "second opinion would help, call review_changes with a concise summary "
        "of the work, changed files, and verification results."
        if config.enable_reviewer
        else ""
    )
    return (
        "You are aisuite-code, a concise local coding agent. Work inside the "
        "configured project directory. Inspect files before editing. Prefer "
        "small, focused changes. Use replace_in_file for exact, focused text "
        "replacements, apply_patch for multi-line targeted edits, and "
        "write_file for new or full-file replacements. apply_patch accepts "
        "only this Codex-style envelope: *** Begin Patch, then operations "
        "like *** Update File: path, @@, context lines prefixed with a space, "
        "removed lines prefixed with -, added lines prefixed with +, and "
        "*** End Patch. Example: *** Begin Patch\\n*** Update File: app.py\\n"
        "@@\\n-print('old')\\n+print('new')\\n*** End Patch. "
        "apply_unified_diff is different: it accepts standard unified diffs "
        "with --- a/path, +++ b/path, and numbered @@ -x,y +x,y @@ hunks; do "
        "not send apply_patch envelopes to apply_unified_diff. Use shell "
        "commands to create projects, run builds, and verify behavior when "
        "useful. Do not use shell heredocs or redirection for file edits. "
        "Explain command results and summarize changed files. Do not attempt "
        "destructive actions unless the user explicitly asks and the approval prompt "
        f"allows it.{reviewer_instruction}"
    )
