"""P3 gate tests — persistent shell executor.

The executor drives the OS-native shell (bash on POSIX, PowerShell on Windows), so the
command strings here are parameterized per-OS. The behavior under test (cwd/env persistence,
exit codes, timeout-and-recover, truncation) is identical across both.
"""

from __future__ import annotations

import sys
import time

import pytest

from coworker.permissions import PermissionEngine
from coworker.tools import ToolRegistry
from coworker.tools.shell import LocalExecutor, shell_tools

_WIN = sys.platform == "win32"

# Per-OS command snippets exercising the same behavior in the native shell.
SET_ENV = "$env:GREETING='hello_world'" if _WIN else "export GREETING=hello_world"
ECHO_ENV = "echo $env:GREETING" if _WIN else "echo $GREETING"
EXIT_OK = "cmd /c exit 0" if _WIN else "true"
EXIT_FAIL = "cmd /c exit 1" if _WIN else "false"
SLEEP_5 = "Start-Sleep -Seconds 5" if _WIN else "sleep 5"
PRINT_1000 = (
    'foreach ($i in 1..1000) { "line$i" }'
    if _WIN
    else "for i in $(seq 1 1000); do echo line$i; done"
)


@pytest.fixture
def executor(tmp_path):
    ex = LocalExecutor(cwd=tmp_path, default_timeout=10)
    yield ex
    ex.close()


def test_cwd_persists_across_calls(executor, tmp_path):
    (tmp_path / "sub").mkdir()
    executor.run("cd sub")
    result = executor.run("pwd")
    assert result["exit_code"] == 0
    assert "sub" in result["output"]
    assert executor.cwd.endswith("sub")


def test_env_persists_across_calls(executor):
    executor.run(SET_ENV)
    result = executor.run(ECHO_ENV)
    assert "hello_world" in result["output"]


def test_exit_code_captured(executor):
    assert executor.run(EXIT_OK)["exit_code"] == 0
    assert executor.run(EXIT_FAIL)["exit_code"] == 1


def test_timeout_kills_command(executor):
    start = time.monotonic()
    result = executor.run(SLEEP_5, timeout=1)
    elapsed = time.monotonic() - start
    assert result["timed_out"] is True
    assert elapsed < 4.0  # did not block for the full sleep
    # session survives the timeout — still usable (POSIX keeps the shell; Windows respawns)
    assert executor.run("echo alive")["output"].strip().endswith("alive")


def test_large_output_truncated_keeps_tail(tmp_path):
    ex = LocalExecutor(cwd=tmp_path, max_output_chars=200, default_timeout=10)
    try:
        result = ex.run(PRINT_1000)
        assert result["truncated"] is True
        assert len(result["output"]) <= 200
        # the END survives (where test/build verdicts live), the head is dropped
        assert "line1000" in result["output"]
        assert "line1\n" not in result["output"]
    finally:
        ex.close()


# -- output decoding -------------------------------------------------------------

# Emit bytes that are not valid UTF-8 *and* not a valid sequence in the common
# Windows codepages, surrounded by plain ASCII. 0x81 followed by 0x20 is
# rejected by cp936; 0x81 is unmapped in cp1252. Real commands produce such
# bytes routinely: `cat` on a binary file, a compiler quoting a snippet in
# another encoding, curl echoing a response body.
_RAW_BYTES = b"ok-before\n\x81 raw\nok-after\n"
EMIT_RAW_BYTES = (
    "[Console]::OpenStandardOutput().Write("
    "[byte[]]@(%s), 0, %d)" % (",".join(str(b) for b in _RAW_BYTES), len(_RAW_BYTES))
    if _WIN
    else r"printf 'ok-before\n\201 raw\nok-after\n'"
)

# Valid UTF-8 that a legacy codepage would silently turn into mojibake rather
# than reject: cp936 decodes b"caf\xc3\xa9" to "cafÃ©" without raising, and
# cp1252 does the same. Written straight to the stdout handle so the bytes
# reach our pipe unmodified -- PowerShell's `Write-Output` would re-encode
# through its own output encoding first, which is a separate concern from the
# decoding this tests.
_UTF8_BYTES = "café\n".encode("utf-8")
EMIT_UTF8 = (
    "[Console]::OpenStandardOutput().Write("
    "[byte[]]@(%s), 0, %d)" % (",".join(str(b) for b in _UTF8_BYTES), len(_UTF8_BYTES))
    if _WIN
    else r"printf 'caf\303\251\n'"
)


def test_undecodable_output_does_not_kill_the_session(executor):
    """Output the shell's locale cannot decode must not take the reader down.

    The reader thread iterates over `proc.stdout`. If that raises, its `finally`
    pushes the EOF sentinel, `run()` reads that as "shell died" and returns with
    `exit_code=None` — and every surrounding line already buffered is lost with
    it. Decoding as UTF-8 with `errors="replace"` keeps the stream alive and
    degrades only the offending bytes.
    """
    result = executor.run(EMIT_RAW_BYTES)

    # The marker arrived, so the reader survived and the stream stayed in sync.
    assert result["exit_code"] == 0
    # The ASCII on both sides of the bad bytes is intact — nothing was dropped.
    assert "ok-before" in result["output"]
    assert "ok-after" in result["output"]
    # The undecodable byte became the replacement character, not an exception.
    assert "�" in result["output"]

    # And the session is still usable afterwards.
    assert "alive" in executor.run("echo alive")["output"]


def test_utf8_output_is_not_mojibake(executor):
    """UTF-8 command output must decode as UTF-8, not as the platform codepage.

    These bytes decode without raising under cp936/cp1252, so the failure is
    silent: the model is handed a corrupted string it cannot tell from the real
    one. Asserting on the exact text is the only way to catch it.
    """
    result = executor.run(EMIT_UTF8)

    assert result["exit_code"] == 0
    assert "café" in result["output"]
    # The codepage reading of those same bytes must not be what came through.
    assert "cafÃ©" not in result["output"]


def test_background_task_undecodable_output_survives(executor):
    """Same guarantee for background tasks, which use a separate Popen call."""
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    started = reg.execute(
        "run_shell", {"command": EMIT_RAW_BYTES, "run_in_background": True}
    )
    assert started["task_id"]

    acc, res = _poll_output(reg, started["task_id"], until_status="exited")
    assert res["exit_code"] == 0
    assert "ok-before" in acc
    assert "ok-after" in acc
    assert "�" in acc


def test_shell_tool_integration(executor, tmp_path):
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    assert {"run_shell", "shell_task_output", "shell_task_kill"} <= set(reg.names())

    spec = reg.get("run_shell")
    assert spec.metadata.requires_approval is True
    # polling/killing the agent's own background tasks doesn't need approval
    assert reg.get("shell_task_output").metadata.requires_approval is False
    assert reg.get("shell_task_kill").metadata.requires_approval is False

    eng = PermissionEngine(workspace_root=tmp_path)
    decision = eng.evaluate("run_shell", {"command": "echo hi"}, spec.metadata)
    assert not decision.allowed and decision.needs_user  # high-risk → asks

    out = reg.execute("run_shell", {"command": "echo hi"})
    assert "hi" in out["output"]


def test_run_shell_accepts_description_and_clamped_timeout(executor):
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    # `description` rides along for approval prompts/audit; it must not break execution.
    out = reg.execute(
        "run_shell",
        {"command": "echo ok", "description": "Say ok", "timeout_seconds": 99999},
    )
    assert out["exit_code"] == 0 and "ok" in out["output"]


# -- background tasks ------------------------------------------------------------

ECHO_THEN_SLEEP = (
    "Write-Output started; Start-Sleep -Seconds 30"
    if _WIN
    else "echo started; sleep 30"
)
QUICK_ECHO = "Write-Output quick_done" if _WIN else "echo quick_done"


def _poll_output(reg, task_id, *, until_status=None, deadline=10.0):
    """Poll shell_task_output, accumulating output until a status is reached."""
    acc = ""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        res = reg.execute("shell_task_output", {"task_id": task_id})
        acc += res["output"]
        if until_status is None or res["status"] == until_status:
            if until_status is None and not acc:
                time.sleep(0.1)
                continue
            return acc, res
        time.sleep(0.1)
    return acc, res


def test_background_task_runs_and_exits(executor):
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    started = reg.execute(
        "run_shell", {"command": QUICK_ECHO, "run_in_background": True}
    )
    assert started["status"] == "running" and started["task_id"]

    acc, res = _poll_output(reg, started["task_id"], until_status="exited")
    assert res["status"] == "exited"
    assert res["exit_code"] == 0
    assert "quick_done" in acc

    # output reads are incremental: a second read returns nothing new
    again = reg.execute("shell_task_output", {"task_id": started["task_id"]})
    assert again["output"] == ""


def test_background_task_kill(executor):
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    started = reg.execute(
        "run_shell", {"command": ECHO_THEN_SLEEP, "run_in_background": True}
    )
    acc, _ = _poll_output(reg, started["task_id"])
    assert "started" in acc  # it's alive and producing output

    killed = reg.execute("shell_task_kill", {"task_id": started["task_id"]})
    assert killed["status"] == "killed"

    res = reg.execute("shell_task_output", {"task_id": started["task_id"]})
    assert res["status"] == "exited"


def test_background_unknown_task_errors(executor):
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    assert (
        "unknown task"
        in reg.execute("shell_task_output", {"task_id": "bg-99"})["error"]
    )
    assert (
        "unknown task" in reg.execute("shell_task_kill", {"task_id": "bg-99"})["error"]
    )
