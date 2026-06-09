"""P3 gate tests — persistent shell executor."""

from __future__ import annotations

import time

import pytest

from coworker.permissions import PermissionEngine
from coworker.tools import ToolRegistry
from coworker.tools.shell import LocalExecutor, shell_tools


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
    executor.run("export GREETING=hello_world")
    result = executor.run("echo $GREETING")
    assert "hello_world" in result["output"]


def test_exit_code_captured(executor):
    assert executor.run("true")["exit_code"] == 0
    assert executor.run("false")["exit_code"] == 1


def test_timeout_kills_command(executor):
    start = time.monotonic()
    result = executor.run("sleep 5", timeout=1)
    elapsed = time.monotonic() - start
    assert result["timed_out"] is True
    assert elapsed < 4.0  # did not block for the full sleep
    # shell survives the timeout — session still usable
    assert executor.run("echo alive")["output"].strip().endswith("alive")


def test_large_output_truncated(tmp_path):
    ex = LocalExecutor(cwd=tmp_path, max_output_chars=200, default_timeout=10)
    try:
        result = ex.run("for i in $(seq 1 1000); do echo line$i; done")
        assert result["truncated"] is True
        assert len(result["output"]) <= 200
    finally:
        ex.close()


def test_shell_tool_integration(executor, tmp_path):
    reg = ToolRegistry()
    reg.register_all(shell_tools(executor))
    assert "run_shell" in reg.names()

    spec = reg.get("run_shell")
    assert spec.metadata.requires_approval is True

    eng = PermissionEngine(workspace_root=tmp_path)
    decision = eng.evaluate("run_shell", {"command": "echo hi"}, spec.metadata)
    assert not decision.allowed and decision.needs_user  # high-risk → asks

    out = reg.execute("run_shell", {"command": "echo hi"})
    assert "hi" in out["output"]
