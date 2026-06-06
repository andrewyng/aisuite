from __future__ import annotations

import subprocess

import pytest

import aisuite as ai


def test_git_toolkit_status_and_diff_are_read_only(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "app.py").write_text("print('old')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    (tmp_path / "app.py").write_text("print('new')\n", encoding="utf-8")
    tools = {fn.__name__: fn for fn in ai.toolkits.git(root=tmp_path)}

    status = tools["git_status"]()
    diff = tools["git_diff"]("app.py")

    assert status["exit_code"] == 0
    assert "app.py" in status["stdout"]
    assert diff["exit_code"] == 0
    assert "-print('old')" in diff["stdout"]
    assert "+print('new')" in diff["stdout"]
    assert tools["git_status"].__aisuite_tool_metadata__.risk_level == "low"
    assert tools["git_diff"].__aisuite_tool_metadata__.capabilities == ["git_diff"]


def test_git_toolkit_blocks_path_traversal(tmp_path):
    tools = {fn.__name__: fn for fn in ai.toolkits.git(root=tmp_path)}

    with pytest.raises(PermissionError):
        tools["git_diff"]("../outside.py")
