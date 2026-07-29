"""Tests for the Code agent's new tools: grep (ripgrep/python), git_log, web_fetch.

No network: web_fetch is exercised via its URL guard + the HTML→text helper. grep/git_log run
against temp dirs (git_log needs a real `git`, which the dev box has).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from coworker.tools.files import file_tools
from coworker.tools.git import git_tools
from coworker.tools.search import _parse_rg, _py_grep, search_tools
from coworker.web.fetch import _html_to_text, make_web_fetch_tool


# -- grep ----------------------------------------------------------------------
def _seed(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    return 42\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello world\nbye\n", encoding="utf-8")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "junk.py").write_text("hello from deps\n", encoding="utf-8")


def test_grep_finds_matches_and_respects_glob(tmp_path):
    _seed(tmp_path)
    grep = search_tools(str(tmp_path))[0]
    out = grep(pattern="hello")
    files = {m["file"] for m in out["matches"]}
    assert "a.py" in files and "b.txt" in files
    # node_modules is ignored by both engines
    assert not any("node_modules" in f for f in files)
    # glob filter restricts to python files
    only_py = grep(pattern="hello", glob="*.py")
    assert all(m["file"].endswith(".py") for m in only_py["matches"])
    assert only_py["matches"][0]["line"] == 1


def test_grep_rejects_path_escape(tmp_path):
    grep = search_tools(str(tmp_path))[0]
    assert "escapes" in grep(pattern="x", path="../..")["error"]


def test_py_grep_fallback_skips_ignored_dirs(tmp_path):
    _seed(tmp_path)
    res = _py_grep(tmp_path.resolve(), tmp_path.resolve(), "hello", None, 100)
    assert res["count"] == 2  # a.py + b.txt, NOT node_modules
    assert all("node_modules" not in m["file"] for m in res["matches"])


def test_parse_rg_keeps_colons_in_paths_and_text():
    """`grep` passes an absolute path to rg, so match lines start with "D:\\..." on
    Windows. Splitting on ':' would take the drive letter as the filename and the
    rest of the path as the line number. The NUL separator removes the ambiguity;
    colons inside the matched text must survive too.
    """
    root = Path("D:/ws")
    stdout = (
        "D:\\ws\\pkg\\a.py\x0012:    url = 'http://x/y'\nD:\\ws\\b.txt\x007:plain\n"
    )
    res = _parse_rg(stdout, root, 100)

    assert res["count"] == 2
    first = res["matches"][0]
    assert first["file"] == str(Path("pkg/a.py"))  # not "D"
    assert first["line"] == 12  # not 0 from a failed isdigit()
    assert first["text"] == "    url = 'http://x/y'"  # colons kept
    assert res["matches"][1]["line"] == 7


def test_parse_rg_ignores_malformed_lines():
    """rg also writes non-match lines (e.g. "path: No such file") to stdout."""
    res = _parse_rg(
        "some warning with no NUL\n\nD:\\ws\\a.py\x001:hit\n", Path("D:/ws"), 100
    )
    assert res["count"] == 1
    assert res["matches"][0]["text"] == "hit"


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_grep_survives_undecodable_bytes_in_matched_lines(tmp_path):
    """A single non-UTF-8 byte in any matched line must not lose the whole search.

    `subprocess.run(text=True)` decodes rg's output with the platform's preferred
    encoding and no error handler. One byte that codec rejects makes subprocess
    drop `stdout` to None, and `_parse_rg(None, ...)` then raises AttributeError
    outside the `try` — the tool errors out instead of returning the matches it
    already had. Bytes like this are ordinary in a real workspace: a latin-1
    source file, a fixture holding binary data, a minified asset.
    """
    (tmp_path / "clean.txt").write_bytes(b"needle in ascii\n")
    (tmp_path / "raw.txt").write_bytes(b"needle \xff\xfe undecodable\n")

    out = search_tools(str(tmp_path))[0](pattern="needle")

    assert "error" not in out
    files = {m["file"] for m in out["matches"]}
    assert "clean.txt" in files  # the good match is not collateral damage
    assert "raw.txt" in files  # and the offending line still comes through


# -- git_log -------------------------------------------------------------------
def test_git_log_lists_commits(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    run = lambda *a: subprocess.run(
        ["git", "-C", str(ws), *a], capture_output=True, check=True
    )
    run("init", "-q")
    run("config", "user.email", "t@t.io")
    run("config", "user.name", "T")
    (ws / "f.txt").write_text("1", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "first")
    (ws / "f.txt").write_text("2", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "second")

    git_log = git_tools(str(ws))[0]
    out = git_log(max_count=10)
    assert out["count"] == 2
    assert out["commits"][0]["subject"] == "second"  # newest first
    assert set(out["commits"][0]) == {"hash", "author", "date", "subject"}


def test_git_log_errors_outside_repo(tmp_path):
    git_log = git_tools(str(tmp_path))[0]
    assert "error" in git_log()


# -- read_file -----------------------------------------------------------------
def test_read_file_numbers_lines(tmp_path):
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    read_file = file_tools(str(tmp_path))[0]
    out = read_file(path="a.py")
    assert out["total_lines"] == 3
    assert out["start_line"] == 1 and out["end_line"] == 3
    assert out["content"].splitlines()[0].endswith("1\tone")
    assert "note" not in out  # nothing left to read


def test_read_file_windows_and_tells_how_to_continue(tmp_path):
    (tmp_path / "big.txt").write_text(
        "\n".join(f"line{i}" for i in range(1, 51)) + "\n", encoding="utf-8"
    )
    read_file = file_tools(str(tmp_path))[0]
    out = read_file(path="big.txt", start_line=5, max_lines=10)
    assert out["start_line"] == 5 and out["end_line"] == 14
    assert out["total_lines"] == 50
    assert "line5" in out["content"] and "line15" not in out["content"]
    assert "start_line=15" in out["note"]


def test_read_file_truncates_long_lines(tmp_path):
    (tmp_path / "wide.txt").write_text("x" * 2000 + "\n", encoding="utf-8")
    read_file = file_tools(str(tmp_path))[0]
    out = read_file(path="wide.txt")
    assert "(line truncated)" in out["content"]
    assert len(out["content"]) < 1000


def test_read_file_rejects_escape_and_non_files(tmp_path):
    read_file = file_tools(str(tmp_path))[0]
    assert "escapes" in read_file(path="../secret.txt")["error"]
    assert "not a file" in read_file(path="missing.txt")["error"]


# -- web_fetch -----------------------------------------------------------------
def test_web_fetch_rejects_non_http():
    web_fetch = make_web_fetch_tool()
    assert "http" in web_fetch("file:///etc/passwd")["error"]
    assert "http" in web_fetch("javascript:alert(1)")["error"]


def test_html_to_text_strips_scripts_and_tags():
    html = "<html><head><style>x{}</style></head><body><h1>Hi</h1><script>bad()</script><p>Body text</p></body></html>"
    text = _html_to_text(html)
    assert "Hi" in text and "Body text" in text
    assert "bad()" not in text and "x{}" not in text


# -- Code agent wiring ---------------------------------------------------------
def test_code_agent_has_grep_and_git_log_not_search_files(tmp_path):
    from coworker.agents.base import AgentContext
    from coworker.agents.code import code_agent

    ctx = AgentContext(workspace=tmp_path, executor=None, todo=None)
    names = {getattr(t, "__name__", "") for t in code_agent().build_tools(ctx)}
    assert "grep" in names and "git_log" in names
    assert "search_files" not in names  # replaced by grep
    assert "read_file_lines" not in names  # folded into our windowed read_file
    assert {"read_file", "write_file", "git_status", "git_diff"} <= names


def test_cowork_has_grep_not_search_files(tmp_path):
    from coworker.agents.base import AgentContext
    from coworker.agents.cowork import cowork_tool_factory

    names = {
        getattr(t, "__name__", "")
        for t in cowork_tool_factory(AgentContext(workspace=tmp_path))
    }
    assert "grep" in names and "search_files" not in names
    assert "git_log" not in names  # git history isn't useful for Cowork
