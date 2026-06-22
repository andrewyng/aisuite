"""Session scratch-directory hygiene (issue #334).

Each Cowork conversation gets a per-session scratch dir under the scratch base. They
were created eagerly and never removed, so the base filled with empty UUID folders.
These cover: delete removes an *empty* scratch dir, never one holding agent output,
and merely inspecting roots no longer creates a dir as a side effect. No network,
no model calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coworker.server.manager import SessionManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    mgr = SessionManager(data_dir=tmp_path / "data")
    mgr.set_scratch_base(str(tmp_path / "scratch"))
    return mgr


def test_delete_session_removes_empty_scratch_dir(manager):
    sid = "sess-empty01"
    scratch = Path(manager._provision_scratch(sid))
    assert scratch.is_dir() and not any(scratch.iterdir())

    manager.delete_session(sid)

    assert not scratch.exists()


def test_delete_session_keeps_scratch_dir_with_files(manager):
    sid = "sess-haswork1"
    scratch = Path(manager._provision_scratch(sid))
    (scratch / "report.md").write_text("agent output")

    manager.delete_session(sid)

    assert scratch.is_dir()
    assert (scratch / "report.md").read_text() == "agent output"


def test_delete_session_without_scratch_dir_is_ok(manager):
    res = manager.delete_session("sess-never01")
    assert res["session_id"] == "sess-never01"


def test_delete_session_leaves_sibling_dirs_untouched(manager):
    other = manager.scratch_base() / "someone-elses-folder"
    other.mkdir(parents=True)

    manager.delete_session("sess-x")  # never provisioned

    assert other.is_dir()


def test_delete_session_rejects_path_traversal(manager):
    # a crafted id must never let cleanup escape the scratch base
    sentinel = manager.scratch_base().parent / "do-not-touch"
    sentinel.mkdir(parents=True, exist_ok=True)

    manager.delete_session("..")

    assert sentinel.is_dir()
    assert manager.scratch_base().resolve().is_dir()


def test_internal_session_is_not_deletable(manager):
    res = manager.delete_session("__superagent__")
    assert res["ok"] is False


def test_get_roots_does_not_create_scratch_dir(manager):
    sid = "sess-inspect1"

    roots = manager.get_roots(sid)

    primary = next(r for r in roots if r["primary"])
    assert primary["label"] == "scratch"
    assert primary["exists"] is False
    assert not (manager.scratch_base() / sid).exists()
