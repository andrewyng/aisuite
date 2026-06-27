"""Phase 2 gate — self-wake: timer + on-completion wake records and the tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from coworker.selfwake import WakeStore, selfwake_tools


def _now():
    return datetime.now(timezone.utc)


def test_timer_due_only_after_fire_time(tmp_path):
    store = WakeStore(tmp_path / "wakes.json")
    soon = store.add_timer("s1", _now() + timedelta(seconds=60))
    past = store.add_timer("s1", _now() - timedelta(seconds=1))
    due_ids = {w.id for w in store.due()}
    assert past.id in due_ids and soon.id not in due_ids


def test_completion_due_only_after_job_completes(tmp_path):
    store = WakeStore(tmp_path / "wakes.json")
    w = store.add_completion("s1", job_id="job-42")
    assert w.id not in {x.id for x in store.due()}  # not yet
    marked = store.complete_job("job-42")
    assert [x.id for x in marked] == [w.id]
    assert w.id in {x.id for x in store.due()}  # now due


def test_mark_fired_removes_from_due(tmp_path):
    store = WakeStore(tmp_path / "wakes.json")
    w = store.add_timer("s1", _now() - timedelta(seconds=1))
    store.mark_fired(w.id)
    assert w.id not in {x.id for x in store.due()}
    assert w.id not in {x.id for x in store.pending("s1")}


def test_persistence(tmp_path):
    store = WakeStore(tmp_path / "wakes.json")
    w = store.add_completion("s1", "job-1")
    reloaded = WakeStore(tmp_path / "wakes.json")
    assert any(x.id == w.id for x in reloaded.pending("s1"))


def test_selfwake_tools(tmp_path):
    store = WakeStore(tmp_path / "wakes.json")
    sleep_for, sleep_until, wake_on = selfwake_tools(store, "s1")

    r1 = sleep_for(30)
    assert r1["ok"] and r1["wake_id"]
    r2 = wake_on("job-9")
    assert r2["ok"] and r2["job_id"] == "job-9"
    r3 = sleep_until((_now() + timedelta(minutes=5)).isoformat())
    assert r3["ok"] and r3["fire_at"]

    # Two timers + one completion now pending for the session.
    pend = store.pending("s1")
    assert len(pend) == 3
    assert {w.kind for w in pend} == {"timer", "completion"}
