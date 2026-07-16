import json

import pytest

import aisuite as ai


def make_state(content="Hello"):
    return ai.RunState(
        agent_name="assistant",
        messages=[{"role": "user", "content": content}],
        run_name="chat",
        group_id="group_1",
        tags=["tag"],
        metadata={"request_id": "req_1"},
    )


def assert_stored_state_round_trip(store):
    stored = store.save_state(
        "thread/user:1",
        make_state(),
        metadata={"user_id": "user_1"},
    )

    loaded = store.load_state("thread/user:1")
    assert loaded.thread_id == "thread/user:1"
    assert loaded.revision == 1
    assert loaded.created_at
    assert loaded.updated_at
    assert loaded.metadata == {"user_id": "user_1"}
    assert loaded.state.agent_name == "assistant"
    assert loaded.state.messages == [{"role": "user", "content": "Hello"}]
    assert loaded.state.metadata == {"request_id": "req_1"}

    loaded.state.add_user_message("Follow up")
    updated = store.save_state(
        loaded.thread_id,
        loaded.state,
        revision=loaded.revision,
    )
    assert updated.revision == stored.revision + 1
    assert updated.created_at == stored.created_at
    assert updated.metadata == {"user_id": "user_1"}
    assert updated.state.messages[-1] == {"role": "user", "content": "Follow up"}


def test_in_memory_state_store_round_trips_and_updates_with_revision():
    assert_stored_state_round_trip(ai.InMemoryStateStore())


def test_file_state_store_round_trips_and_reloads(tmp_path):
    root = tmp_path / "state"
    assert_stored_state_round_trip(ai.FileStateStore(root))

    reloaded = ai.FileStateStore(root).load_state("thread/user:1")
    assert reloaded.revision == 2
    assert reloaded.state.messages[-1] == {"role": "user", "content": "Follow up"}

    files = list(root.iterdir())
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["thread_id"] == "thread/user:1"


def test_state_store_rejects_stale_revision(tmp_path):
    store = ai.FileStateStore(tmp_path / "state")
    stored = store.save_state("thread_1", make_state())
    store.save_state("thread_1", make_state("Second"), revision=stored.revision)

    with pytest.raises(ai.StateConflictError, match="State revision conflict"):
        store.save_state("thread_1", make_state("Stale"), revision=stored.revision)


def test_state_store_can_clear_metadata(tmp_path):
    store = ai.FileStateStore(tmp_path / "state")
    stored = store.save_state(
        "thread_1",
        make_state(),
        metadata={"user_id": "user_1"},
    )

    updated = store.save_state(
        "thread_1",
        make_state("Second"),
        revision=stored.revision,
        metadata={},
    )

    assert updated.metadata == {}


def test_state_store_delete_is_idempotent(tmp_path):
    store = ai.FileStateStore(tmp_path / "state")
    store.save_state("thread_1", make_state())

    store.delete_state("thread_1")
    store.delete_state("thread_1")

    assert store.load_state("thread_1") is None
