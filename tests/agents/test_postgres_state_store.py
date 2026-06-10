import os

import pytest

import aisuite as ai
from aisuite.agents.postgres_state_store import (
    _message_artifact_refs,
    _replace_ordered_subsequence,
    _shared_message_prefix_length,
)


def make_state():
    return ai.RunState(
        agent_name="assistant",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ],
        run_name="chat",
        group_id="group_1",
        metadata={"request_id": "req_1"},
    )


def test_replace_ordered_subsequence_replaces_contiguous_source():
    assert _replace_ordered_subsequence(
        ["m1", "m2", "m3", "m4"],
        ["m2", "m3"],
        ["summary"],
    ) == ["m1", "summary", "m4"]


@pytest.mark.parametrize(
    "source",
    [[], ["m1", "m3"], ["missing"]],
)
def test_replace_ordered_subsequence_rejects_invalid_source(source):
    with pytest.raises(ValueError):
        _replace_ordered_subsequence(["m1", "m2", "m3"], source, ["summary"])


def test_shared_message_prefix_length_detects_reused_context_prefix():
    old_messages = [
        {"role": "system", "content": "summary"},
        {"role": "user", "content": "latest"},
    ]
    new_messages = [
        {"role": "system", "content": "summary"},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "answer"},
    ]

    assert _shared_message_prefix_length(old_messages, new_messages) == 2
    assert _shared_message_prefix_length(old_messages, new_messages[1:]) == 0


def test_message_artifact_refs_extracts_nested_artifacts_once():
    artifact_value = {
        "type": "artifact_ref",
        "preview": "abc",
        "artifact_ref": {
            "artifact_id": "artifact_1",
            "uri": "artifact://artifact_1",
            "media_type": "text/plain",
            "size_bytes": 3,
            "metadata": {"field": "stdout"},
        },
    }
    message = {
        "role": "tool",
        "content": {"stdout": artifact_value},
        "artifact_refs": [artifact_value],
    }

    refs = _message_artifact_refs(message)

    assert len(refs) == 1
    assert refs[0]["artifact_ref"]["artifact_id"] == "artifact_1"
    assert refs[0]["artifact_ref"]["metadata"]["field"] == "stdout"


def test_compaction_record_serializes():
    record = ai.CompactionRecord(
        compaction_id="cmp_1",
        thread_id="thread_1",
        source_message_ids=["m1", "m2"],
        summary_message_id="m_summary",
        summary_text="Earlier context summary",
        model="openai:gpt-4o-mini",
        input_token_count=100,
        output_token_count=20,
        metadata={"kind": "manual"},
    )

    assert record.to_dict()["source_message_ids"] == ["m1", "m2"]
    assert record.to_dict()["summary_text"] == "Earlier context summary"


@pytest.mark.integration
def test_postgres_state_store_round_trip_and_compaction():
    dsn = os.environ.get("AISUITE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set AISUITE_TEST_POSTGRES_DSN to run Postgres integration test")

    store = ai.PostgresStateStore.from_dsn(dsn, create_schema=True)
    thread_id = "test_thread_state_store"
    store.delete_state(thread_id)

    stored = store.save_state(
        thread_id,
        make_state(),
        metadata={"user_id": "user_1"},
    )
    loaded = store.load_state(thread_id)
    assert loaded.revision == 1
    assert loaded.metadata == {"user_id": "user_1"}
    assert [message["content"] for message in loaded.state.messages] == [
        "Hello",
        "Hi",
        "Continue",
    ]

    head = store.get_thread_head(thread_id)
    assert head["model_context_message_ids"] == head["full_history_message_ids"]
    assert len(head["model_context_message_ids"]) == 3

    compacted = store.compact_state(
        thread_id,
        head["model_context_message_ids"][:2],
        {
            "role": "system",
            "content": "Summary: user said hello and assistant replied.",
        },
        revision=stored.revision,
        reason="test compaction",
        model="test-model",
        input_token_count=25,
        output_token_count=9,
        metadata={"kind": "manual"},
    )
    assert compacted.revision == 2
    assert [message["content"] for message in compacted.state.messages] == [
        "Summary: user said hello and assistant replied.",
        "Continue",
    ]

    compacted_head = store.get_thread_head(thread_id)
    assert len(compacted_head["model_context_message_ids"]) == 2
    assert len(compacted_head["full_history_message_ids"]) == 3
    assert (
        compacted_head["compacted_from_message_ids"]
        == head["model_context_message_ids"][:2]
    )

    compactions = store.list_compactions(thread_id)
    assert len(compactions) == 1
    assert compactions[0].source_message_ids == head["model_context_message_ids"][:2]
    assert (
        compactions[0].summary_text == "Summary: user said hello and assistant replied."
    )

    continued_state = compacted.state
    continued_state.add_user_message("After compaction")
    saved_after_compaction = store.save_state(
        thread_id,
        continued_state,
        revision=compacted.revision,
    )
    final_head = store.get_thread_head(thread_id)
    assert saved_after_compaction.revision == 3
    assert len(final_head["model_context_message_ids"]) == 3
    assert len(final_head["full_history_message_ids"]) == 4

    with pytest.raises(ai.StateConflictError):
        store.save_state(thread_id, make_state(), revision=stored.revision)

    store.delete_state(thread_id)
    assert store.load_state(thread_id) is None
    assert store.get_thread_head(thread_id) is None
    assert store.list_compactions(thread_id) == []
