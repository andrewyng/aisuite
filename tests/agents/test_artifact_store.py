import hashlib

import pytest

from unittest.mock import Mock

import aisuite as ai
from aisuite.agents.artifacts import dehydrate_messages, hydrate_messages
from tests.agents.helpers import chat_response


def assert_artifact_store_round_trip(store):
    ref = store.put(
        "hello world",
        media_type="text/plain",
        metadata={"kind": "stdout"},
    )

    assert ref.artifact_id
    assert ref.media_type == "text/plain"
    assert ref.size_bytes == len(b"hello world")
    assert ref.metadata["kind"] == "stdout"
    assert ref.metadata["sha256"] == hashlib.sha256(b"hello world").hexdigest()

    artifact = store.get(ref)
    assert artifact.ref.to_dict() == ref.to_dict()
    assert artifact.text() == "hello world"
    assert store.get(ref.artifact_id).data == b"hello world"
    assert ai.ArtifactRef.from_dict(ref.to_dict()).to_dict() == ref.to_dict()

    store.delete(ref)
    with pytest.raises(KeyError):
        store.get(ref)


def test_in_memory_artifact_store_round_trips_and_deletes():
    assert_artifact_store_round_trip(ai.InMemoryArtifactStore())


def test_file_artifact_store_round_trips_and_deletes(tmp_path):
    assert_artifact_store_round_trip(ai.FileArtifactStore(tmp_path / "artifacts"))


def test_file_artifact_store_accepts_artifact_uri(tmp_path):
    store = ai.FileArtifactStore(tmp_path / "artifacts")
    ref = store.put(b"\x00\x01", media_type="application/octet-stream")

    assert store.get(ref.uri).data == b"\x00\x01"


def test_artifact_metadata_must_be_json_serializable():
    store = ai.InMemoryArtifactStore()

    with pytest.raises(TypeError, match="JSON serializable"):
        store.put("bad", media_type="text/plain", metadata={"bad": object()})


def test_delete_missing_artifact_is_idempotent(tmp_path):
    ai.FileArtifactStore(tmp_path / "artifacts").delete("missing")
    ai.InMemoryArtifactStore().delete("missing")


def test_message_dehydration_and_hydration_preserves_model_context():
    store = ai.InMemoryArtifactStore()
    large_content = "x" * 25_000
    messages = [{"role": "tool", "content": large_content}]

    dehydrated = dehydrate_messages(messages, store, threshold_chars=100)

    assert dehydrated[0]["content"]["type"] == "artifact_ref"
    assert dehydrated[0]["content"]["preview"] == large_content[:4000]
    assert hydrate_messages(dehydrated, store) == messages


def test_runner_persists_artifactized_state_but_hydrates_before_model_call():
    client = ai.Client()
    client.chat.completions.create = Mock(
        side_effect=[chat_response("first"), chat_response("second")]
    )
    agent = ai.Agent(name="assistant", model="openai:gpt-4o")
    state_store = ai.InMemoryStateStore()
    artifact_store = ai.InMemoryArtifactStore()
    large_assistant_message = "large-output-" + ("x" * 25_000)

    first_state = ai.RunState(
        agent_name="assistant",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": large_assistant_message},
        ],
    )
    state_store.save_state(
        "thread_1",
        ai.RunState(
            agent_name="assistant",
            messages=dehydrate_messages(first_state.messages, artifact_store),
        ),
    )

    result = ai.Runner.continue_sync(
        agent,
        "Follow up",
        client=client,
        state_store=state_store,
        thread_id="thread_1",
        artifact_store=artifact_store,
    )

    sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent_messages[1]["content"] == large_assistant_message
    assert sent_messages[2] == {"role": "user", "content": "Follow up"}
    assert result.final_output == "first"

    stored = state_store.load_state("thread_1")
    assert stored.revision == 2
    assert stored.state.messages[1]["content"]["type"] == "artifact_ref"
    assert (
        hydrate_messages(stored.state.messages, artifact_store)[1]["content"]
        == large_assistant_message
    )


def test_structured_message_fields_can_be_artifactized_and_hydrated():
    store = ai.InMemoryArtifactStore()
    large_stdout = "line\n" * 5000
    messages = [
        {
            "role": "tool",
            "content": {
                "command": "python3 big.py",
                "stdout": large_stdout,
                "stderr": "",
            },
        }
    ]

    dehydrated = dehydrate_messages(messages, store, threshold_chars=100)

    stdout = dehydrated[0]["content"]["stdout"]
    assert stdout["type"] == "artifact_ref"
    assert stdout["preview"] == large_stdout[:4000]
    assert stdout["artifact_ref"]["metadata"]["field"] == "stdout"
    assert hydrate_messages(dehydrated, store) == messages
