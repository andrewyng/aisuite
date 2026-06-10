import json
from unittest.mock import Mock

import pytest

import aisuite as ai
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message
from tests.agents.helpers import chat_response


def tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_files_toolkit_lists_reads_and_searches_under_root(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.txt").write_text(
        "hello world\nsecond line",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text("hello markdown", encoding="utf-8")
    tools = {fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path)}

    assert tools["list_files"]() == ["b.md", "notes/a.txt"]
    assert tools["read_file"]("notes/a.txt") == "hello world\nsecond line"
    assert tools["read_file_lines"]("notes/a.txt", max_lines=1) == {
        "path": "notes/a.txt",
        "start_line": 1,
        "end_line": 1,
        "total_lines": 2,
        "content": "hello world",
    }
    assert tools["search_files"]("hello") == [
        {"path": "b.md", "line": 1, "text": "hello markdown"},
        {"path": "notes/a.txt", "line": 1, "text": "hello world"},
    ]


def test_files_toolkit_reads_line_ranges(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tools = {fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path)}

    assert tools["read_file_lines"]("sample.txt", start_line=2, max_lines=2) == {
        "path": "sample.txt",
        "start_line": 2,
        "end_line": 3,
        "total_lines": 4,
        "content": "two\nthree",
    }
    assert tools["read_file_lines"]("sample.txt", start_line=10, max_lines=2) == {
        "path": "sample.txt",
        "start_line": 10,
        "end_line": 9,
        "total_lines": 4,
        "content": "",
    }

    with pytest.raises(ValueError):
        tools["read_file_lines"]("sample.txt", start_line=0)
    with pytest.raises(ValueError):
        tools["read_file_lines"]("sample.txt", max_lines=0)


def test_files_toolkit_ignores_noisy_directories_by_default(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.pyc").write_text("binary", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("needle", encoding="utf-8")
    tools = {fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path)}

    assert tools["list_files"]() == ["src/kept.py"]
    assert tools["search_files"]("binary") == []


def test_files_toolkit_ignore_empty_list_disables_default_ignores(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "included.pyc").write_text("binary", encoding="utf-8")
    tools = {fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, ignore=[])}

    assert tools["list_files"]() == ["__pycache__/included.pyc"]
    assert tools["search_files"]("binary") == [
        {"path": "__pycache__/included.pyc", "line": 1, "text": "binary"}
    ]


def test_files_toolkit_blocks_path_traversal(tmp_path):
    tools = {fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path)}

    with pytest.raises(PermissionError):
        tools["read_file"]("../outside.txt")
    with pytest.raises(PermissionError):
        tools["read_file_lines"]("../outside.txt")


def test_files_toolkit_write_is_opt_in(tmp_path):
    read_only = {fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path)}
    writable = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    assert "write_file" not in read_only
    assert "apply_unified_diff" not in read_only
    assert "apply_patch" not in read_only
    assert writable["write_file"]("created.txt", "hello") == "created.txt"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "hello"


def test_files_toolkit_applies_unified_diff_to_existing_file(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    result = tools["apply_unified_diff"]("""--- a/sample.txt
+++ b/sample.txt
@@ -1,3 +1,3 @@
 one
-two
+TWO
 three
""")

    assert result == {
        "changed_files": ["sample.txt"],
        "added_files": [],
        "deleted_files": [],
        "file_count": 1,
        "hunk_count": 1,
    }
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == (
        "one\nTWO\nthree\n"
    )


def test_files_toolkit_applies_unified_diff_add_and_delete(tmp_path):
    (tmp_path / "remove.txt").write_text("bye\nnow\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    result = tools["apply_unified_diff"]("""--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+hello
+world
--- a/remove.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-bye
-now
""")

    assert result == {
        "changed_files": ["new.txt"],
        "added_files": ["new.txt"],
        "deleted_files": ["remove.txt"],
        "file_count": 2,
        "hunk_count": 2,
    }
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert not (tmp_path / "remove.txt").exists()


def test_files_toolkit_applies_codex_patch_to_existing_file(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    result = tools["apply_patch"]("""*** Begin Patch
*** Update File: sample.txt
@@
 one
-two
+TWO
 three
*** End Patch
""")

    assert result == {
        "changed_files": ["sample.txt"],
        "added_files": [],
        "deleted_files": [],
        "file_count": 1,
        "hunk_count": 1,
    }
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == (
        "one\nTWO\nthree\n"
    )


def test_files_toolkit_applies_codex_patch_add_delete_and_move(tmp_path):
    (tmp_path / "remove.txt").write_text("bye\n", encoding="utf-8")
    (tmp_path / "old.txt").write_text("old\nname\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    result = tools["apply_patch"]("""*** Begin Patch
*** Add File: new.txt
+hello
+world
*** Delete File: remove.txt
*** Update File: old.txt
*** Move to: renamed.txt
@@
-old
+new
 name
*** End Patch
""")

    assert result == {
        "changed_files": ["new.txt", "renamed.txt"],
        "added_files": ["new.txt"],
        "deleted_files": ["remove.txt", "old.txt"],
        "file_count": 3,
        "hunk_count": 3,
    }
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert (tmp_path / "renamed.txt").read_text(encoding="utf-8") == "new\nname\n"
    assert not (tmp_path / "remove.txt").exists()
    assert not (tmp_path / "old.txt").exists()


def test_files_toolkit_applies_codex_patch_multiple_hunks_and_insertions(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    result = tools["apply_patch"]("""*** Begin Patch
*** Update File: sample.txt
@@
-one
+ONE
 two
@@
 three
+inserted
 four
*** End Patch
""")

    assert result == {
        "changed_files": ["sample.txt"],
        "added_files": [],
        "deleted_files": [],
        "file_count": 1,
        "hunk_count": 2,
    }
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == (
        "ONE\ntwo\nthree\ninserted\nfour\n"
    )


def test_files_toolkit_rejects_bad_or_escaping_codex_patch(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    with pytest.raises(ValueError, match="context does not match"):
        tools["apply_patch"]("""*** Begin Patch
*** Update File: sample.txt
@@
 wrong
-two
+TWO
*** End Patch
""")

    with pytest.raises(PermissionError):
        tools["apply_patch"]("""*** Begin Patch
*** Add File: ../outside.txt
+outside
*** End Patch
""")


def test_files_toolkit_rejects_ambiguous_or_contextless_codex_patch(tmp_path):
    (tmp_path / "sample.txt").write_text("target\nx\ntarget\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    with pytest.raises(ValueError, match="ambiguous"):
        tools["apply_patch"]("""*** Begin Patch
*** Update File: sample.txt
@@
-target
+TARGET
*** End Patch
""")

    with pytest.raises(ValueError, match="context or removal"):
        tools["apply_patch"]("""*** Begin Patch
*** Update File: sample.txt
@@
+inserted
*** End Patch
""")


def test_files_toolkit_rejects_codex_patch_move_over_existing_file(tmp_path):
    (tmp_path / "old.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "existing.txt").write_text("existing\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    with pytest.raises(FileExistsError):
        tools["apply_patch"]("""*** Begin Patch
*** Update File: old.txt
*** Move to: existing.txt
@@
-old
+new
*** End Patch
""")
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "existing\n"
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "old\n"


def test_files_toolkit_rejects_bad_or_escaping_unified_diff(tmp_path):
    (tmp_path / "sample.txt").write_text("one\ntwo\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    with pytest.raises(ValueError, match="context does not match"):
        tools["apply_unified_diff"]("""--- a/sample.txt
+++ b/sample.txt
@@ -1,2 +1,2 @@
 wrong
-two
+TWO
""")

    with pytest.raises(PermissionError):
        tools["apply_unified_diff"]("""--- a/../outside.txt
+++ b/../outside.txt
@@ -0,0 +1,1 @@
+outside
""")


def test_files_toolkit_replaces_exact_text(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha beta beta\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    result = tools["replace_in_file"](
        "sample.txt",
        old="beta",
        new="gamma",
        expected_replacements=2,
    )

    assert result == {
        "path": "sample.txt",
        "replacements": 2,
        "chars_before": 16,
        "chars_after": 18,
    }
    assert (tmp_path / "sample.txt").read_text(
        encoding="utf-8"
    ) == "alpha gamma gamma\n"


def test_files_toolkit_replace_requires_expected_count(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha beta beta\n", encoding="utf-8")
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    with pytest.raises(ValueError, match="Expected 1 replacement"):
        tools["replace_in_file"]("sample.txt", old="beta", new="gamma")


def test_files_toolkit_attaches_tool_metadata(tmp_path):
    tools = {
        fn.__name__: fn for fn in ai.toolkits.files(root=tmp_path, allow_write=True)
    }

    read_metadata = tools["read_file"].__aisuite_tool_metadata__
    write_metadata = tools["write_file"].__aisuite_tool_metadata__

    assert read_metadata.category == "filesystem"
    assert read_metadata.risk_level == "low"
    assert read_metadata.capabilities == ["read_file"]
    assert tools["read_file_lines"].__aisuite_tool_metadata__.capabilities == [
        "read_file_lines"
    ]
    assert write_metadata.risk_level == "medium"
    assert write_metadata.requires_approval is True
    assert tools["apply_unified_diff"].__aisuite_tool_metadata__.capabilities == [
        "apply_patch"
    ]
    assert (
        tools["apply_unified_diff"].__aisuite_tool_metadata__.requires_approval is True
    )
    assert tools["apply_patch"].__aisuite_tool_metadata__.capabilities == [
        "apply_patch"
    ]
    assert tools["apply_patch"].__aisuite_tool_metadata__.requires_approval is True
    assert tools["replace_in_file"].__aisuite_tool_metadata__.capabilities == [
        "edit_file"
    ]
    assert tools["replace_in_file"].__aisuite_tool_metadata__.requires_approval is True


def test_files_toolkit_can_be_used_by_agent_tool_loop(tmp_path):
    (tmp_path / "answer.txt").write_text("agent file answer", encoding="utf-8")
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("read_file", '{"path": "answer.txt"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("The file says: agent file answer"),
    ]
    client.providers["openai"] = provider

    result = ai.Runner.run_sync(
        ai.Agent(
            name="reader",
            model="openai:gpt-4o",
            tools=ai.toolkits.files(root=tmp_path),
        ),
        "Read answer.txt",
        client=client,
    )

    assert result.final_output == "The file says: agent file answer"
    assert result.steps[-2].data["tool_metadata"]["category"] == "filesystem"
    tool_message = provider.chat_completions_create.call_args_list[1].args[1][-1]
    assert tool_message["content"] == '"agent file answer"'


def test_write_file_large_content_is_artifactized_in_trace_not_tool_execution(tmp_path):
    client = ai.Client()
    provider = Mock()
    large_content = "abc\n" * 6000
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[
            tool_call(
                "write_file",
                json.dumps({"path": "large.txt", "content": large_content}),
            )
        ],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("wrote file"),
    ]
    client.providers["openai"] = provider
    artifact_store = ai.InMemoryArtifactStore()

    result = ai.Runner.run_sync(
        ai.Agent(
            name="writer",
            model="openai:gpt-4o",
            tools=ai.toolkits.files(root=tmp_path, allow_write=True),
        ),
        "Write large file",
        client=client,
        artifact_store=artifact_store,
    )

    assert (tmp_path / "large.txt").read_text(encoding="utf-8") == large_content
    tool_call_steps = [step for step in result.steps if step.type == "tool_call"]
    argument_ref = tool_call_steps[0].data["argument_artifacts"][0]
    assert argument_ref["artifact_ref"]["metadata"]["field"] == "content"
    assert tool_call_steps[0].data["arguments"]["content"]["type"] == "artifact_ref"
    ref = ai.ArtifactRef.from_dict(argument_ref["artifact_ref"])
    assert artifact_store.get(ref).text() == large_content
    tool_message = provider.chat_completions_create.call_args_list[1].args[1][-1]
    assert tool_message["content"] == '"large.txt"'
