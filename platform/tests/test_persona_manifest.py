"""Phase 1 gate — persona manifest parsing + validation."""

from __future__ import annotations

import pytest

from coworker.personas.manifest import ManifestError, parse_manifest

VALID = """---
id: demo
name: Demo Coworker
icon: demo
tagline: A demo
family: knowledge
workspace: deliverable
tools: [files, search, shell, todo]
messaging: true
connectors: true
recommended_models: [anthropic:claude-opus-4-8]
default_permission_mode: interactive
---
You are a demo coworker. Do helpful things.
"""


def test_parse_valid():
    m = parse_manifest(VALID)
    assert m.id == "demo" and m.name == "Demo Coworker"
    assert m.tools == ["files", "search", "shell", "todo"]
    assert m.family == "knowledge" and m.workspace == "deliverable"
    assert m.messaging is True and m.connectors is True
    assert m.recommended_models == ["anthropic:claude-opus-4-8"]
    assert m.needs_workspace is True
    assert m.system_prompt.startswith("You are a demo coworker")


def test_to_agent_carries_traits_and_tools(tmp_path):
    from coworker.agents.base import AgentContext
    from coworker.tools.todo import TodoList

    agent = parse_manifest(VALID).to_agent()
    assert agent.name == "demo" and agent.family == "knowledge"
    assert agent.messaging and agent.connectors
    ctx = AgentContext(workspace=tmp_path, executor=object(), todo=TodoList())
    names = {getattr(t, "__name__", "") for t in agent.build_tools(ctx)}
    assert {"read_file", "grep", "run_shell", "todo_write"} <= names


def test_list_field_accepts_comma_string():
    text = VALID.replace("tools: [files, search, shell, todo]", "tools: files, search")
    assert parse_manifest(text).tools == ["files", "search"]


def test_no_workspace_persona():
    text = """---
id: chatty
workspace: none
tools: []
---
Just chat.
"""
    m = parse_manifest(text)
    assert m.needs_workspace is False and m.tools == []


@pytest.mark.parametrize(
    "text,needle",
    [
        ("no frontmatter here", "frontmatter"),
        ("---\nid: x\ntools: [files]\n", "unterminated"),
        ("---\nname: x\n---\nbody", "id"),
        ("---\nid: x\ntools: [files]\n---\n", "no body"),
        ("---\nid: x\ntools: [nope]\n---\nbody", "unknown tool"),
        ("---\nid: x\nfamily: alien\ntools: []\n---\nbody", "family"),
        ("---\nid: x\nworkspace: cloud\ntools: []\n---\nbody", "workspace"),
        ("---\nid: x\ndefault_permission_mode: yolo\ntools: []\n---\nbody", "permission"),
    ],
)
def test_invalid_manifests_rejected(text, needle):
    with pytest.raises(ManifestError) as e:
        parse_manifest(text)
    assert needle in str(e.value).lower()


def test_fallback_id_from_filename():
    m = parse_manifest("---\nname: X\ntools: []\n---\nbody", fallback_id="ops")
    assert m.id == "ops"
