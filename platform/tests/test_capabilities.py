"""Per-model capability lookup (`capabilities_for`).

The table is a heuristic keyed on the provider prefix or the model name. These pin every
branch and guard the documented contract that *bare* names (no `provider:` prefix) resolve
the same as their provider-qualified form — previously bare `claude-*`/`gemini-*` fell
through to the conservative default and were misreported as text-only.
"""

from __future__ import annotations

from coworker.providers.capabilities import capabilities_for


def _caps(model):
    c = capabilities_for(model)
    return (c.tools, c.vision, c.parallel_tool_calls, c.streaming)


def test_ollama_is_conservative_but_tool_capable():
    assert _caps("ollama:qwen3-coder:30b") == (True, False, False, True)


def test_anthropic_and_gemini_prefixed_are_full():
    assert _caps("anthropic:claude-sonnet-4-6") == (True, True, True, True)
    assert _caps("gemini:gemini-2.5-flash") == (True, True, True, True)


def test_bare_claude_and_gemini_match_prefixed():
    # the documented "accepts bare names" contract — these used to be misreported
    assert _caps("claude-sonnet-4-6") == (True, True, True, True)
    assert _caps("gemini-2.5-flash") == (True, True, True, True)


def test_bare_and_prefixed_openai_resolve_identically():
    assert _caps("gpt-5.5") == _caps("openai:gpt-5.5") == (True, True, True, True)
    assert _caps("gpt-4o") == (True, True, True, True)


def test_openai_reasoning_models_constrained():
    assert _caps("o3-mini") == (True, False, False, True)
    assert _caps("openai:o1") == (True, False, False, True)


def test_deepseek_has_parallel_tools_no_vision():
    assert _caps("deepseek-chat") == (True, False, True, True)


def test_unknown_model_falls_back_to_conservative_default():
    assert _caps("some-unknown-model") == (True, False, False, True)
    assert _caps("mystery:whatever-v9") == (True, False, False, True)
