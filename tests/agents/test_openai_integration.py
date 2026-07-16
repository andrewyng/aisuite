import os

import pytest

import aisuite as ai

pytestmark = [pytest.mark.integration, pytest.mark.llm]


pytest.importorskip("openai")


def require_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")


def test_openai_runner_smoke_without_tools():
    require_openai_key()
    agent = ai.Agent(
        name="smoke_assistant",
        model="openai:gpt-4o-mini",
        instructions="Reply with exactly: aisuite-ok",
        model_settings={"temperature": 0},
    )

    result = ai.Runner.run_sync(agent, "Say the required phrase.")

    assert "aisuite-ok" in result.final_output.lower()
    assert result.trace_id.startswith("trace_")
    assert [step.type for step in result.steps] == ["agent", "model_response"]


def test_openai_runner_smoke_with_tool_and_policy():
    require_openai_key()
    calls = []

    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        calls.append(city)
        return f"The weather in {city} is sunny."

    def allow_policy(context: ai.ToolPolicyContext) -> bool:
        return context.tool_name == "get_weather"

    agent = ai.Agent(
        name="weather_assistant",
        model="openai:gpt-4o-mini",
        instructions=(
            "Use the get_weather tool to answer weather questions. "
            "Mention the city and condition from the tool result."
        ),
        tools=[get_weather],
        model_settings={"temperature": 0},
    )

    result = ai.Runner.run_sync(
        agent,
        "What is the weather in Paris?",
        max_turns=3,
        group_id="openai_integration_weather",
        metadata={"test": "openai_tool_smoke"},
        tool_policy=allow_policy,
    )

    step_types = [step.type for step in result.steps]
    assert calls
    assert "Paris" in result.final_output
    assert "sunny" in result.final_output.lower()
    assert "tool_call" in step_types
    assert "tool_result" in step_types


def test_openai_runner_smoke_with_denied_tool_policy():
    require_openai_key()
    calls = []

    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        calls.append(city)
        return f"The weather in {city} is sunny."

    def deny_policy(context: ai.ToolPolicyContext) -> ai.ToolPolicyDecision:
        return ai.ToolPolicyDecision(allowed=False, reason="test denial")

    agent = ai.Agent(
        name="weather_assistant",
        model="openai:gpt-4o-mini",
        instructions=(
            "Use the get_weather tool to answer weather questions. If the tool "
            "is unavailable or denied, say you cannot access weather data."
        ),
        tools=[get_weather],
        model_settings={"temperature": 0},
    )

    result = ai.Runner.run_sync(
        agent,
        "What is the weather in Paris?",
        max_turns=3,
        tool_policy=deny_policy,
    )

    denied_steps = [
        step
        for step in result.steps
        if step.type == "tool_call" and step.data.get("allowed") is False
    ]
    assert calls == []
    assert denied_steps
