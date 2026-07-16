"""
Minimal Agent API example.

Set the provider API key in your environment before running, for example:

    export OPENAI_API_KEY="..."
    python examples/agents/simple_agent.py
"""

import aisuite as ai


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."


def allow_safe_tools(context: ai.ToolPolicyContext) -> bool:
    """Allow only the tools this example expects."""
    return context.tool_name in {"get_weather"}


agent = ai.Agent(
    name="weather_assistant",
    model="openai:gpt-4o",
    instructions="Answer briefly. Use tools when they help.",
    tools=[get_weather],
    model_settings={"temperature": 0.2},
    tags=["example", "weather"],
    metadata={"app": "simple_agent_example"},
)

result = ai.Runner.run_sync(
    agent,
    "What is the weather in San Francisco?",
    max_turns=3,
    run_name="weather_lookup",
    group_id="example_conversation_1",
    metadata={"request_id": "example_request_1", "user_id": "example_user"},
    tool_policy=allow_safe_tools,
)

print(result.final_output)
result.print_trace()
result.write_trace_jsonl(".aisuite/runs.jsonl")

next_result = ai.Runner.continue_sync(
    result,
    "What about Oakland?",
    tool_policy=allow_safe_tools,
)

print(next_result.final_output)
next_result.print_trace()
next_result.write_trace_jsonl(".aisuite/runs.jsonl")
