# Agents quickstart

Build agents across multiple LLMs: pass real Python functions as tools, run multi-turn loops, attach toolkits and MCP servers, and govern execution with policies.

Start with the [Chat Completions quickstart](chat-completions-quickstart.md) if you haven't installed aisuite and set up keys yet.

## Tool calling with `max_turns`

Pass plain Python functions — aisuite generates the schemas from signatures and docstrings, executes the calls, and feeds results back to the model until it finishes (or `max_turns` is reached):

```python
import aisuite as ai

def will_it_rain(location: str, time_of_day: str):
    """Check if it will rain in a location at a given time today.

    Args:
        location (str): Name of the city
        time_of_day (str): Time of the day in HH:MM format.
    """
    return "YES"

client = ai.Client()
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{
        "role": "user",
        "content": "I live in San Francisco. Can you check for weather "
                   "and plan an outdoor picnic for me at 2pm?"
    }],
    tools=[will_it_rain],
    max_turns=2
)
print(response.choices[0].message.content)
```

`response.choices[0].intermediate_messages` carries the full tool-interaction history — append it to your messages to continue the conversation.

## Manual tool handling

Omit `max_turns` for full control of the loop: pass OpenAI-format JSON tool specs, and aisuite returns the model's tool-call requests for you to execute, validate, or filter yourself.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "will_it_rain",
        "description": "Check if it will rain in a location at a given time today",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "Name of the city"},
                "time_of_day": {"type": "string", "description": "Time of the day in HH:MM format."}
            },
            "required": ["location", "time_of_day"]
        }
    }
}]

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=tools
)
```

This mode suits custom error handling, selective execution, or integrating an existing tool pipeline. Both styles are shown in [`examples/tool_calling_abstraction.ipynb`](../examples/tool_calling_abstraction.ipynb).

## The Agents API

For longer-running, structured work, declare an `Agent` once and run it with the `Runner`. **Toolkits** are prebuilt, sandboxed tool families — files, git, and shell — ready to attach:

```python
import aisuite as ai
from aisuite import Agent, Runner

agent = Agent(
    name="repo-helper",
    model="anthropic:claude-sonnet-4-6",
    instructions="You are a careful repo assistant. Use your tools to answer from the code.",
    tools=[*ai.toolkits.files(root="."), *ai.toolkits.git(root=".")],
)

result = Runner.run(agent, "What changed in the last commit? Summarize in 3 bullets.")
print(result.final_output)
```

Pieces a production harness needs:

- **Tool policies** — gate execution with `RequireApprovalPolicy`, `AllowToolsPolicy` / `DenyAllToolPolicy`, or any callable receiving a `ToolPolicyContext`.
- **State stores** — persist runs and resume them across processes: `InMemoryStateStore`, `FileStateStore`, or `PostgresStateStore` with `thread_id`.
- **Artifacts** — store what the agent produced (`FileArtifactStore`, `InMemoryArtifactStore`).
- **Tracing** — every `RunResult` carries its steps, raw responses, and a `trace_id`; plug in trace sinks for observability.

## MCP tools

Any [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro) server's tools can be handed to a model (`pip install 'aisuite[mcp]'`).

Inline config for simple cases:

```python
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "List the files in the current directory"}],
    tools=[{
        "type": "mcp",
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    }],
    max_turns=3
)
```

Or an explicit client you create once and reuse — with security filters and tool prefixing:

```python
from aisuite.mcp import MCPClient

mcp = MCPClient(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "List the files"}],
    tools=mcp.get_callable_tools(),
    max_turns=3
)
mcp.close()
```

See [`examples/mcp_tools_example.ipynb`](../examples/mcp_tools_example.ipynb) for detailed usage.

## Going further

- Want a ready-made desktop AI coworker instead of building your own? See the [OpenCoworker quickstart](opencoworker-quickstart.md).
- OpenCoworker's source under [`platform/`](../platform/) is a working reference for building a full agent harness with aisuite.
