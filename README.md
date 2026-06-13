> ![NEW](https://img.shields.io/badge/%E2%9C%A8_NEW-8250df?style=for-the-badge)
> ## OpenCoworker
> **An AI agent that lives on your desktop, built on aisuite.**
>
> OpenCoworker is a desktop AI agent that can not only chat, but also do deep research and carry out tasks for
> you on your computer. It can read files (with permission) to gain context, read/send messages (slack, email, etc.),
> and create real deliverables like PDF reports, documents, spreadsheets. It also supports scheduled automations,
> such as providing you a daily news summary. 
>
> Requires bringing your own API key (OpenAI, Anthropic, Google) or run fully local with Ollama. Your data stays on your machine.
>
> [**⬇ Download for macOS**](https://github.com/andrewyng/aisuite/releases/latest/download/OpenCoworker-macos-arm64.dmg)
> <sub>&nbsp;&nbsp;macOS 13+ (Apple Silicon)</sub> &nbsp;&nbsp;
> 
> [**⬇ Download for Windows**](https://github.com/andrewyng/aisuite/releases/latest/download/OpenCoworker-windows-setup.exe)
> <sub>&nbsp;&nbsp;Windows 10/11 (x64) &nbsp;·&nbsp; </sub>
>
> [**Quickstart:**](docs/opencoworker-quickstart.md) — install, connect a model, first tasks, automations.
> 
> Its source lives in this repository under `platform/` — a working reference for building your own agent harness on aisuite.

<br>

#  aisuite

[![PyPI](https://img.shields.io/pypi/v/aisuite)](https://pypi.org/project/aisuite/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`aisuite` is a lightweight Python library for building with LLMs, in two layers: a unified **Chat Completions API** across providers, and an **Agents API** with tools and toolkits on top. This repo is also home to **OpenCoworker**, a desktop AI coworker built using aisuite:

```text
┌───────────────────────────────────────────────┐
│                 OpenCoworker                  │   agent harness for doing everyday tasks
├───────────────────────────────────────────────┤
│        Agents API  ·  Toolkits  ·  MCP        │   build agents across multiple LLMs
├───────────────────────────────────────────────┤
│             Chat Completions API              │   one API across multiple LLM providers
├────────┬───────────┬────────┬────────┬────────┤
│ OpenAI │ Anthropic │ Google │ Ollama │ Others │
└────────┴───────────┴────────┴────────┴────────┘
```

* **[Chat Completions API](#chat-completions)** — a unified, OpenAI-style interface for *OpenAI, Anthropic, Google, Mistral, Hugging Face, AWS, Cohere, Ollama, OpenRouter*, and more. Swap providers by changing one string.
* **[Agents API · Toolkits · MCP](#agents)** — give models real Python functions as tools, run multi-turn loops, attach ready-made toolkits (files, git, shell) or any MCP server, and govern it all with tool policies.
* **[OpenCoworker](docs/opencoworker-quickstart.md)** — a desktop AI coworker built using aisuite, shipped as an app for everyday tasks.

---

## Installation

### The aisuite library (Python)

Install the base package, or include the SDKs of the providers you plan to use:

```shell
pip install aisuite               # base package, no provider SDKs
pip install 'aisuite[anthropic]'  # with a specific provider's SDK
pip install 'aisuite[all]'        # with all provider SDKs
```

You'll also need API keys for the providers you call — the [Chat Completions quickstart](docs/chat-completions-quickstart.md) covers key setup and your first calls.

### The OpenCoworker app (desktop)

Download the installer and bring your own API key (or run local models with Ollama):

[**⬇ macOS (Apple Silicon)**](https://github.com/andrewyng/aisuite/releases/latest/download/OpenCoworker-macos-arm64.dmg) &nbsp;·&nbsp; [**⬇ Windows 10/11 (x64)**](https://github.com/andrewyng/aisuite/releases/latest/download/OpenCoworker-windows-setup.exe) &nbsp;·&nbsp; [OpenCoworker quickstart](docs/opencoworker-quickstart.md)

---

<a id="chat-completions"></a>
## Chat Completions — one API across providers

The chat API provides a high-level abstraction for model interactions. It supports all core parameters (`temperature`, `max_tokens`, `tools`, etc.) in a provider-agnostic way, and standardizes request and response structures so you can focus on logic rather than SDK differences.

Model names use the format `<provider>:<model-name>`; aisuite routes the call to the right provider with the right parameters:

```python
import aisuite as ai
client = ai.Client()

models = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.75
    )
    print(response.choices[0].message.content)
```

**→ Quickstart:** [docs/chat-completions-quickstart.md](docs/chat-completions-quickstart.md) — install, key setup, local models, and more examples.

---

<a id="agents"></a>
## Agents — give models real tools

aisuite turns tool calling into a one-liner: pass plain Python functions and it generates the schemas, executes the calls, and feeds results back to the model.

### Tool calling with `max_turns`

```python
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
    max_turns=2  # Maximum number of back-and-forth tool calls
)
print(response.choices[0].message.content)
```

With `max_turns` set, aisuite sends your message, executes any tool calls the model requests, returns the results to the model, and repeats until the conversation completes. `response.choices[0].intermediate_messages` carries the full tool interaction history if you want to continue the conversation.

Prefer full manual control? Omit `max_turns` and pass OpenAI-format JSON tool specs — aisuite returns the model's tool-call requests and you run the loop yourself. See `examples/tool_calling_abstraction.ipynb` for both styles.

### The Agents API

For longer-running, structured work there is a first-class Agents API: declare an agent once, run it with a `Runner`, and attach **toolkits** — prebuilt, sandboxed tool families for files, git, and shell:

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

The Agents API also gives you the pieces a production harness needs:

* **Tool policies** — `RequireApprovalPolicy`, allow/deny lists, or your own callable deciding which tool calls run.
* **State stores** — persist and resume runs (in-memory, file, or Postgres) and continue conversations across processes.
* **Artifacts & tracing** — capture what an agent produced and every step it took along the way.

### MCP tools

aisuite natively supports the [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro), so any MCP server's tools can be handed to a model without boilerplate (`pip install 'aisuite[mcp]'`):

```python
client = ai.Client()
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
print(response.choices[0].message.content)
```

For reusable connections, security filters, and tool prefixing, use the explicit `MCPClient`.

**→ Quickstart:** [docs/agents-quickstart.md](docs/agents-quickstart.md) — manual tool handling, the full Agents API, policies, state stores, and MCP in depth.

---

## Extending aisuite: Adding a Provider

New providers can be added by implementing a lightweight adapter. The system uses a naming convention for discovery:

| Element         | Convention                         |
| --------------- | ---------------------------------- |
| **Module file** | `<provider>_provider.py`           |
| **Class name**  | `<Provider>Provider` (capitalized) |

Example:

```python
# providers/openai_provider.py
class OpenaiProvider(BaseProvider):
    ...
```

This convention ensures consistency and enables automatic loading of new integrations.

---

## Contributing

Contributions are welcome. Please review the [Contributing Guide](https://github.com/andrewyng/aisuite/blob/main/CONTRIBUTING.md) and join our [Discord](https://discord.gg/T6Nvn8ExSb) for discussions.

---

## License

Released under the **MIT License** — free for commercial and non-commercial use.

---
