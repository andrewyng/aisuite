# aisuite

> **🆕 Introducing Open Coworker** — an open agent harness that runs tasks and
> automations on your machine. Desktop app, your own API keys, your files.
> **→ [Get started in 2 minutes](docs/coworker/quickstart.md)**

[![PyPI](https://img.shields.io/pypi/v/aisuite.svg)](https://pypi.org/project/aisuite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/andrewyng/aisuite/actions/workflows/run_pytest.yml/badge.svg)](https://github.com/andrewyng/aisuite/actions)

Simple, unified tools for building with LLMs — from a drop-in chat client to a
full agent harness. Three layers, each usable on its own.

---

## Chat Completions

A uniform client across 20+ providers using an OpenAI-compatible interface. Swap
models with a string — no SDK juggling, no rewrites.

```python
import aisuite as ai

client = ai.Client()
response = client.chat.completions.create(
    model="anthropic:claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.choices[0].message.content)
```

```bash
pip install aisuite          # add provider extras, e.g. aisuite[anthropic]
```

→ [examples](examples/chat-completion) · [docs](docs/chat)

---

## Agent API

A lightweight `Agent` + `Runner` built on the client — tools, continuation
state, and tracing, without adopting a heavy framework. The agent is the
reusable definition; the runner owns execution.

```python
import aisuite as ai

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."

agent = ai.Agent(
    name="assistant",
    model="openai:gpt-5.5",
    instructions="Answer briefly. Use tools when they help.",
    tools=[get_weather],
)

result = ai.Runner.run_sync(agent, "What's the weather in San Francisco?")
print(result.final_output)
```

```bash
pip install aisuite[agents]
```

→ [examples](examples/agents-api) · [docs](docs/agents)

---

## Open Coworker

An open agent harness for automating daily tasks and more — a desktop app where
an agent works in your folders, uses connectors and tools, produces artifacts,
and runs scheduled automations. Bring your own API keys; your files and keys stay
on your machine.

- **Multi-folder file access** with per-folder read-only / read-write grants
- **Connectors & tools** — browser automation, integrations, MCP servers
- **Artifacts** — Markdown, images, PDF, CSV, spreadsheets, and Office files
- **Automations** — scheduled runs that continue as conversations
- **Your models** — OpenAI, Anthropic, Gemini, and local models via Ollama

**[Download for macOS / Windows](https://github.com/andrewyng/aisuite/releases)**
· [quickstart](docs/coworker/quickstart.md) · [docs](docs/coworker)

---

## How they stack

Each layer builds on the one below it, and each is usable on its own:

```
Open Coworker        an agent harness for tasks & automations  (app)
      │  built on
Agent API            Agent + Runner: tools, state, tracing      (lib)
      │  built on
Chat Completions     one client, many providers                 (lib)
```

Start at whatever layer fits: call a model, wrap it in an agent, or run the whole
harness.

---

## Repository layout

```
libs/
  aisuite-py/     the `aisuite` package — Chat Completions + Agent API
  aisuite-js/     JavaScript/TypeScript port
apps/
  opencoworker/   the Open Coworker harness (desktop app + server)
  code-cli/       aisuite-code, the command-line coding agent
examples/         runnable examples per product
docs/             chat / agents / coworker
```

## Documentation

- **Chat Completions** — [docs/chat](docs/chat)
- **Agent API** — [docs/agents](docs/agents)
- **Open Coworker** — [docs/coworker](docs/coworker)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## License

[MIT](LICENSE)
