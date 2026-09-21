# The Grid

To use [The Grid](https://thegrid.ai) with `aisuite`, create an API key at [thegrid.ai](https://thegrid.ai) and add it to your environment:

```shell
export THEGRID_API_KEY="your-thegrid-api-key"
```

## What you are selecting

The Grid's model ids are **market instruments**, not fixed foundation models. You pick a quality tier and The Grid acquires qualifying inference on its market to serve the request:

| Instrument | Tier |
|---|---|
| `text-standard`, `code-standard`, `agent-standard` | everyday work |
| `text-prime`, `code-prime`, `agent-prime` | stronger reasoning |
| `text-max`, `code-max`, `agent-max` | frontier |

Lab-specific markets are also available when you want a particular family — `claude-opus-latest`, `gpt-sol-latest`, `gemini-pro-latest`, `kimi-latest`, `deepseek-pro-latest`, `glm-latest`, `minimax-latest`, `bytedance-pro-latest`.

Because an instrument pools several backing models, the `model` field of a response names the model that **actually served** the request rather than the instrument you asked for. Do not assume the two match.

Per-instrument context limits, capability flags and live prices are published at `GET https://api.thegrid.ai/v1/models`.

## Create a Chat Completion

Install the `openai` Python client:

Example with pip:
```shell
pip install openai
```

Example with poetry:
```shell
poetry add openai
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "thegrid"
model_id = "text-standard"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many times has Jurgen Klopp won the Champions League?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

## Tool calling

Every instrument supports tools and structured outputs:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="thegrid:agent-standard",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools,
)
```

## A note on `max_tokens`

Instruments reason before answering, and reasoning tokens count against `max_tokens`. A budget that is too small can be consumed entirely by reasoning, returning empty content with `finish_reason: "length"`. Leave headroom — a few hundred tokens is usually enough for short answers.

See the [The Grid documentation](https://thegrid.ai/docs) for the full list of instruments and their limits.

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
