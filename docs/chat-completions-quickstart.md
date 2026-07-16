# Chat Completions quickstart

One API across multiple LLM providers: write your code once, switch models by changing a string.

## 1. Install

```shell
pip install aisuite               # base package, no provider SDKs
pip install 'aisuite[anthropic]'  # with one provider's SDK
pip install 'aisuite[all]'        # with every provider SDK
```

## 2. Set your API keys

You need keys only for the providers you call. The [provider guides](../guides/README.md) walk through obtaining a key for each one.

Set them as environment variables (tools like [`python-dotenv`](https://pypi.org/project/python-dotenv/) or [`direnv`](https://direnv.net/) help manage them):

```shell
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Keys can also be passed programmatically to the `Client` constructor:

```python
client = ai.Client({"openai": {"api_key": "..."}})
```

## 3. Your first completions

Model names use the format `<provider>:<model-name>` — aisuite routes each call to the right provider and translates parameters and responses:

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

Core parameters (`temperature`, `max_tokens`, `tools`, …) work provider-agnostically; aisuite maps them to each SDK's conventions.

## Local models

Run fully local via [Ollama](https://ollama.com) — no API key required:

```python
response = client.chat.completions.create(
    model="ollama:llama3.3",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Going further

- The list of supported providers lives in [`aisuite/providers/`](../aisuite/providers/) (`<provider>_provider.py`).
- Runnable notebooks live in [`examples/`](../examples/).
- Ready to give the model tools? Continue with the [Agents quickstart](agents-quickstart.md).
