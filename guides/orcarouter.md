# OrcaRouter

To use [OrcaRouter](https://www.orcarouter.ai) with `aisuite`, you'll need an [OrcaRouter account](https://www.orcarouter.ai). After logging in, open your dashboard and create an API key (keys are prefixed with `sk-orca-`). Once you have your key, add it to your environment as follows:

```shell
export ORCAROUTER_API_KEY="your-orcarouter-api-key"
```

OrcaRouter is an OpenAI-compatible gateway that exposes models from many providers behind a single API key and endpoint. Models use the `provider/model` naming scheme (e.g. `openai/gpt-5.5`, `anthropic/claude-sonnet-4.6`, `google/gemini-3.6-flash`), and `orcarouter/auto` lets the gateway pick a model per request.

Optionally, identify your app to OrcaRouter so it shows up in the gateway's attribution stats:

```shell
export ORCAROUTER_SITE_URL="https://your-app.example"   # sent as HTTP-Referer
export ORCAROUTER_APP_NAME="your-app"                   # sent as X-Title
```

## Create a Chat Completion

(Note: OrcaRouter uses an API format consistent with OpenAI, hence why we need to install `openai`.)

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

provider = "orcarouter"
model_id = "openai/gpt-5.5"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

## Streaming

The OrcaRouter provider also implements the streaming contract, so you can consume
OpenAI-shaped `chat.completion.chunk` objects as they arrive:

```python
for chunk in client.chat.completions.create(
    model="orcarouter:openai/gpt-5.5",
    messages=messages,
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

## Tool calling

Tools and the Agents API work through the gateway unchanged. One upstream caveat worth
knowing: OpenAI's newest reasoning models (e.g. `openai/gpt-5.6-sol`) reject function tools
on `/v1/chat/completions` unless reasoning is switched off, so pass `reasoning_effort="none"`
alongside `tools` for those. Other models — including `anthropic/*`, `google/*`, and earlier
`openai/gpt-5.x` — need no extra argument.

See the [OrcaRouter model catalog](https://www.orcarouter.ai/models) for the full list of available models.

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
