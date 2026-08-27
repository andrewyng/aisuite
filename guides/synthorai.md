# Synthorai

To use [Synthorai](https://synthorai.io) with `aisuite`, you'll need a [Synthorai account](https://synthorai.io). After logging in, open the console and generate an API key. Once you have your key, add it to your environment as follows:

```shell
export SYNTHORAI_API_KEY="your-synthorai-api-key"
```

Synthorai is an OpenAI-compatible gateway that serves models from Anthropic, OpenAI, Google, DeepSeek, Qwen, Moonshot and Z.ai through a single API key. Unlike most gateways the model ids are bare rather than vendor-prefixed (e.g. `claude-opus-5`, `gpt-5.6-sol`, `deepseek-v4-pro`); the live list is at [https://synthorai.io/models/](https://synthorai.io/models/).

## Create a Chat Completion

(Note: Synthorai uses an API format consistent with OpenAI, hence why we need to install `openai`.)

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

provider = "synthorai"
model_id = "claude-opus-5"

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

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
