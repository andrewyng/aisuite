# Synap

[Synap](https://synap.linkrra.com) is Linkrra's OpenAI-compatible inference pool, routing chat
completions across several open models (e.g. `gpt-oss-20b`, `qwen3-coder-30b`). To use Synap with
`aisuite`, create an account at [synap.linkrra.com](https://synap.linkrra.com), generate an API key,
and add it to your environment as follows:

```shell
export SYNAP_API_KEY="your-synap-api-key"
```

## Create a Chat Completion

(Note: Synap uses an API format consistent with OpenAI, hence why we need to install the `openai`
Python client; there is no separate Synap library.)

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

provider = "synap"
model_id = "gpt-oss-20b"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

The current model catalogue (with live pricing) is available at
`https://pool.linkrra.com/v1/models?format=openrouter`.

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
