# Minimax

To use Minimax with `aisuite`, you'll need a [Minimax account](https://platform.minimax.io/). After logging in, go to the [API Keys](https://platform.minimax.io/user-center/basic-information/interface-key) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export MINIMAX_API_KEY="your-minimax-api-key"
```

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

provider = "minimax"
model_id = "MiniMax-Text-01"

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

## Available Models

- `MiniMax-M2.1` - Multimodal model
- `MiniMax-M2.1-lightning` - Fast multimodal model
- `MiniMax-M2` - Older multimodal model

For the full list of available models, see the [Minimax API documentation](https://platform.minimax.io/docs/api-reference/api-overview).

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
