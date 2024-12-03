# OpenRouter

To use OpenRouter with `aisuite`, you'll need an [OpenRouter](https://openrouter.ai/) account. After signing up, go to the [API Keys](https://openrouter.ai/keys) section and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export OPENROUTER_API_KEY="your-openrouter-api-key"
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

provider = "openrouter"
model_id = "anthropic/claude-3-opus"  # Or any other model available on OpenRouter

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

Happy coding! If you'd like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).