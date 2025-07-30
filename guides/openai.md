# OpenAI

To use OpenAI with `aisuite`, you’ll need an [OpenAI account](https://platform.openai.com/). After logging in, go to the [API Keys](https://platform.openai.com/account/api-keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export OPENAI_API_KEY="your-openai-api-key"
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

provider = "openai"
model_id = "gpt-4-turbo"

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

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
