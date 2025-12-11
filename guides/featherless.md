# Featherless AI

To use Featherless with `aisuite`, you’ll need an [Featherless account](https://featherless.ai/). After logging in, go to the [API Keys](https://featherless.ai/account/api-keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export FEATHERLESS_API_KEY="your-featherless-api-key"
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

provider = "featherless"
model_id = "deepseek-ai/DeepSeek-R1"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the fastest way to get to the airport?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
