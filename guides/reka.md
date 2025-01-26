# Reka

To use Reka with `aisuite`, you’ll need an [Reka account](https://platform.reka.ai/). After logging in, go to the [API Keys](https://platform.reka.ai/apikeys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export REKA_API_KEY="your-reka-api-key"
```

## Create a Chat Completion

Install the `reka-api` Python client:

Example with pip:
```shell
pip install reka-api
```

Example with poetry:
```shell
poetry add reka-api
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "reka"
model_id = "reka-core"

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
