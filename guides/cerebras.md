# Cerebras

To use Cerebras with `aisuite`, you'll need a [Cerebras account](https://console.cerebras.net/). After logging in, navigate to your account settings to generate an API key. Once you have your key, add it to your environment as follows:

```shell
export CEREBRAS_API_KEY="your-cerebras-api-key"
```

## Create a Chat Completion

Install the `cerebras` Python client:

Example with pip:
```shell
pip install cerebras_cloud_sdk
```

Example with poetry:
```shell
poetry add cerebras_cloud_sdk
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "cerebras"
model_id = "llama3.1-8b"


messages = [
    {"content": "Why is fast inference important?"},
]

response = client.chat.completions.create(provider=provider, model_id=model_id, messages=messages)  

print(response.choices[0].message.content)

```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).