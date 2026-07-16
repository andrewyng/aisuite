# Crusoe

To use Crusoe Managed Inference with `aisuite`, you need a [Crusoe Cloud](https://crusoecloud.com/) account. Log in to the [Crusoe Cloud Console](https://console.crusoecloud.com/) and generate an API key. Once you have a key, add it to your environment as follows:

```shell
export CRUSOE_API_KEY="your-crusoe-api-key"
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

provider = "crusoe"
model_id = "meta-llama/Llama-3.3-70B-Instruct"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many times has Jurgen Klopp won the Champions League?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

See the [Crusoe Managed Inference documentation](https://docs.crusoecloud.com/managed-inference/overview) for the full list of available models.

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
