# Sarvam AI

To use Sarvam AI with `aisuite`, you'll need a [Sarvam Platform](https://dashboard.sarvam.ai) account. After signing up, generate an API key from the platform dashboard. Once you have your key, add it to your environment as follows:

```shell
export SARVAM_API_KEY="your-sarvam-api-key"
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

provider = "sarvam"
model_id = "sarvam-m"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about India."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you'd like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
