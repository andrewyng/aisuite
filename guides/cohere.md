# Cohere

To use Cohere with `aisuite`, you’ll need an [Cohere account](https://cohere.com/). After logging in, go to the [API Keys](https://dashboard.cohere.com/api-keys) section in your account settings, agree to the terms of service, connect your card, and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export CO_API_KEY="your-cohere-api-key"
```

## Create a Chat Completion

Install the `cohere` Python client:

Example with pip:
```shell
pip install cohere
```

Example with poetry:
```shell
poetry add cohere
```

In your code:
```python
import aisuite as ai

client = ai.Client()

provider = "cohere"
model_id = "command-r-plus-08-2024"

messages = [
    {"role": "user", "content": "Hi, how are you?"}
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
