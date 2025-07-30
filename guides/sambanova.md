# Sambanova

To use Sambanova with `aisuite`, you’ll need a [Sambanova Cloud](https://cloud.sambanova.ai/) account. After logging in, go to the [API](https://cloud.sambanova.ai/apis) section and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export SAMBANOVA_API_KEY="your-sambanova-api-key"
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

provider = "sambanova"
model_id = "Meta-Llama-3.1-405B-Instruct"

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

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
