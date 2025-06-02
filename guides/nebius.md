# Nebius AI Studio

To use Nebius AI Studio with `aisuite`, you need an AI Studio account. Go to [AI Studio](https://studio.nebius.ai/) and press "Log in to AI Studio" in the right top corner. After logging in, go to the [API Keys](https://studio.nebius.ai/settings/api-keys) section and generate a new key. Once you have a key, add it to your environment as follows:

```shell
export NEBIUS_API_KEY="your-nebius-api-key"
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

provider = "nebius"
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

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
