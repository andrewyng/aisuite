# OpenRouter

To use OpenRouter with `aisuite`, you’ll need an [OpenRouter account](https://openrouter.ai). After logging in, go to the [API Keys](https://openrouter.ai/settings/keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

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

```py
import aisuite as ai

client = ai.Client()

provider = "openrouter"
model_id = "qwen/qwq-32b-preview"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```
