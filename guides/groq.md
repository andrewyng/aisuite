# Groq

To use Groq with `aisuite`, you’ll need a free [Groq account](https://console.groq.com/). After logging in, go to the [API Keys](https://console.groq.com/keys) section in your account settings and generate a new Groq API key. Once you have your key, add it to your environment as follows:

```shell
export GROQ_API_KEY="your-groq-api-key"
```

## Create a Python Chat Completion

1. First, install the `groq` Python client library:

```shell
pip install groq
```

2. Now you can simply create your first chat completion with the following example code or customize by swapoping out the `model_id` with any of the other available [models powered by Groq](https://console.groq.com/docs/models) and `messages` array with whatever you'd like:
```python
import aisuite as ai
client = ai.Client()

provider = "groq"
model_id = "llama-3.2-3b-preview"

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
