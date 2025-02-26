# DeepSeek

To use DeepSeek with `aisuite`, you’ll need an [DeepSeek account](https://platform.deepseek.com). After logging in, go to the [API Keys](https://platform.deepseek.com/api_keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

## Create a Chat Completion

(Note: The DeepSeek uses an API format consistent with OpenAI, hence why we need to install OpenAI, there is no DeepSeek Library at least not for now)

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

provider = "deepseek"
model_id = "deepseek-chat"

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
