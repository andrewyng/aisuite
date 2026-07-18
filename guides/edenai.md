# Eden AI

To use [Eden AI](https://www.edenai.co) with `aisuite`, you'll need an [Eden AI account](https://app.edenai.run). After logging in, open your account settings and generate an API key. Once you have your key, add it to your environment as follows:

```shell
export EDENAI_API_KEY="your-edenai-api-key"
```

Eden AI is an EU-hosted, OpenAI-compatible gateway that exposes 100+ models from many providers through a single API key. Models use the `provider/model` naming scheme (e.g. `anthropic/claude-sonnet-4-5`, `mistral/codestral-latest`).

## Create a Chat Completion

(Note: Eden AI uses an API format consistent with OpenAI, hence why we need to install `openai`.)

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

provider = "edenai"
model_id = "anthropic/claude-sonnet-4-5"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
