# vLLM

To use [vLLM](https://docs.vllm.ai/en/latest/index.html) with `aisuite`, you’ll need to [start the OpenAI compatible vLLM server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-chat-completions-api-with-vllm).

Once your server has started, you can specify the API key and base URL:

```shell
export VLLM_API_KEY="your-vllm-api-key"
export VLLM_API_BASE_URL="your-vllm-base-url"
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

provider = "vllm"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

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
