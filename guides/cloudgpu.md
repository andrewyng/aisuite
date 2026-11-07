# CloudGPU

To use CloudGPU with `aisuite`, you need a [CloudGPU](https://cloudgpu.app) account and an API key. CloudGPU serves DeepSeek, Kimi, GLM, MiniMax, Qwen and more through an OpenAI-compatible API at `https://cloudgpu.app/v1`. Once you have a key, add it to your environment as follows:

```shell
export CLOUDGPU_API_KEY="your-cloudgpu-api-key"
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

provider = "cloudgpu"
model_id = "deepseek-v4-flash"

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

Other chat models include `kimi-k2.6`, `glm-5.3`, `minimax-m3` and `qwen3-coder-480b`. The full list is at `https://cloudgpu.app/v1/models`.

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
