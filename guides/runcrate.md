# Runcrate

To use Runcrate with `aisuite`, you need a [Runcrate](https://www.runcrate.ai/?utm_source=aisuite&utm_medium=directory&utm_campaign=gpu_pricing) account. Sign in, open [API keys](https://www.runcrate.ai/dashboard/api-keys) and create a key. Open models are free with your GitHub account. Once you have a key, add it to your environment as follows:

```shell
export RUNCRATE_API_KEY="your-runcrate-api-key"
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

provider = "runcrate"
model_id = "deepseek-ai/DeepSeek-V4.1-Flash"

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

Other chat models include `zai-org/GLM-5.3`, `Qwen/Qwen3.8-Flash` and `openai/gpt-oss-120b`. The full list is at `https://api.runcrate.ai/v1/models`.

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
