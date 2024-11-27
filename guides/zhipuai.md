# ZhipuAI

To use ZhipuAI with `aisuite`, you'll need a [ZhipuAI account](https://open.zhipuai.cn/). After logging in, obtain your API key from your account settings. Once you have your key, add it to your environment as follows:

```shell
export ZHIPUAI_API_KEY="your-zhipuai-api-key"
```

## Create a Chat Completion

Install the `zhipuai` Python client:

Example with pip:
```shell
pip install zhipuai
```

Example with poetry:
```shell
poetry add zhipuai
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "zhipuai"
model_id = "glm-4"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in Beijing?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md). 