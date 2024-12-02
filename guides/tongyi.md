# Tongyi

To use the Alibaba Tongyi model, you need to log in to [Alibaba Cloud Bailian](https://bailian.console.aliyun.com) to obtain an API key. Once you have the API key, set it as an environment variable as follows:

```shell
export DASHSCOPE_API_KEY="your-api-key"
```

## Create a Chat Completion

### Example Using pip:

```shell
pip install aisuite[all]
```

### Example with poetry:

```shell
poetry add aisuite
```

### In Your Code:

```python
import aisuite as ai

# Initialize the client
client = ai.Client()

# Set provider and model ID
provider = "tongyi"
model_id = "qwen-plus"

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Make a chat completion request
response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

# Output the response
print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
