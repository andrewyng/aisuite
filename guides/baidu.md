# Baidu

To use Baidu's model, you must log in to the [Baidu AI Cloud Qianfan Large Model Service and Development Platform](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html) to obtain an Access Key and Secret Key. After obtaining these keys, configure them as environment variables using the following approach:

```shell
export QIANFAN_ACCESS_KEY="your-access-key"

export QIANFAN_SECRET_KEY="your-secret-key"
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
provider = "baidu"
model_id = "ERNIE-3.5-8K"

# Prepare messages
messages = [
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
