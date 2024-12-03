# CentML Platform

To use CentML Platform with the `aisuite` library, you'll need a [CentML Platform](https://app.centml.com) account. After logging in, go to the [User credentials](https://app.centml.com/user/credentials) section in your account settings and generate a new API key. Once you have your key, add it to your environment as follows:

```shell
export CENTML_API_KEY="your-centml-api-key"
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "centml"
model_id = "meta-llama/Llama-3.1-405B-Instruct-FP8"

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
