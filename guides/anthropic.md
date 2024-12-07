# Anthropic

To use Anthropic with `aisuite` you will need to [create an account](https://console.anthropic.com/login). Once logged in, go to the [API Keys](https://console.anthropic.com/settings/keys)
and click the "Create Key" button and export that key into your environment.


```shell
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Create a Chat Completion

Install the `anthropic` python client

Example with pip
```shell
pip install anthropic
```

Example with poetry
```shell
poetry add anthropic
```

In your code:
```python
import aisuite as ai
client = ai.Client()


provider = "anthropic"
model_id = "claude-3-5-sonnet-20241022"

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
