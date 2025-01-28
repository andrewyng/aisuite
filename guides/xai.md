# xAI

To use xAI with `aisuite`, you’ll need an [API key](https://console.x.ai/). Generate a new key and once you have your key, add it to your environment as follows:

```shell
export XAI_API_KEY="your-xai-api-key"
```

## Create a Chat Completion

Sample code:
```python
import aisuite as ai
client = ai.Client()

models = ["xai:grok-beta"]

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.75
    )
    print(response.choices[0].message.content)

```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
