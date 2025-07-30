# Mistral

To use Mistral with `aisuite`, you’ll need a [Mistral account](https://console.mistral.ai/). 

After logging in, go to [Workspace billing](https://console.mistral.ai/billing) and choose a plan
- **Experiment** *(Free, 1 request per second); or*
- **Scale** *(Pay per use).*

Visit the [API Keys](https://console.mistral.ai/api-keys/) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export MISTRAL="your-mistralai-api-key"
```
## Create a Chat Completion

Install the `mistralai` Python client:

Example with pip:
```shell
pip install mistralai
```

Example with poetry:
```shell
poetry add mistralai
```

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "mistral"
model_id = "mistral-large-latest"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in Montréal?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
