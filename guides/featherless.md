# Featherless.ai

To use Featherless with `aisuite`, you'll need an [featherless](https://featherless.ai/) account. After signing up, go to the [API Keys](https://featherless.ai/account/api-keys). Once you have your key, add it to your environment as follows:

```shell
export FEATHERLESS_API_KEY="your-featherless-api-key"
```

## Create a Chat Completion

Install the `openai` Python library:

Example with pip:
```shell
pip install openai
```

In your code:
```python
import aisuite as ai

client = ai.Client()

models = ["featherless:meta-llama/Meta-Llama-3.1-8B-Instruct", "featherless:meta-llama/Meta-Llama-3.1-8B-Instruct"]

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

Happy coding! If you'd like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).