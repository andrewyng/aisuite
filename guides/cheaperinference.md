# Cheaper Inference

To use [Cheaper Inference](https://cheaperinference.com) with `aisuite`, you'll need a [Cheaper Inference account](https://cheaperinference.com/signup). After signing up, generate an API key. Once you have your key, add it to your environment as follows:

```shell
export CHEAPER_INFERENCE_API_KEY="your-cheaper-inference-api-key"
```

Cheaper Inference is an OpenAI-compatible gateway that gives access to models from many labs through a single API key. Each model costs 15–60% less than the list price of its lab. Model ids are bare (e.g. `gpt-5.4-mini`, `gpt-5.4`, `claude-sonnet-5`, `gemini-3.1-pro`). See the [model list](https://cheaperinference.com/#models) and the [docs](https://cheaperinference.com/docs) for details.

## Create a Chat Completion

(Note: Cheaper Inference uses an API format consistent with OpenAI, hence why we need to install `openai`.)

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

provider = "cheaperinference"
model_id = "gpt-5.4-mini"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you'd like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
