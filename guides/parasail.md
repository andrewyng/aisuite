# Parasail

Parasail exposes an OpenAI-compatible API through `aisuite`. Create an API key from the [Parasail API Keys page](https://www.saas.parasail.io/keys), copy it when it is displayed, and export it:

```shell
export PARASAIL_API_KEY="your-parasail-api-key"
```

Install `aisuite` with the Parasail provider dependency:

```shell
pip install 'aisuite[parasail]'
```

## Create a chat completion

```python
import aisuite as ai

client = ai.Client()
response = client.chat.completions.create(
    model="parasail:deepseek-ai/DeepSeek-V4-Flash-0731",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of New York?"},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

The text after `parasail:` is the Parasail model ID. Use Parasail's [`/models` endpoint](https://docs.parasail.io/parasail-docs/api-reference/models-endpoint) to list currently available models.

## Stream a chat completion

```python
stream = client.chat.completions.create(
    model="parasail:deepseek-ai/DeepSeek-V4-Flash-0731",
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

See the [Parasail Chat Completions reference](https://docs.parasail.io/parasail-docs/api-reference/chat-completions) for supported parameters, tool calling, and response details.
