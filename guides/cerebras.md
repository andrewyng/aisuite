# Cerebras AI Suite Provider Guide

## About Cerebras

At Cerebras, we've developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). The Cerebras CS-3 system, powered by the WSE-3, represents a new class of AI supercomputer that sets the standard for generative AI training and inference with unparalleled performance and scalability.

With Cerebras as your inference provider, you can:
- Achieve unprecedented speed for AI inference workloads
- Build commercially with high throughput
- Effortlessly scale your AI workloads with our seamless clustering technology

Our CS-3 systems can be quickly and easily clustered to create the largest AI supercomputers in the world, making it simple to place and run the largest models. Leading corporations, research institutions, and governments are already using Cerebras solutions to develop proprietary models and train popular open-source models.

Want to experience the power of Cerebras? Check out our [website](https://cerebras.net) for more resources and explore options for accessing our technology through the Cerebras Cloud or on-premise deployments!

> [!NOTE]  
> This SDK has a mechanism that sends a few requests to `/v1/tcp_warming` upon construction to reduce the TTFT. If this behaviour is not desired, set `warm_tcp_connection=False` in the constructor.
>
> If you are repeatedly reconstructing the SDK instance it will lead to poor performance. It is recommended that you construct the SDK once and reuse the instance if possible.

## Documentation

The REST API documentation can be found on [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai).


## Usage
Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:

```shell
export CEREBRAS_API_KEY="your-cerebras-api-key"
```

Use the python client.

```python
import aisuite as ai
client = ai.Client()

models = "cerebras:llama3.1-8b"

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.75
)
print(response.choices[0].message.content)

```

The full API of this library can be found at https://inference-docs.cerebras.ai/api-reference.

### Chat Completion
<!-- RUN TEST: ChatStandard -->
```python
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is fast inference important?",
        }
    ],
    model="llama3.1-8b",
)

print(chat_completion)
```

### Text Completion
<!-- RUN TEST: TextStandard -->
```python
completion = client.completions.create(
    prompt="It was a dark and stormy ",
    max_tokens=100,
    model="llama3.1-8b",
)

print(completion)
```

## Streaming responses

We provide support for streaming responses using Server Side Events (SSE).

Note that when streaming, `usage` and `time_info` will be information will only be included in the final chunk.

### Chat Completion
<!-- RUN TEST: ChatStreaming -->
```python
stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is fast inference important?",
        }
    ],
    model="llama3.1-8b",
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### Text Completion
<!-- RUN TEST: TextStreaming -->
```python
stream = client.completions.create(
    prompt="It was a dark and stormy ",
    max_tokens=100,
    model="llama3.1-8b",
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].text or "", end="")
```

### Retries

Certain errors are automatically retried 2 times by default, with a short exponential backoff.
Connection errors (for example, due to a network connectivity problem), 408 Request Timeout, 409 Conflict,
429 Rate Limit, and >=500 Internal errors are all retried by default.

You can use the `max_retries` option to configure or disable retry settings:

<!-- RUN TEST: Retries -->
```python
from cerebras.cloud.sdk import Cerebras

# Configure the default for all requests:
client = Cerebras(
    # default is 2
    max_retries=0,
)

# Or, configure per-request:
client.with_options(max_retries=5).chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is fast inference important?",
        }
    ],
    model="llama3.1-8b",
)
```

### Timeouts

By default requests time out after 1 minute. You can configure this with a `timeout` option,
which accepts a float or an [`httpx.Timeout`](https://www.python-httpx.org/advanced/#fine-tuning-the-configuration) object:

<!-- RUN TEST: Timeout -->
```python
from cerebras.cloud.sdk import Cerebras
import httpx

# Configure the default for all requests:
client = Cerebras(
    # 20 seconds (default is 1 minute)
    timeout=20.0,
)

# More granular control:
client = Cerebras(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)

# Override per-request:
client.with_options(timeout=5.0).chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Why is fast inference important?",
        }
    ],
    model="llama3.1-8b",
)
```

On timeout, an `APITimeoutError` is thrown.

Note that requests that time out are [retried twice by default](#retries).

## Advanced

### Logging

We use the standard library [`logging`](https://docs.python.org/3/library/logging.html) module.

You can enable logging by setting the environment variable `CEREBRAS_LOG` to `info`.

```shell
$ export CEREBRAS_LOG=info
```

Or to `debug` for more verbose logging.

#### Undocumented request params

If you want to explicitly send an extra param, you can do so with the `extra_query`, `extra_body`, and `extra_headers` request
options.

### Configuring the HTTP client

You can directly override the [httpx client](https://www.python-httpx.org/api/#client) to customize it for your use case, including:

- Support for [proxies](https://www.python-httpx.org/advanced/proxies/)
- Custom [transports](https://www.python-httpx.org/advanced/transports/)
- Additional [advanced](https://www.python-httpx.org/advanced/clients/) functionality

```python
import httpx
from cerebras.cloud.sdk import Cerebras, DefaultHttpxClient

client = Cerebras(
    # Or use the `CEREBRAS_BASE_URL` env var
    base_url="http://my.test.server.example.com:8083",
    http_client=DefaultHttpxClient(
        proxy="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

You can also customize the client on a per-request basis by using `with_options()`:

```python
client.with_options(http_client=DefaultHttpxClient(...))
```

## Requirements

Python 3.8 or higher.
