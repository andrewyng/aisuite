# MLX-LM (local)

MLX-LM lets you run local models on Apple Silicon and exposes an OpenAI-compatible HTTP API via `mlx_lm.server`.

## Prerequisites

Install MLX-LM:

```bash
pip install mlx-lm
```

## Start the MLX-LM server

```bash
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

By default, the server listens on `http://localhost:8080`. You can change the host/port:

```bash
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --host 127.0.0.1 --port 8080
```

## Use with aisuite

```python
import aisuite as ai

client = ai.Client(
    provider_configs={
        "mlx": {
            "api_url": "http://localhost:8080",  # optional, this is the default
            "timeout": 30,                       # optional, this is the default
        }
    }
)

messages = [{"role": "user", "content": "Say this is a test!"}]

response = client.chat.completions.create(
    model="mlx:mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    messages=messages,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

Notes:
- `aisuite` requires model strings formatted as `provider:model`.
- The `model` value after the `:` is forwarded directly to the MLX-LM server as the `model` field. If the server was started with a fixed model, the value is effectively ignored by the server.
- If `MLX_API_URL` is set in your environment, the provider will use that as the base URL instead of the default.

## Environment variable

You can configure the server URL via environment variable instead of passing it in `provider_configs`:

```bash
export MLX_API_URL=http://localhost:8080
```
