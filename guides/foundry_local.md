# Foundry Local

[Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) runs
open-source models directly on your device through Microsoft's on-device runtime.
It exposes an OpenAI-compatible API, so you can use the `aisuite` interface for
chat completions. No API key is needed for these locally hosted models.

There are two ways to use the `foundry_local` provider.

## Managed mode (recommended)

Install [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/get-started)
and the Foundry Local Python SDK. The SDK is included in the `foundry-local` extra
(`pip install 'aisuite[foundry-local]'`) on Python 3.11+. You can also install it
directly, picking the package that matches your hardware
(see the [SDK reference](https://learn.microsoft.com/azure/foundry-local/reference/reference-sdk-current?pivots=programming-language-python)):

```shell
pip install foundry-local-sdk
```

In managed mode, `aisuite` uses the SDK to start the local service on demand,
download and load the requested model, and resolve the model alias (e.g.
`phi-3.5-mini`) to the concrete model id served by the endpoint. Because Foundry
Local picks a dynamic port, the SDK is the easiest way to discover the endpoint.
Both the current `foundry-local-sdk` (imported as `foundry_local_sdk`) and the
legacy 0.x package (imported as `foundry_local`) are supported.

Sample code:
```python
import aisuite as ai

def main():
    client = ai.Client()
    messages = [
        {"role": "system", "content": "Be verbose"},
        {"role": "user", "content": "What is the golden ratio?"},
    ]

    # Use a Foundry Local model alias.
    foundry_phi = "foundry_local:phi-3.5-mini"

    response = client.chat.completions.create(
        model=foundry_phi,
        messages=messages,
        temperature=0.75,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

## Explicit endpoint mode

If you already have the Foundry Local service running, you can point `aisuite`
directly at its OpenAI-compatible endpoint via the `api_url`/`base_url` config key
or the `FOUNDRY_LOCAL_API_URL` environment variable. The SDK is not required in
this mode. You can use either a model alias (e.g. `phi-3.5-mini`) — `aisuite`
resolves it to the concrete id the endpoint serves by querying its `/v1/models`
list — or pass a concrete model id directly. Make sure the model is already
loaded on the running service (e.g. `foundry model run phi-3.5-mini`).

```python
import aisuite as ai

def main():
    client = ai.Client(
        provider_configs={
            "foundry_local": {
                "api_url": "http://localhost:5273",
                "timeout": 300,
            }
        }
    )
    messages = [
        {"role": "user", "content": "What is the golden ratio?"},
    ]

    response = client.chat.completions.create(
        # A friendly alias is resolved against the endpoint; a concrete id
        # (e.g. "Phi-3.5-mini-instruct-generic-cpu") also works.
        model="foundry_local:phi-3.5-mini",
        messages=messages,
        temperature=0.75,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

## Tool (function) calling

Foundry Local exposes an OpenAI-compatible API, so tool calls flow through
`aisuite` unchanged for models that support them. Passing `tools=` to
`client.chat.completions.create(...)` currently requires the `mcp` extra to be
installed, otherwise the call fails with `NameError: name 'is_mcp_config' is not
defined` (this affects every provider, not just Foundry Local):

```shell
pip install 'aisuite[mcp]'
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
