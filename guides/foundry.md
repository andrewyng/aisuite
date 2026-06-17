# Foundry Local

[Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/) runs
open-source models directly on your device through Microsoft's on-device runtime.
It exposes an OpenAI-compatible API, so you can use the `aisuite` interface for
chat completions. No API key is needed for these locally hosted models.

There are two ways to use the `foundry` provider.

## Managed mode (recommended)

Install [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/get-started)
and the Python SDK:

```shell
pip install foundry-local-sdk
```

In managed mode, `aisuite` uses the SDK's `FoundryLocalManager` to start the local
service on demand, download and load the requested model, and resolve the model
alias (e.g. `phi-3.5-mini`) to the concrete model id served by the endpoint.
Because Foundry Local picks a dynamic port, the manager is the easiest way to
discover the endpoint.

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
    foundry_phi = "foundry:phi-3.5-mini"

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
this mode, and the model string is passed through unchanged, so you must use the
concrete model id served by the endpoint.

```python
import aisuite as ai

def main():
    client = ai.Client(
        provider_configs={
            "foundry": {
                "api_url": "http://localhost:5273",
                "timeout": 300,
            }
        }
    )
    messages = [
        {"role": "user", "content": "What is the golden ratio?"},
    ]

    response = client.chat.completions.create(
        model="foundry:Phi-3.5-mini-instruct-generic-cpu",
        messages=messages,
        temperature=0.75,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
