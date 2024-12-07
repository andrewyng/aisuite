# aisuite

[![PyPI](https://img.shields.io/pypi/v/aisuite)](https://pypi.org/project/aisuite/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple, unified interface to multiple Generative AI providers.

`aisuite` makes it easy for developers to use multiple LLM through a standardized interface. Using an interface similar to OpenAI's, `aisuite` makes it easy to interact with the most popular LLMs and compare the results. It is a thin wrapper around python client libraries, and allows creators to seamlessly swap out and test responses from different LLM providers without changing their code. Today, the library is primarily focussed on chat completions. We will expand it cover more use cases in near future.

Currently supported providers are -
OpenAI, Anthropic, Azure, Google, AWS, Groq, Mistral, HuggingFace Ollama and Sambanova.
To maximize stability, `aisuite` uses either the HTTP endpoint or the SDK for making calls to the provider.

## Installation

You can install just the base `aisuite` package, or install a provider's package along with `aisuite`.

This installs just the base package without installing any provider's SDK.

```shell
pip install aisuite
```

This installs aisuite along with anthropic's library.
```shell
pip install 'aisuite[anthropic]'
```

This installs all the provider-specific libraries
```shell
pip install 'aisuite[all]'
```

## Set up

To get started, you will need API Keys for the providers you intend to use. You'll need to
install the provider-specific library either separately or when installing aisuite.

The API Keys can be set as environment variables, or can be passed as config to the aisuite Client constructor.
You can use tools like [`python-dotenv`](https://pypi.org/project/python-dotenv/) or [`direnv`](https://direnv.net/) to set the environment variables manually. Please take a look at the `examples` folder to see usage.

Here is a short example of using `aisuite` to generate chat completion responses from gpt-4o and claude-3-5-sonnet.

Set the API keys.
```shell
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Use the python client.
```python
import aisuite as ai
client = ai.Client()

models = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]

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
Note that the model name in the create() call uses the format - `<provider>:<model-name>`.
`aisuite` will call the appropriate provider with the right parameters based on the provider value.
For a list of provider values, you can look at the directory - `aisuite/providers/`. The list of supported providers are of the format - `<provider>_provider.py` in that directory. We welcome  providers adding support to this library by adding an implementation file in this directory. Please see section below for how to contribute.

For more examples, check out the `examples` directory where you will find several notebooks that you can run to experiment with the interface.

## License

aisuite is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Contributing

If you would like to contribute, please read our [Contributing Guide](https://github.com/andrewyng/aisuite/blob/main/CONTRIBUTING.md) and join our [Discord](https://discord.gg/T6Nvn8ExSb) server!

## Adding support for a provider
We have made easy for a provider or volunteer to add support for a new platform.

### Naming Convention for Provider Modules

We follow a convention-based approach for loading providers, which relies on strict naming conventions for both the module name and the class name. The format is based on the model identifier in the form `provider:model`.

- The provider's module file must be named in the format `<provider>_provider.py`.
- The class inside this module must follow the format: the provider name with the first letter capitalized, followed by the suffix `Provider`.

#### Examples:

- **Hugging Face**:
  The provider class should be defined as:
  ```python
  class HuggingfaceProvider(BaseProvider)
  ```
  in providers/huggingface_provider.py.
  
- **OpenAI**:
  The provider class should be defined as:
  ```python
  class OpenaiProvider(BaseProvider)
  ```
  in providers/openai_provider.py

This convention simplifies the addition of new providers and ensures consistency across provider implementations.
