# aisuite

[![PyPI](https://img.shields.io/pypi/v/aisuite)](https://pypi.org/project/aisuite/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple, unified interface to multiple Generative AI providers.

`aisuite` makes it easy for developers to use multiple LLM through a standardized interface. Using an interface similar to OpenAI's, `aisuite` makes it easy to interact with the most popular LLMs and compare the results. It is a thin wrapper around python client libraries, and allows creators to seamlessly swap out and test responses from different LLM providers without changing their code. Today, the library is primarily focussed on chat completions. We will expand it cover more use cases in near future.

Currently supported providers are -
OpenAI, Anthropic, Azure, Google, AWS, Groq, Mistral, HuggingFace Ollama, Sambanova and Watsonx.
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

## Async Support

AISuite now provides comprehensive async support for concurrent operations and streaming responses. The async implementation allows you to:
- Make concurrent requests to multiple models
- Stream responses asynchronously
- Efficiently manage resources with async context managers

### Basic Async Usage

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        response = await client.chat.completions.create(
            model="openai:gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What's the capital of France?"}
            ]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

### Concurrent Requests

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        messages = [{"role": "user", "content": "What's the capital of France?"}]
        models = ["openai:gpt-3.5-turbo", "openai:gpt-4"]
        
        tasks = [
            client.chat.completions.create(model=model, messages=messages)
            for model in models
        ]
        responses = await asyncio.gather(*tasks)
        
        for model, response in zip(models, responses):
            print(f"{model}: {response.choices[0].message.content}")

asyncio.run(main())
```

### Streaming Responses

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        async for chunk in client.chat.completions.create(
            model="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=True
        ):
            if chunk.choices[0].delta:
                print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
```

### Error Handling

The async client provides comprehensive error handling:

```python
from aisuite import AsyncClient
from aisuite.providers.async_openai_provider import OpenAIError

async def main():
    try:
        async with AsyncClient() as client:
            response = await client.chat.completions.create(
                model="openai:gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
            print(response.choices[0].message.content)
    except OpenAIError as e:
        print(f"API error occurred: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

asyncio.run(main())
```

### Configuration

The async client accepts the same configuration options as the synchronous client:

```python
import os
from aisuite import AsyncClient

client = AsyncClient({
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY")
    },
    # Add other provider configurations as needed
})
```

### Supported Features

The async implementation supports:
- All OpenAI chat completion models
- Streaming responses
- Function calling
- Tool calling
- Concurrent requests
- Automatic retries
- Resource cleanup
- Comprehensive error handling

### Requirements

To use the async features, ensure you have the following dependencies:
- Python 3.10+
- `httpx` >= 0.27.0
- `aiohttp` >= 3.9.0
- `openai` with async support

Add these to your requirements.txt or install them via pip:
```bash
pip install "httpx>=0.27.0" "aiohttp>=3.9.0" "openai>=1.0.0"
```

## Advanced Async Features

### Function Calling

The async client supports OpenAI's function calling feature:

```python
import asyncio
from aisuite import AsyncClient

async def get_weather(location: str, unit: str = "celsius") -> str:
    # Mock weather function
    return f"The weather in {location} is 22°{unit[0].upper()}"

async def main():
    async with AsyncClient() as client:
        functions = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]

        response = await client.chat.completions.create(
            model="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            functions=functions,
            function_call="auto"
        )

        if response.choices[0].message.function_call:
            func_call = response.choices[0].message.function_call
            # Call the function and get the result
            result = await get_weather(**eval(func_call.arguments))
            print(result)

asyncio.run(main())
```

### Handling Multiple Providers

The async client can handle multiple providers concurrently:

```python
import asyncio
import os
from aisuite import AsyncClient

async def main():
    async with AsyncClient({
        "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
        "anthropic": {"api_key": os.getenv("ANTHROPIC_API_KEY")}
    }) as client:
        messages = [{"role": "user", "content": "What's the capital of Japan?"}]
        
        # Create tasks for different providers
        tasks = [
            client.chat.completions.create(
                model="openai:gpt-4",
                messages=messages
            ),
            client.chat.completions.create(
                model="anthropic:claude-3-sonnet",
                messages=messages
            )
        ]
        
        # Run tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        for model, response in zip(["GPT-4", "Claude"], responses):
            if isinstance(response, Exception):
                print(f"{model} error: {str(response)}")
            else:
                print(f"{model}: {response.choices[0].message.content}")

asyncio.run(main())
```

### Streaming with Callbacks

You can use callbacks with streaming responses:

```python
import asyncio
from aisuite import AsyncClient

async def process_chunk(chunk):
    """Process each chunk of the stream."""
    if chunk.choices[0].delta:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)

async def main():
    async with AsyncClient() as client:
        async for chunk in client.chat.completions.create(
            model="openai:gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Write a short story"}],
            stream=True
        ):
            await process_chunk(chunk)

asyncio.run(main())
```

### Error Recovery and Retries

The async client includes built-in error recovery:

```python
import asyncio
from aisuite import AsyncClient
from aisuite.providers.async_openai_provider import OpenAIError

async def create_completion_with_retry(client, retries=3):
    """Create a completion with automatic retries."""
    for attempt in range(retries):
        try:
            return await client.chat.completions.create(
                model="openai:gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}]
            )
        except OpenAIError as e:
            if attempt == retries - 1:  # Last attempt
                raise  # Re-raise the last error
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

async def main():
    async with AsyncClient() as client:
        try:
            response = await create_completion_with_retry(client)
            print(response.choices[0].message.content)
        except OpenAIError as e:
            print(f"All retries failed: {str(e)}")

asyncio.run(main())
```

### Resource Management

The async client automatically manages resources:
- Closes connections when exiting the context manager
- Handles cleanup of streaming responses
- Manages concurrent connection limits
- Implements proper error handling and resource cleanup

### Performance Tips

1. Use `asyncio.gather()` for concurrent requests:
```python
responses = await asyncio.gather(*tasks, return_exceptions=True)
```

2. Implement timeouts for long-running operations:
```python
async def with_timeout(coro, timeout=30):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise OpenAIError("Request timed out")
```

3. Use streaming for long responses:
```python
async for chunk in client.chat.completions.create(..., stream=True):
    process_chunk(chunk)
```

4. Implement proper error handling and retries:
```python
try:
    async with AsyncClient() as client:
        # Your code here
except OpenAIError as e:
    handle_error(e)
```

### Best Practices

1. Always use async context managers:
```python
async with AsyncClient() as client:
    # Your code here
```

2. Handle errors appropriately:
```python
try:
    response = await client.chat.completions.create(...)
except OpenAIError as e:
    handle_error(e)
```

3. Clean up resources:
```python
finally:
    await client.close()
```

4. Use proper typing for better code quality:
```python
from typing import AsyncGenerator, List, Dict
async def get_stream() -> AsyncGenerator[str, None]:
    # Your code here
```

## Asynchronous Usage

AISuite now supports asynchronous operations for improved performance when working with multiple LLM providers concurrently. Here's how to use the async features:

### Basic Async Usage

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        response = await client.chat.completions.create(
            model="openai:gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

### Concurrent Requests

You can make concurrent requests to multiple models:

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        models = ["openai:gpt-4", "anthropic:claude-3"]
        messages = [{"role": "user", "content": "Hello!"}]
        
        tasks = [
            client.chat.completions.create(model=model, messages=messages)
            for model in models
        ]
        responses = await asyncio.gather(*tasks)
        
        for model, response in zip(models, responses):
            print(f"{model}: {response.choices[0].message.content}")

asyncio.run(main())
```

### Streaming Responses

The async client also supports streaming responses:

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        async for chunk in client.chat.completions.create(
            model="openai:gpt-4",
            messages=[{"role": "user", "content": "Tell me a story."}],
            stream=True
        ):
            print(chunk.choices[0].message.content, end="", flush=True)

asyncio.run(main())
```

### Concurrent Streaming

You can even stream from multiple models concurrently:

```python
import asyncio
from aisuite import AsyncClient

async def main():
    async with AsyncClient() as client:
        models = ["openai:gpt-4", "anthropic:claude-3"]
        messages = [{"role": "user", "content": "Tell me a story."}]

        async def process_stream(model):
            print(f"\nStream from {model}:")
            async for chunk in client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            ):
                print(chunk.choices[0].message.content, end="")

        tasks = [process_stream(model) for model in models]
        await asyncio.gather(*tasks)

asyncio.run(main())
```

The async client automatically manages resources and connections through its context manager interface. It's recommended to use the `async with` statement to ensure proper cleanup of resources.

## Examples

### Basic Usage

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

### Async Streaming Example

The `async_streaming_large.py` example demonstrates how to stream large responses from GPT-4 with real-time progress tracking and formatted output. Features include:

- Real-time streaming of responses
- Progress tracking with current section display
- Formatted output with colored headings and subheadings
- Performance statistics (tokens per second, total time, etc.)

To run the streaming example:

```bash
python3 examples/async_streaming_large.py --api-key YOUR_API_KEY
```

The example will:
1. Generate a detailed technical analysis of multiple topics
2. Stream the response in real-time with a progress indicator
3. Format headings in cyan and subheadings in yellow
4. Display performance statistics when complete

Required packages:
- rich (for console formatting)
- openai (for API access)
- aisuite (this library)

Example output:
```
⠋ Streaming response... Section: The Evolution of Artificial Intelligence
THE EVOLUTION OF ARTIFICIAL INTELLIGENCE
Historical Background: The field of artificial intelligence...
[Statistics shown when complete]
```

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

#### Examples

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
