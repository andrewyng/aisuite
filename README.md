#  aisuite

[![PyPI](https://img.shields.io/pypi/v/aisuite)](https://pypi.org/project/aisuite/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`aisuite` is a lightweight Python library that provides a **unified API for working with multiple Generative AI providers**.  
It offers a consistent interface for models from *OpenAI, Anthropic, Google, Hugging Face, AWS, Cohere, Mistral, Ollama*, and others—abstracting away SDK differences, authentication details, and parameter variations.  
Its design is modeled after OpenAI’s API style, making it instantly familiar and easy to adopt.

`aisuite` lets developers build and **run LLM-based or agentic applications across providers** with minimal setup.  
While it’s not a full-blown agents framework, it includes simple abstractions for creating standalone, lightweight agents.  
It’s designed for low learning curve — so you can focus on building AI systems, not integrating APIs.

---

## Key Features

`aisuite` is designed to eliminate the complexity of working with multiple LLM providers while keeping your code simple and portable. Whether you're building a chatbot, an agentic application, or experimenting with different models, `aisuite` provides the abstractions you need without getting in your way.

* **Unified API for multiple model providers** – Write your code once and run it with any supported provider. Switch between OpenAI, Anthropic, Google, and others with a single parameter change.
* **Easy agentic app or agent creation** – Build multi-turn agentic applications using a single parameter `max_turns`. No need to manually manage tool execution loops.
* **Pass Tool calls easily** – Pass real Python functions instead of JSON specs; aisuite handles schema generation and execution automatically.
* **MCP tools** – Connect to MCP-based tools without writing boilerplate; aisuite handles connection, schema and execution seamlessly.
* **Modular and extensible provider architecture** – Add support for new providers with minimal code. The plugin-style architecture makes extensions straightforward.

---

## Installation

You can install just the base `aisuite` package, or install a provider's package along with `aisuite`.

Install just the base package without any provider SDKs:

```shell
pip install aisuite
```

Install aisuite with a specific provider (e.g., Anthropic):

```shell
pip install 'aisuite[anthropic]'
```

Install aisuite with all provider libraries:

```shell
pip install 'aisuite[all]'
```

## Setup

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
For a list of provider values, you can look at the directory - `aisuite/providers/`. The list of supported providers are of the format - `<provider>_provider.py` in that directory. We welcome providers to add support to this library by adding an implementation file in this directory. Please see section below for how to contribute.

For more examples, check out the `examples` directory where you will find several notebooks that you can run to experiment with the interface.

---

## Chat Completions

The chat API provides a high-level abstraction for model interactions. It supports all core parameters (`temperature`, `max_tokens`, `tools`, etc.) in a provider-agnostic way.

```python
response = client.chat.completions.create(
    model="google:gemini-pro",
    messages=[{"role": "user", "content": "Summarize this paragraph."}],
)
print(response.choices[0].message.content)
```

`aisuite` standardizes request and response structures so you can focus on logic rather than SDK differences.

---

## Tool Calling & Agentic apps

`aisuite` provides a simple abstraction for tool/function calling that works across supported providers. This is in addition to the regular abstraction of passing JSON spec of the tool to the `tools` parameter. The tool calling abstraction makes it easy to use tools with different LLMs without changing your code.

There are two ways to use tools with `aisuite`:

### 1. Manual Tool Handling

This is the default behavior when `max_turns` is not specified. In this mode, you have full control over the tool execution flow. You pass tools using the standard OpenAI JSON schema format, and `aisuite` returns the LLM's tool call requests in the response. You're then responsible for executing the tools, processing results, and sending them back to the model in subsequent requests.

This approach is useful when you need:
- Fine-grained control over tool execution logic
- Custom error handling or validation before executing tools
- The ability to selectively execute or skip certain tool calls
- Integration with existing tool execution pipelines

You can pass tools in the OpenAI tool format:

```python
def will_it_rain(location: str, time_of_day: str):
    """Check if it will rain in a location at a given time today.
    
    Args:
        location (str): Name of the city
        time_of_day (str): Time of the day in HH:MM format.
    """
    return "YES"

tools = [{
    "type": "function",
    "function": {
        "name": "will_it_rain",
        "description": "Check if it will rain in a location at a given time today",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Name of the city"
                },
                "time_of_day": {
                    "type": "string",
                    "description": "Time of the day in HH:MM format."
                }
            },
            "required": ["location", "time_of_day"]
        }
    }
}]

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=tools
)
```

### 2. Automatic Tool Execution

When `max_turns` is specified, you can pass a list of callable Python functions as the `tools` parameter. `aisuite` will automatically handle the tool calling flow:

```python
def will_it_rain(location: str, time_of_day: str):
    """Check if it will rain in a location at a given time today.
    
    Args:
        location (str): Name of the city
        time_of_day (str): Time of the day in HH:MM format.
    """
    return "YES"

client = ai.Client()
messages = [{
    "role": "user",
    "content": "I live in San Francisco. Can you check for weather "
               "and plan an outdoor picnic for me at 2pm?"
}]

# Automatic tool execution with max_turns
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=[will_it_rain],
    max_turns=2  # Maximum number of back-and-forth tool calls
)
print(response.choices[0].message.content)
```

When `max_turns` is specified, `aisuite` will:
1. Send your message to the LLM
2. Execute any tool calls the LLM requests
3. Send the tool results back to the LLM
4. Repeat until the conversation is complete or max_turns is reached

In addition to `response.choices[0].message`, there is an additional field `response.choices[0].intermediate_messages` which contains the list of all messages including tool interactions used. This can be used to continue the conversation with the model.
For more detailed examples of tool calling, check out the `examples/tool_calling_abstraction.ipynb` notebook.

### Model Context Protocol (MCP) Integration

`aisuite` natively supports **MCP**, a standard protocol that allows LLMs to securely call external tools and access data. You can connect to MCP servers—such as a filesystem or database—and expose their tools directly to your model.
Read more about MCP here - https://modelcontextprotocol.io/docs/getting-started/intro

Install aisuite with MCP support:

```shell
pip install 'aisuite[mcp]'
```

You'll also need an MCP server. For example, to use the filesystem server:

```shell
npm install -g @modelcontextprotocol/server-filesystem
```

There are two ways to use MCP tools with aisuite:

#### Option 1: Config Dict Format (Recommended for Simple Use Cases)

```python
import aisuite as ai

client = ai.Client()
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "List the files in the current directory"}],
    tools=[{
        "type": "mcp",
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    }],
    max_turns=3
)

print(response.choices[0].message.content)
```

#### Option 2: Explicit MCPClient (Recommended for Advanced Use Cases)

```python
import aisuite as ai
from aisuite.mcp import MCPClient

# Create MCP client once, reuse across requests
mcp = MCPClient(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)

# Use with aisuite
client = ai.Client()
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "List the files"}],
    tools=mcp.get_callable_tools(),
    max_turns=3
)

print(response.choices[0].message.content)
mcp.close()  # Clean up
```

For detailed usage (security filters, tool prefixing, and `MCPClient` management), see [docs/mcp-tools.md](docs/mcp-tools.md).
For detailed examples, see `examples/mcp_tools_example.ipynb`.

---

## Extending aisuite: Adding a Provider

New providers can be added by implementing a lightweight adapter. The system uses a naming convention for discovery:

| Element         | Convention                         |
| --------------- | ---------------------------------- |
| **Module file** | `<provider>_provider.py`           |
| **Class name**  | `<Provider>Provider` (capitalized) |

Example:

```python
# providers/openai_provider.py
class OpenaiProvider(BaseProvider):
    ...
```

This convention ensures consistency and enables automatic loading of new integrations.

---

## Contributing

Contributions are welcome. Please review the [Contributing Guide](https://github.com/andrewyng/aisuite/blob/main/CONTRIBUTING.md) and join our [Discord](https://discord.gg/T6Nvn8ExSb) for discussions.

---

## License

Released under the **MIT License** — free for commercial and non-commercial use.

---
