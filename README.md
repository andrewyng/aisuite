# aisuite

[![PyPI](https://img.shields.io/pypi/v/aisuite)](https://pypi.org/project/aisuite/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Simple, unified interface to multiple Generative AI providers.

`aisuite` makes it easy for developers to interact with multiple Gen-AI services through a standardized interface. Using an interface similar to OpenAI's, `aisuite` supports **chat completions** and **audio transcription**, making it easy to work with the most popular AI providers and compare results. It is a thin wrapper around python client libraries, and allows creators to seamlessly swap out and test different providers without changing their code.

All of the top providers are supported.
Sample list of supported providers include - Anthropic, AWS, Azure, Cerebras, Cohere, Google, Groq, HuggingFace, Ollama, Mistral, OpenAI, Sambanova, Watsonx and others.

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

## Tool Calling

`aisuite` provides a simple abstraction for tool/function calling that works across supported providers. This is in addition to the regular abstraction of passing JSON spec of the tool to the `tools` parameter. The tool calling abstraction makes it easy to use tools with different LLMs without changing your code.

There are two ways to use tools with `aisuite`:

### 1. Manual Tool Handling

This is the default behavior when `max_turns` is not specified.
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

In addition to `response.choices[0].message`, there is an additional field `response.choices[0].intermediate_messages`: which contains the list of all messages including tool interactions used. This can be used to continue the conversation with the model.
For more detailed examples of tool calling, check out the `examples/tool_calling_abstraction.ipynb` notebook.

## Audio Transcription

> **Note:** Audio transcription support is currently under development. The API and features described below are subject to change.

`aisuite` provides audio transcription (speech-to-text) with the same unified interface pattern used for chat completions. Transcribe audio files across multiple providers with consistent code.

### Basic Usage

```python
import aisuite as ai
client = ai.Client()

# Transcribe an audio file
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="meeting.mp3"
)
print(result.text)

# Switch providers without changing your code
result = client.audio.transcriptions.create(
    model="deepgram:nova-2",
    file="meeting.mp3"
)
print(result.text)
```

### Common Parameters

Use OpenAI-style parameters that work across all providers:

```python
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="interview.mp3",
    language="en",           # Specify audio language
    prompt="Technical discussion about AI",  # Context hints
    temperature=0.2          # Sampling temperature (where supported)
)
```

These parameters are automatically mapped to each provider's native format.

### Provider-Specific Features

Each provider offers unique capabilities you can access directly:

**OpenAI Whisper:**
```python
result = client.audio.transcriptions.create(
    model="openai:whisper-1",
    file="speech.mp3",
    response_format="verbose_json",       # Get detailed metadata
    timestamp_granularities=["word"]      # Word-level timestamps
)
```

**Deepgram:**
```python
result = client.audio.transcriptions.create(
    model="deepgram:nova-2",
    file="meeting.mp3",
    punctuate=True,                       # Auto-add punctuation
    diarize=True,                         # Identify speakers
    sentiment=True,                       # Sentiment analysis
    summarize=True                        # Auto-summarization
)
```

**Google Speech-to-Text:**
```python
result = client.audio.transcriptions.create(
    model="google:default",
    file="call.mp3",
    enable_automatic_punctuation=True,
    enable_speaker_diarization=True,
    diarization_speaker_count=2
)
```

**Hugging Face:**
```python
result = client.audio.transcriptions.create(
    model="huggingface:openai/whisper-large-v3",
    file="presentation.mp3",
    return_timestamps="word"                  # Word-level timestamps
)
```

### Streaming Transcription

For real-time or large audio files, use streaming:

```python
async def transcribe_stream():
    stream = client.audio.transcriptions.create_stream_output(
        model="deepgram:nova-2",
        file="long_recording.mp3"
    )

    async for chunk in stream:
        print(chunk.text, end="", flush=True)
        if chunk.is_final:
            print()  # New line for final results

# Run the async function
import asyncio
asyncio.run(transcribe_stream())
```

### Supported Providers

- **OpenAI**: `whisper-1`
- **Deepgram**: `nova-2`, `nova`, `enhanced`, `base`
- **Google**: `default`, `latest_long`, `latest_short`
- **Hugging Face**: `openai/whisper-large-v3`, `openai/whisper-tiny`, `facebook/wav2vec2-base-960h`, `facebook/wav2vec2-large-xlsr-53`

### Installation

Install transcription providers:

```shell
# Install with specific provider
pip install 'aisuite[openai]'      # For OpenAI Whisper
pip install 'aisuite[deepgram]'    # For Deepgram
pip install 'aisuite[google]'      # For Google Speech-to-Text
pip install 'aisuite[huggingface]' # For Hugging Face models

# Install all providers
pip install 'aisuite[all]'
```

Set API keys:

```shell
export OPENAI_API_KEY="your-openai-api-key"
export DEEPGRAM_API_KEY="your-deepgram-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export HF_TOKEN="your-huggingface-token"
```

For more examples and advanced usage, check out `examples/asr_example.ipynb`.

## MCP (Model Context Protocol) Integration

`aisuite` provides seamless integration with MCP servers, allowing AI models to access external tools and data sources through the Model Context Protocol. MCP tools work exactly like regular Python functions in aisuite's tool calling system.

### What is MCP?

MCP (Model Context Protocol) is a standardized protocol that enables AI applications to connect to external data sources and tools. MCP servers expose tools, resources, and prompts that AI models can use to interact with filesystems, databases, APIs, and more.

### Installation

Install aisuite with MCP support:

```shell
pip install 'aisuite[mcp]'
```

You'll also need an MCP server. For example, to use the filesystem server:

```shell
npm install -g @modelcontextprotocol/server-filesystem
```

### Basic Usage

```python
import aisuite as ai
from aisuite.mcp import MCPClient

# Connect to an MCP server
mcp = MCPClient(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)

# Use MCP tools with any provider
client = ai.Client()
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "List the files in the current directory"}],
    tools=mcp.get_callable_tools(),  # MCP tools work like Python functions!
    max_turns=3
)

print(response.choices[0].message.content)
```

### Mixing MCP Tools with Python Functions

You can seamlessly combine MCP tools with regular Python functions:

```python
# Define a custom Python function
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Mix MCP and Python tools
all_tools = mcp.get_callable_tools() + [get_current_time]

response = client.chat.completions.create(
    model="anthropic:claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content": "What time is it? Also list the files."}],
    tools=all_tools,
    max_turns=3
)
```

### Using Specific MCP Tools

You can select specific tools instead of using all available tools:

```python
# Get only specific tools
read_file = mcp.get_tool("read_file")
write_file = mcp.get_tool("write_file")

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=[read_file, write_file],
    max_turns=2
)
```

### Multiple MCP Servers

Connect to multiple MCP servers and combine their tools:

```python
# Connect to different MCP servers
filesystem_mcp = MCPClient(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/docs"]
)

database_mcp = MCPClient(
    command="python",
    args=["path/to/database_server.py"]
)

# Combine tools from multiple sources
all_tools = (
    filesystem_mcp.get_callable_tools() +
    database_mcp.get_callable_tools()
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=messages,
    tools=all_tools,
    max_turns=5
)
```

### Context Manager (Recommended)

Use MCPClient as a context manager to ensure proper cleanup:

```python
with MCPClient(command="npx", args=["server"]) as mcp:
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=messages,
        tools=mcp.get_callable_tools(),
        max_turns=2
    )
```

### Provider Compatibility

MCP tools work with all aisuite providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- And all other supported providers

### Available MCP Servers

Explore available MCP servers:
- **Filesystem**: Access local files and directories
- **Database**: Query databases (PostgreSQL, MySQL, SQLite)
- **GitHub**: Interact with GitHub repositories
- **Google Drive**: Access Google Drive files
- **Slack**: Send and read Slack messages
- And many more at: https://github.com/modelcontextprotocol/servers

For detailed examples, see `examples/mcp_tools_example.ipynb`.

## License

aisuite is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Contributing

If you would like to contribute, please read our [Contributing Guide](https://github.com/andrewyng/aisuite/blob/main/CONTRIBUTING.md) and join our [Discord](https://discord.gg/T6Nvn8ExSb) server!
