"""
MCP HTTP Transport Example

This example demonstrates how to use HTTP-based MCP servers with aisuite.

Prerequisites:
- An HTTP MCP server running (e.g., http://localhost:8000)
- OpenAI API key in .env file or OPENAI_API_KEY environment variable
- pip install 'aisuite[mcp]'
- pip install python-dotenv

Note: This example assumes you have an HTTP MCP server running.
If you don't have one, this is a demonstration of the API usage.
"""

import aisuite as ai
from aisuite.mcp import MCPClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_1_config_dict_format():
    """Example 1: Using HTTP MCP server with config dict format."""
    print("=" * 60)
    print("Example 1: HTTP MCP with Config Dict")
    print("=" * 60)

    client = ai.Client()

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Use the available tools to get the current weather data.",
            }
        ],
        tools=[
            {
                "type": "mcp",
                "name": "weather-api",
                "server_url": "http://localhost:8000/mcp/v1",  # Full endpoint URL
                "timeout": 30.0,  # Optional: request timeout in seconds
            }
        ],
        max_turns=3,
    )

    print(response.choices[0].message.content)
    print()


def example_2_explicit_mcp_client():
    """Example 2: Using HTTP MCP server with explicit MCPClient."""
    print("=" * 60)
    print("Example 2: HTTP MCP with Explicit MCPClient")
    print("=" * 60)

    # Create HTTP-based MCP client
    mcp = MCPClient(
        server_url="http://localhost:8000/mcp/v1",  # Full endpoint URL
        name="weather-api",
        timeout=30.0,
    )

    # List available tools
    print("Available tools:")
    for tool in mcp.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")
    print()

    # Use with aisuite
    client = ai.Client()
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "What tools are available?"}],
        tools=mcp.get_callable_tools(),
        max_turns=2,
    )

    print(response.choices[0].message.content)

    # Clean up
    mcp.close()
    print()


def example_3_with_authentication():
    """Example 3: HTTP MCP server with authentication headers."""
    print("=" * 60)
    print("Example 3: HTTP MCP with Authentication")
    print("=" * 60)

    # Get API token from environment
    api_token = os.getenv("MCP_API_TOKEN", "your-token-here")

    client = ai.Client()

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "Fetch the user profile using the API."}],
        tools=[
            {
                "type": "mcp",
                "name": "api-server",
                "server_url": "https://api.example.com/mcp/v1",  # Full endpoint URL
                "headers": {
                    "Authorization": f"Bearer {api_token}",
                    "X-API-Version": "2024-01",
                },
                "timeout": 60.0,
            }
        ],
        max_turns=3,
    )

    print(response.choices[0].message.content)
    print()


def example_4_context_manager():
    """Example 4: Using context manager for automatic cleanup."""
    print("=" * 60)
    print("Example 4: HTTP MCP with Context Manager")
    print("=" * 60)

    with MCPClient(
        server_url="http://localhost:8000/mcp/v1",
        name="api-server",  # Full endpoint URL
    ) as mcp:
        client = ai.Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "List available data."}],
            tools=mcp.get_callable_tools(),
            max_turns=2,
        )

        print(response.choices[0].message.content)
    # mcp.close() is called automatically
    print()


def example_5_mixing_http_and_python_functions():
    """Example 5: Mixing HTTP MCP tools with regular Python functions."""
    print("=" * 60)
    print("Example 5: Mixing HTTP MCP with Python Functions")
    print("=" * 60)

    # Define a custom Python function
    def get_current_time() -> str:
        """Get the current date and time in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    client = ai.Client()

    response = client.chat.completions.create(
        model="anthropic:claude-sonnet-4-5",
        messages=[
            {
                "role": "user",
                "content": "What time is it now? Also get the weather data from the API.",
            }
        ],
        tools=[
            get_current_time,  # Regular Python function
            {
                "type": "mcp",
                "name": "weather-api",
                "server_url": "http://localhost:8000/mcp/v1",  # Full endpoint URL
            },  # HTTP MCP server
        ],
        max_turns=3,
    )

    print(response.choices[0].message.content)
    print()


def example_6_tool_filtering():
    """Example 6: Using allowed_tools to restrict available tools."""
    print("=" * 60)
    print("Example 6: HTTP MCP with Tool Filtering")
    print("=" * 60)

    client = ai.Client()

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "Get the weather forecast."}],
        tools=[
            {
                "type": "mcp",
                "name": "api-server",
                "server_url": "http://localhost:8000/mcp/v1",  # Full endpoint URL
                "allowed_tools": ["get_weather"],  # Only allow this specific tool
            }
        ],
        max_turns=2,
    )

    print(response.choices[0].message.content)
    print()


def example_7_multiple_http_servers():
    """Example 7: Using multiple HTTP MCP servers with prefixing."""
    print("=" * 60)
    print("Example 7: Multiple HTTP MCP Servers with Prefixing")
    print("=" * 60)

    client = ai.Client()

    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Get weather data and user data.",
            }
        ],
        tools=[
            {
                "type": "mcp",
                "name": "weather",
                "server_url": "http://localhost:8000/mcp/v1",  # Full endpoint URL
                "use_tool_prefix": True,  # Tools: weather__get_forecast, etc.
            },
            {
                "type": "mcp",
                "name": "users",
                "server_url": "http://localhost:9000/mcp/v1",  # Full endpoint URL
                "use_tool_prefix": True,  # Tools: users__get_profile, etc.
            },
        ],
        max_turns=3,
    )

    print(response.choices[0].message.content)
    print()


if __name__ == "__main__":
    print("\nMCP HTTP Transport Examples")
    print("=" * 60)
    print()
    print("Note: These examples require an HTTP MCP server to be running.")
    print("Uncomment the examples you want to run.\n")

    # Uncomment the examples you want to run:

    # example_1_config_dict_format()
    # example_2_explicit_mcp_client()
    # example_3_with_authentication()
    # example_4_context_manager()
    # example_5_mixing_http_and_python_functions()
    # example_6_tool_filtering()
    # example_7_multiple_http_servers()

    print("\nTo run these examples:")
    print("1. Start an HTTP MCP server (e.g., on http://localhost:8000)")
    print("2. Set your OPENAI_API_KEY environment variable")
    print("3. Uncomment the example functions you want to run")
    print("4. Run: python examples/mcp_http_example.py")
