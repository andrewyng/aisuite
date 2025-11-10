"""
MCP Tools with Config Dict Format - Example

This example demonstrates using MCP tools with the simplified config dict format.
Instead of explicitly creating MCPClient objects, you can pass MCP server configs
directly to the tools parameter.
"""

import os
from dotenv import load_dotenv
import aisuite as ai

# Load environment variables
load_dotenv()

# Create aisuite client
client = ai.Client()

print("=" * 70)
print("Example 1: Basic Config Dict Usage")
print("=" * 70)

# Instead of creating MCPClient explicitly, pass config dict directly!
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[
        {"role": "user", "content": "List all Python files in the current directory"}
    ],
    tools=[
        {
            "type": "mcp",
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
        }
    ],
    max_turns=2,
)

print(response.choices[0].message.content)

print("\n" + "=" * 70)
print("Example 2: Filtering Tools with allowed_tools")
print("=" * 70)

# Only allow specific tools for security
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Read the README.md file"}],
    tools=[
        {
            "type": "mcp",
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
            "allowed_tools": ["read_file"],  # Security: only allow reading, not writing
        }
    ],
    max_turns=2,
)

print(response.choices[0].message.content)

print("\n" + "=" * 70)
print("Example 3: Multiple MCP Servers with Tool Prefixing")
print("=" * 70)

import tempfile

temp_dir = tempfile.mkdtemp()

# Connect to two different filesystem servers with prefixing
# This avoids tool name collisions
response = client.chat.completions.create(
    model="anthropic:claude-3-5-sonnet-20240620",
    messages=[
        {
            "role": "user",
            "content": "How many files are in the current directory vs the temp directory?",
        }
    ],
    tools=[
        {
            "type": "mcp",
            "name": "current_dir",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
            "use_tool_prefix": True,  # Tools named "current_dir__list_directory", etc.
        },
        {
            "type": "mcp",
            "name": "temp_dir",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", temp_dir],
            "use_tool_prefix": True,  # Tools named "temp_dir__list_directory", etc.
        },
    ],
    max_turns=3,
)

print(response.choices[0].message.content)

print("\n" + "=" * 70)
print("Example 4: Mixing MCP Configs with Python Functions")
print("=" * 70)

from datetime import datetime


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_stats(numbers: list) -> dict:
    """Calculate basic statistics for a list of numbers.

    Args:
        numbers: List of numbers to analyze
    """
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers) if numbers else 0,
    }


# Mix everything: MCP configs + Python functions!
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "What time is it? Also, list all files in the current directory.",
        }
    ],
    tools=[
        get_current_time,  # Regular Python function
        calculate_stats,  # Another Python function
        {
            "type": "mcp",
            "name": "filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
        },  # MCP config dict
    ],
    max_turns=3,
)

print(response.choices[0].message.content)

print("\n" + "=" * 70)
print("Example 5: When to Use Config Dict vs MCPClient")
print("=" * 70)

print(
    """
Use Config Dict When:
✓ Quick prototypes and simple scripts
✓ One-off tool usage
✓ Don't need to reuse MCP client across multiple requests
✓ Want automatic cleanup
✓ Less code is better

Use Explicit MCPClient When:
✓ Need to reuse the same MCP connection across multiple requests
✓ Want to inspect available tools before using them
✓ Need fine-grained control over connection lifecycle
✓ Building a long-running application
✓ Want to manually manage resources

Example of explicit MCPClient:
"""
)

from aisuite.mcp import MCPClient

# Create once, reuse many times
mcp = MCPClient(
    command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()]
)

# Inspect available tools
print(f"\\nAvailable tools: {[t['name'] for t in mcp.list_tools()]}")

# Reuse across multiple requests
for query in ["List files", "Count files", "Check if README exists"]:
    response = client.chat.completions.create(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": query}],
        tools=mcp.get_callable_tools(),
        max_turns=2,
    )
    print(f"\\n{query}: {response.choices[0].message.content[:100]}...")

mcp.close()

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
