"""
Real LLM End-to-End Tests for MCP Integration.

These tests make ACTUAL API calls to LLM providers (OpenAI, Anthropic) to verify
that MCP tools work correctly with real models. Unlike test_e2e.py which mocks
LLM responses, these tests verify the complete integration stack.

⚠️ WARNING: These tests will make real API calls and incur costs!
   - Each test costs ~$0.01-0.05 depending on the model
   - Tests are marked with @pytest.mark.llm
   - Tests are skipped if API keys are not present

Requirements:
    - Node.js and npx (for MCP filesystem server)
    - API keys in .env file:
        OPENAI_API_KEY=your-key
        ANTHROPIC_API_KEY=your-key
    - pytest-asyncio, python-dotenv

Running:
    # Run ONLY LLM tests (⚠️ costs money):
    pytest tests/mcp/test_llm_e2e.py -v -m llm

    # Skip LLM tests (default, free):
    pytest tests/mcp/ -v -m "integration and not llm"
"""

import pytest
import os
from pathlib import Path
from aisuite import Client


# Helper function to check if we have API keys
def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.mark.llm
@pytest.mark.integration
class TestOpenAIWithMCP:
    """Test OpenAI models with real MCP tools."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_gpt4o_reads_file_via_mcp(self, temp_test_dir, skip_if_no_npx):
        """Test GPT-4o can read a file using MCP filesystem tools."""
        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f'Use read_file to read the file at path "{temp_test_dir}/test.txt" and tell me what it contains.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        temp_test_dir,
                    ],
                    "allowed_tools": ["read_file"],  # Security: only allow reading
                }
            ],
            max_turns=3,
        )

        # Debug: Print intermediate messages to see what happened
        if hasattr(response.choices[0], "intermediate_messages"):
            print("\n=== Intermediate Messages ===")
            import json

            for i, msg in enumerate(response.choices[0].intermediate_messages):
                print(f"\nMessage {i}:")
                # Handle both dict and object formats
                if isinstance(msg, dict):
                    print(json.dumps(msg, indent=2, default=str))
                else:
                    print(f"Role: {msg.role}")
                    if hasattr(msg, "content") and msg.content:
                        print(f"Content: {msg.content[:200]}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(
                                f"Tool Call: {tc.function.name}({tc.function.arguments})"
                            )

        # Verify the LLM actually read the file
        content = response.choices[0].message.content.lower()
        assert (
            "hello from mcp test" in content or "hello from mcp" in content
        ), f"Expected file content in response, got: {content}"

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_gpt4o_lists_files_via_mcp(self, temp_test_dir, skip_if_no_npx):
        """Test GPT-4o can list directory contents using MCP tools."""
        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f'Use list_directory to list all files in the directory at path "{temp_test_dir}" and tell me what you find.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        temp_test_dir,
                    ],
                    "allowed_tools": ["list_directory"],  # Security: only allow listing
                }
            ],
            max_turns=3,
        )

        # Verify the LLM found the test files
        content = response.choices[0].message.content.lower()
        # Test dir has: test.txt, README.md, data.json, subdir/
        assert (
            "test.txt" in content or "readme" in content
        ), f"Expected file names in response, got: {content}"

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_gpt4o_mixed_tools(self, temp_test_dir, skip_if_no_npx):
        """Test GPT-4o with both MCP tools and regular Python functions."""

        # Define a Python function
        def get_current_date() -> str:
            """Get the current date in YYYY-MM-DD format."""
            from datetime import datetime

            return datetime.now().strftime("%Y-%m-%d")

        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f'First use get_current_date to get today\'s date, then use read_file to read "{temp_test_dir}/test.txt" and tell me both.',
                }
            ],
            tools=[
                get_current_date,  # Python function
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        temp_test_dir,
                    ],
                    "allowed_tools": ["read_file"],
                },
            ],
            max_turns=5,
        )

        # Verify both tools were used
        content = response.choices[0].message.content.lower()
        # Should mention the date (from Python function)
        assert any(
            str(y) in content for y in [2024, 2025, 2026]
        ), f"Expected date in response, got: {content}"
        # Should mention the file content (from MCP tool)
        assert (
            "hello" in content or "mcp test" in content
        ), f"Expected file content in response, got: {content}"


@pytest.mark.llm
@pytest.mark.integration
class TestAnthropicWithMCP:
    """Test Anthropic Claude models with real MCP tools."""

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_claude_reads_file_via_mcp(self, temp_test_dir, skip_if_no_npx):
        """Test Claude can read a file using MCP filesystem tools."""
        client = Client()

        response = client.chat.completions.create(
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": f'Use read_file to read the file at path "{temp_test_dir}/test.txt" and tell me what it contains.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        temp_test_dir,
                    ],
                    "allowed_tools": ["read_file"],
                }
            ],
            max_turns=3,
        )

        # Verify Claude actually read the file
        content = response.choices[0].message.content.lower()
        assert (
            "hello from mcp test" in content or "hello from mcp" in content
        ), f"Expected file content in response, got: {content}"

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_claude_lists_files_via_mcp(self, temp_test_dir, skip_if_no_npx):
        """Test Claude can list directory contents using MCP tools."""
        client = Client()

        response = client.chat.completions.create(
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": f'Use list_directory with path "{temp_test_dir}" to list all files.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        temp_test_dir,
                    ],
                    "allowed_tools": ["list_directory"],
                }
            ],
            max_turns=3,
        )

        # Verify Claude found the test files
        content = response.choices[0].message.content.lower()
        assert (
            "test.txt" in content or "readme" in content or "data.json" in content
        ), f"Expected file names in response, got: {content}"

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_claude_mixed_tools(self, temp_test_dir, skip_if_no_npx):
        """Test Claude with both MCP tools and regular Python functions."""

        def get_weather(location: str) -> str:
            """Get the weather for a location (mock function).

            Args:
                location: The city name
            """
            # Mock weather function for testing
            return f"The weather in {location} is sunny and 72°F"

        client = Client()

        response = client.chat.completions.create(
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": f'Use get_weather for San Francisco, then use read_file to read "{temp_test_dir}/README.md". Tell me both results.',
                }
            ],
            tools=[
                get_weather,  # Python function
                {
                    "type": "mcp",
                    "name": "filesystem",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        temp_test_dir,
                    ],
                    "allowed_tools": ["read_file"],
                },
            ],
            max_turns=5,
        )

        # Verify both tools were used
        content = response.choices[0].message.content.lower()
        # Should mention weather (from Python function)
        assert (
            "weather" in content or "sunny" in content or "72" in content
        ), f"Expected weather info in response, got: {content}"
        # Should mention the README content (from MCP tool)
        assert (
            "test directory" in content or "readme" in content
        ), f"Expected README content in response, got: {content}"


@pytest.mark.llm
@pytest.mark.integration
class TestToolPrefixingWithLLM:
    """Test tool prefixing works with real LLMs."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_multiple_mcp_servers_with_prefixing(self, temp_test_dir, skip_if_no_npx):
        """Test using multiple MCP servers with prefixing to avoid name collisions."""
        # Create two subdirectories
        dir1 = Path(temp_test_dir) / "dir1"
        dir2 = Path(temp_test_dir) / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "file1.txt").write_text("Content from dir1")
        (dir2 / "file2.txt").write_text("Content from dir2")

        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f'Use dir1_fs__list_directory with path "{dir1}" to list dir1, then use dir2_fs__list_directory with path "{dir2}" to list dir2.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "dir1_fs",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(dir1),
                    ],
                    "use_tool_prefix": True,  # Tools will be "dir1_fs__list_directory", etc.
                    "allowed_tools": ["list_directory"],
                },
                {
                    "type": "mcp",
                    "name": "dir2_fs",
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(dir2),
                    ],
                    "use_tool_prefix": True,  # Tools will be "dir2_fs__list_directory", etc.
                    "allowed_tools": ["list_directory"],
                },
            ],
            max_turns=5,
        )

        # Verify the LLM found files from both directories
        content = response.choices[0].message.content.lower()
        assert (
            "file1" in content or "dir1" in content
        ), f"Expected dir1 content, got: {content}"
        assert (
            "file2" in content or "dir2" in content
        ), f"Expected dir2 content, got: {content}"
