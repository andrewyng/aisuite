"""
End-to-end integration tests for MCP with aisuite.

These tests verify the complete flow of using MCP tools with aisuite's
chat.completions.create() API, including:
- Config dict format
- Mixing MCP tools with Python functions
- Multiple MCP servers with prefixing
- Automatic cleanup

Requirements:
    - Node.js and npx must be installed
    - Tests are marked with @pytest.mark.integration
    - Run with: pytest tests/mcp/test_e2e.py -v -m integration
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from aisuite import Client


def create_mock_response(content="Test response", tool_calls=None):
    """Helper to create a mock chat completion response."""
    # Create a simple mock object that mimics the response structure
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].intermediate_messages = [response.choices[0].message]
    response.intermediate_responses = []

    return response


@pytest.mark.integration
class TestMCPConfigDictFormat:
    """Test using MCP config dict format in chat.completions.create()."""

    def test_basic_config_dict(self, temp_test_dir, skip_if_no_npx):
        """Test basic MCP config dict usage."""
        client = Client()

        # Mock the provider to avoid actual LLM API calls
        with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
            mock_runner.return_value = create_mock_response("Files listed successfully")

            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "List all files"}],
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
                    }
                ],
                max_turns=2,
            )

            # Verify response
            assert response.choices[0].message.content == "Files listed successfully"

            # Verify tool_runner was called with processed tools
            assert mock_runner.called
            call_args = mock_runner.call_args
            tools_arg = call_args[0][3]  # tools is 4th positional arg

            # Verify tools were converted to callables
            assert isinstance(tools_arg, list)
            assert all(callable(t) for t in tools_arg)

    def test_config_dict_with_allowed_tools(self, temp_test_dir, skip_if_no_npx):
        """Test MCP config dict with allowed_tools filtering."""
        client = Client()

        with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
            mock_runner.return_value = create_mock_response("File read successfully")

            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "Read test.txt"}],
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
                        "allowed_tools": ["read_file"],  # Only allow reading
                    }
                ],
                max_turns=2,
            )

            assert response.choices[0].message.content == "File read successfully"

            # Verify only read_file tool was passed
            call_args = mock_runner.call_args
            tools_arg = call_args[0][3]

            assert len(tools_arg) == 1
            assert tools_arg[0].__name__ == "read_file"

    def test_config_dict_with_prefixing(self, temp_test_dir, skip_if_no_npx):
        """Test MCP config dict with tool name prefixing."""
        client = Client()

        with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
            mock_runner.return_value = create_mock_response("Success")

            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                tools=[
                    {
                        "type": "mcp",
                        "name": "docs",
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            temp_test_dir,
                        ],
                        "use_tool_prefix": True,
                    }
                ],
                max_turns=2,
            )

            # Verify tools have prefixes
            call_args = mock_runner.call_args
            tools_arg = call_args[0][3]

            tool_names = [t.__name__ for t in tools_arg]
            assert any(name.startswith("docs__") for name in tool_names)


@pytest.mark.integration
class TestMCPWithPythonFunctions:
    """Test mixing MCP tools with regular Python functions."""

    def test_mix_mcp_and_python_functions(self, temp_test_dir, skip_if_no_npx):
        """Test using MCP config dict alongside Python functions."""
        client = Client()

        # Define a Python function
        def get_current_time() -> str:
            """Get the current time."""
            return "2025-01-01 12:00:00"

        with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
            mock_runner.return_value = create_mock_response("Mixed tools work!")

            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "What time is it?"}],
                tools=[
                    get_current_time,  # Python function
                    {
                        "type": "mcp",
                        "name": "filesystem",
                        "command": "npx",
                        "args": [
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            temp_test_dir,
                        ],
                    },  # MCP config
                ],
                max_turns=2,
            )

            assert response.choices[0].message.content == "Mixed tools work!"

            # Verify both types of tools were passed
            call_args = mock_runner.call_args
            tools_arg = call_args[0][3]

            # Should have Python function + MCP tools
            assert len(tools_arg) > 1

            # Verify Python function is in there
            assert any(t.__name__ == "get_current_time" for t in tools_arg)

            # Verify MCP tools are in there
            assert any(t.__name__ == "read_file" for t in tools_arg)


@pytest.mark.integration
class TestMultipleMCPServers:
    """Test using multiple MCP servers simultaneously."""

    def test_multiple_servers_with_prefixing(self, temp_test_dir, skip_if_no_npx):
        """Test multiple MCP servers with tool name prefixing to avoid collisions."""
        import tempfile

        client = Client()

        # Create a second temp directory
        with tempfile.TemporaryDirectory() as temp_dir_2:
            with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
                mock_runner.return_value = create_mock_response(
                    "Multiple servers work!"
                )

                response = client.chat.completions.create(
                    model="openai:gpt-4o",
                    messages=[{"role": "user", "content": "Compare directories"}],
                    tools=[
                        {
                            "type": "mcp",
                            "name": "dir1",
                            "command": "npx",
                            "args": [
                                "-y",
                                "@modelcontextprotocol/server-filesystem",
                                temp_test_dir,
                            ],
                            "use_tool_prefix": True,
                        },
                        {
                            "type": "mcp",
                            "name": "dir2",
                            "command": "npx",
                            "args": [
                                "-y",
                                "@modelcontextprotocol/server-filesystem",
                                temp_dir_2,
                            ],
                            "use_tool_prefix": True,
                        },
                    ],
                    max_turns=2,
                )

                assert response.choices[0].message.content == "Multiple servers work!"

                # Verify tools from both servers with prefixes
                call_args = mock_runner.call_args
                tools_arg = call_args[0][3]

                tool_names = [t.__name__ for t in tools_arg]

                # Should have tools from both servers
                assert any(name.startswith("dir1__") for name in tool_names)
                assert any(name.startswith("dir2__") for name in tool_names)

                # Should have both read_file tools with different prefixes
                assert "dir1__read_file" in tool_names
                assert "dir2__read_file" in tool_names


@pytest.mark.integration
class TestMCPCleanup:
    """Test that MCP clients are properly cleaned up."""

    def test_cleanup_after_success(self, temp_test_dir, skip_if_no_npx):
        """Test MCP clients are cleaned up after successful request."""
        client = Client()

        with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
            mock_runner.return_value = create_mock_response("Success")

            # Patch MCPClient to track close() calls
            with patch("aisuite.client.MCPClient") as mock_mcp_class:
                mock_mcp_instance = MagicMock()
                mock_mcp_class.from_config.return_value = mock_mcp_instance
                mock_mcp_instance.get_callable_tools.return_value = []

                response = client.chat.completions.create(
                    model="openai:gpt-4o",
                    messages=[{"role": "user", "content": "Test"}],
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
                        }
                    ],
                    max_turns=2,
                )

                # Verify MCP client was used as context manager (cleanup called)
                mock_mcp_instance.__enter__.assert_called_once()
                mock_mcp_instance.__exit__.assert_called_once()

    def test_cleanup_after_error(self, temp_test_dir, skip_if_no_npx):
        """Test MCP clients are cleaned up even after error."""
        client = Client()

        with patch.object(client.chat.completions, "_tool_runner") as mock_runner:
            # Make tool_runner raise an error
            mock_runner.side_effect = ValueError("Test error")

            # Patch MCPClient to track close() calls
            with patch("aisuite.client.MCPClient") as mock_mcp_class:
                mock_mcp_instance = MagicMock()
                mock_mcp_class.from_config.return_value = mock_mcp_instance
                mock_mcp_instance.get_callable_tools.return_value = []

                with pytest.raises(ValueError, match="Test error"):
                    client.chat.completions.create(
                        model="openai:gpt-4o",
                        messages=[{"role": "user", "content": "Test"}],
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
                            }
                        ],
                        max_turns=2,
                    )

                # Even after error, MCP client context manager exit should be called
                mock_mcp_instance.__enter__.assert_called_once()
                mock_mcp_instance.__exit__.assert_called_once()


@pytest.mark.integration
class TestMCPErrorHandling:
    """Test error handling for MCP integration."""

    def test_invalid_mcp_config_raises_error(self):
        """Test that invalid MCP config raises clear error."""
        client = Client()

        with pytest.raises(ValueError, match="must have 'name'"):
            client.chat.completions.create(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                tools=[
                    {
                        "type": "mcp",
                        # Missing 'name' field
                        "command": "npx",
                        "args": ["server"],
                    }
                ],
                max_turns=2,
            )

    def test_mcp_not_installed_raises_error(self, temp_test_dir, skip_if_no_npx):
        """Test that helpful error is raised if MCP package not installed."""
        client = Client()

        # Simulate MCP not being installed
        with patch("aisuite.client.MCP_AVAILABLE", False):
            with pytest.raises(ImportError, match="mcp.*package"):
                client.chat.completions.create(
                    model="openai:gpt-4o",
                    messages=[{"role": "user", "content": "Test"}],
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
                        }
                    ],
                    max_turns=2,
                )
