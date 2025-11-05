"""
Integration tests for MCPClient.

These tests use the real Anthropic filesystem MCP server
(@modelcontextprotocol/server-filesystem) to verify that MCPClient
can connect to, discover tools from, and execute tools on real MCP servers.

Requirements:
    - Node.js and npx must be installed
    - Tests are marked with @pytest.mark.integration
    - Run with: pytest tests/mcp/test_client.py -v -m integration
"""

import pytest
from aisuite.mcp import MCPClient
from aisuite.mcp.config import validate_mcp_config


@pytest.mark.integration
class TestMCPClientConnection:
    """Test MCPClient connection and basic functionality."""

    def test_connect_to_filesystem_server(self, temp_test_dir, skip_if_no_npx):
        """Test connecting to real Anthropic filesystem MCP server."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
            name="test_filesystem",
        )

        try:
            # Verify client is connected
            assert mcp._session is not None
            assert mcp.name == "test_filesystem"

            # List tools
            tools = mcp.list_tools()
            assert len(tools) > 0

            # Verify expected tools are present
            tool_names = [t["name"] for t in tools]
            assert "read_file" in tool_names
            assert "list_directory" in tool_names

            # Verify tools have descriptions
            for tool in tools:
                assert "name" in tool
                assert "description" in tool or "inputSchema" in tool

        finally:
            mcp.close()

    def test_list_tools_returns_schemas(self, temp_test_dir, skip_if_no_npx):
        """Test that list_tools returns proper tool schemas."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            tools = mcp.list_tools()

            # Find read_file tool
            read_file_tool = next((t for t in tools if t["name"] == "read_file"), None)
            assert read_file_tool is not None

            # Verify it has an input schema
            assert "inputSchema" in read_file_tool
            assert "properties" in read_file_tool["inputSchema"]

        finally:
            mcp.close()

    def test_context_manager(self, temp_test_dir, skip_if_no_npx):
        """Test MCPClient as context manager."""
        with MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        ) as mcp:
            tools = mcp.list_tools()
            assert len(tools) > 0

        # After exiting context, session should be closed
        # (We don't have a good way to verify this without inspecting internals)


@pytest.mark.integration
class TestMCPClientToolExecution:
    """Test executing tools via MCPClient."""

    def test_call_read_file_tool(self, temp_test_dir, skip_if_no_npx):
        """Test calling the read_file tool."""
        import os

        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            # Call read_file tool with absolute path
            test_file_path = os.path.join(temp_test_dir, "test.txt")
            result = mcp.call_tool("read_file", {"path": test_file_path})

            # Verify result contains file content
            assert "Hello from MCP test!" in result

        finally:
            mcp.close()

    def test_call_list_directory_tool(self, temp_test_dir, skip_if_no_npx):
        """Test calling the list_directory tool."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            # Call list_directory tool with absolute path
            result = mcp.call_tool("list_directory", {"path": temp_test_dir})

            # Verify result contains our test files
            assert "test.txt" in result or "README.md" in result

        finally:
            mcp.close()


@pytest.mark.integration
class TestMCPClientCallableTools:
    """Test getting callable tools from MCPClient."""

    def test_get_callable_tools(self, temp_test_dir, skip_if_no_npx):
        """Test getting all tools as callables."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            tools = mcp.get_callable_tools()

            # Verify we got callables
            assert len(tools) > 0
            assert all(callable(t) for t in tools)

            # Verify callables have expected attributes
            for tool in tools:
                assert hasattr(tool, "__name__")
                assert hasattr(tool, "__doc__")
                assert hasattr(tool, "__annotations__")

            # Find read_file callable
            read_file = next((t for t in tools if t.__name__ == "read_file"), None)
            assert read_file is not None

            # Test calling it with absolute path
            import os

            test_file_path = os.path.join(temp_test_dir, "test.txt")
            result = read_file(path=test_file_path)
            assert "Hello from MCP test!" in result

        finally:
            mcp.close()

    def test_get_callable_tools_with_filtering(self, temp_test_dir, skip_if_no_npx):
        """Test filtering tools with allowed_tools parameter."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            # Get only read_file tool
            tools = mcp.get_callable_tools(allowed_tools=["read_file"])

            # Should only get one tool
            assert len(tools) == 1
            assert tools[0].__name__ == "read_file"

            # Test it works with absolute path
            import os

            test_file_path = os.path.join(temp_test_dir, "test.txt")
            result = tools[0](path=test_file_path)
            assert "Hello from MCP test!" in result

        finally:
            mcp.close()

    def test_get_callable_tools_with_prefixing(self, temp_test_dir, skip_if_no_npx):
        """Test tool name prefixing with use_tool_prefix."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
            name="filesystem",
        )

        try:
            # Get tools with prefixing
            tools = mcp.get_callable_tools(use_tool_prefix=True)

            # Verify tools are prefixed
            tool_names = [t.__name__ for t in tools]
            assert any(name.startswith("filesystem__") for name in tool_names)
            assert "filesystem__read_file" in tool_names

        finally:
            mcp.close()

    def test_get_specific_tool(self, temp_test_dir, skip_if_no_npx):
        """Test getting a specific tool by name."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            # Get specific tool
            read_file = mcp.get_tool("read_file")

            assert read_file is not None
            assert callable(read_file)
            assert read_file.__name__ == "read_file"

            # Test it works with absolute path
            import os

            readme_path = os.path.join(temp_test_dir, "README.md")
            result = read_file(path=readme_path)
            assert "Test Directory" in result

            # Test getting non-existent tool
            fake_tool = mcp.get_tool("nonexistent_tool")
            assert fake_tool is None

        finally:
            mcp.close()


@pytest.mark.integration
class TestMCPClientFromConfig:
    """Test creating MCPClient from configuration dict."""

    def test_from_config_stdio(self, temp_test_dir, skip_if_no_npx):
        """Test creating MCPClient from config dict with stdio transport."""
        config = {
            "type": "mcp",
            "name": "test_filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        }

        mcp = MCPClient.from_config(config)

        try:
            assert mcp.name == "test_filesystem"
            tools = mcp.list_tools()
            assert len(tools) > 0

        finally:
            mcp.close()

    def test_from_config_with_env(self, temp_test_dir, skip_if_no_npx):
        """Test creating MCPClient with environment variables."""
        import os

        config = {
            "type": "mcp",
            "name": "test_filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
            "env": {"TEST_VAR": "test_value"},
        }

        mcp = MCPClient.from_config(config)

        try:
            tools = mcp.list_tools()
            assert len(tools) > 0

        finally:
            mcp.close()

    def test_get_tools_from_config(self, temp_test_dir, skip_if_no_npx):
        """Test get_tools_from_config convenience method."""
        config = {
            "type": "mcp",
            "name": "test_filesystem",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
            "allowed_tools": ["read_file"],
            "use_tool_prefix": True,
        }

        # Note: This creates a client internally and doesn't provide a way to close it
        # In production, this would be managed by the Completions class
        tools = MCPClient.get_tools_from_config(config)

        assert len(tools) == 1
        assert tools[0].__name__ == "test_filesystem__read_file"
        assert callable(tools[0])

        # Test the tool works with absolute path
        import os

        test_file_path = os.path.join(temp_test_dir, "test.txt")
        result = tools[0](path=test_file_path)
        assert "Hello from MCP test!" in result


@pytest.mark.integration
class TestMCPClientErrorHandling:
    """Test error handling in MCPClient."""

    def test_invalid_command_raises_error(self, temp_test_dir):
        """Test that invalid command raises appropriate error."""
        with pytest.raises(Exception):
            # This should fail to connect
            mcp = MCPClient(
                command="nonexistent_command_12345",
                args=["--test"],
            )

    def test_call_nonexistent_tool_returns_error(self, temp_test_dir, skip_if_no_npx):
        """Test that calling non-existent tool returns error or raises exception."""
        mcp = MCPClient(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir],
        )

        try:
            # Calling a non-existent tool should either raise an error or return error message
            try:
                result = mcp.call_tool("nonexistent_tool_xyz_123", {})
                # If it doesn't raise, the result should contain an error message
                assert "error" in result.lower() or "not found" in result.lower()
            except Exception:
                # It's also acceptable to raise an exception
                pass

        finally:
            mcp.close()
