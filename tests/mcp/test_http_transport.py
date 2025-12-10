"""
Tests for MCP HTTP Transport.

These tests verify that the MCPClient works correctly with HTTP-based MCP servers.
All HTTP requests are mocked to avoid requiring a real HTTP MCP server.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import httpx
from aisuite.mcp.client import MCPClient


@pytest.mark.integration
class TestHTTPTransportBasics:
    """Test basic HTTP transport functionality."""

    def test_create_http_client_success(self):
        """Test creating an HTTP MCPClient with valid parameters."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            # Mock the HTTP client
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock initialize response
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                },
            }
            mock_response_init.raise_for_status = MagicMock()

            # Mock tools/list response
            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"param": {"type": "string"}},
                            },
                        }
                    ]
                },
            }
            mock_response_tools.raise_for_status = MagicMock()

            # Set up post responses in order (init + notification + tools)
            mock_client_instance.post = AsyncMock(
                side_effect=[
                    mock_response_init,
                    MagicMock(),  # initialized notification
                    mock_response_tools,
                ]
            )

            # Create client
            mcp = MCPClient(server_url="http://localhost:8000", name="test-server")

            # Verify client was created
            assert mcp.server_url == "http://localhost:8000"
            assert mcp.name == "test-server"
            assert len(mcp.list_tools()) == 1
            assert mcp.list_tools()[0]["name"] == "test_tool"

            # Cleanup
            mcp.close()

    def test_create_http_client_with_headers(self):
        """Test creating an HTTP MCPClient with custom headers."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            # Create client with headers
            headers = {"Authorization": "Bearer secret-token"}
            mcp = MCPClient(
                server_url="http://localhost:8000", headers=headers, name="test"
            )

            # Verify headers were stored
            assert mcp.headers == headers

            # Cleanup
            mcp.close()

    def test_http_client_validation_errors(self):
        """Test that validation errors are raised for invalid parameters."""
        # Test: no command or server_url
        with pytest.raises(ValueError, match="Must provide either"):
            MCPClient()

        # Test: both command and server_url
        with pytest.raises(ValueError, match="Cannot mix stdio parameters"):
            MCPClient(command="npx", server_url="http://localhost:8000")


@pytest.mark.integration
class TestHTTPToolCalling:
    """Test HTTP tool discovery and calling."""

    def test_list_tools_http(self):
        """Test listing tools via HTTP transport."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "tool1",
                            "description": "First tool",
                            "inputSchema": {},
                        },
                        {
                            "name": "tool2",
                            "description": "Second tool",
                            "inputSchema": {},
                        },
                    ]
                },
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            mcp = MCPClient(server_url="http://localhost:8000")

            tools = mcp.list_tools()
            assert len(tools) == 2
            assert tools[0]["name"] == "tool1"
            assert tools[1]["name"] == "tool2"

            mcp.close()

    def test_call_tool_http(self):
        """Test calling a tool via HTTP transport."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock init and tools/list
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo tool",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                            },
                        }
                    ]
                },
            }
            mock_response_tools.raise_for_status = MagicMock()

            # Mock tool call response
            mock_response_call = MagicMock()
            mock_response_call.headers = {"content-type": "application/json"}
            mock_response_call.json.return_value = {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"content": [{"text": "Hello, World!"}]},
            }
            mock_response_call.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[
                    mock_response_init,
                    MagicMock(),
                    mock_response_tools,
                    mock_response_call,
                ]
            )

            mcp = MCPClient(server_url="http://localhost:8000")

            # Call tool
            result = mcp.call_tool("echo", {"message": "Hello"})
            assert result == "Hello, World!"

            mcp.close()

    def test_get_callable_tools_http(self):
        """Test getting callable tools via HTTP transport."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            mcp = MCPClient(server_url="http://localhost:8000")

            tools = mcp.get_callable_tools()
            assert len(tools) == 1
            assert callable(tools[0])
            assert tools[0].__name__ == "test_tool"

            mcp.close()


@pytest.mark.integration
class TestHTTPFromConfig:
    """Test creating HTTP MCPClient from config dict."""

    def test_from_config_http(self):
        """Test creating HTTP client from config."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            config = {
                "type": "mcp",
                "name": "test-server",
                "server_url": "http://localhost:8000",
                "headers": {"Authorization": "Bearer token"},
                "timeout": 60.0,
            }

            mcp = MCPClient.from_config(config)

            assert mcp.server_url == "http://localhost:8000"
            assert mcp.headers == {"Authorization": "Bearer token"}
            assert mcp.timeout == 60.0
            assert mcp.name == "test-server"

            mcp.close()

    def test_get_tools_from_config_http(self):
        """Test getting tools from HTTP config."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "tool1", "description": "Tool 1", "inputSchema": {}},
                        {"name": "tool2", "description": "Tool 2", "inputSchema": {}},
                    ]
                },
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            config = {
                "type": "mcp",
                "name": "test",
                "server_url": "http://localhost:8000",
                "allowed_tools": ["tool1"],
            }

            tools = MCPClient.get_tools_from_config(config)

            # Only tool1 should be returned due to allowed_tools filter
            assert len(tools) == 1
            assert tools[0].__name__ == "tool1"


@pytest.mark.integration
class TestHTTPErrorHandling:
    """Test error handling for HTTP transport."""

    def test_http_connection_error(self):
        """Test handling of HTTP connection errors."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock connection error
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(RuntimeError, match="HTTP request to MCP server failed"):
                MCPClient(server_url="http://localhost:8000")

    def test_http_json_rpc_error(self):
        """Test handling of JSON-RPC errors from server."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock JSON-RPC error response
            mock_response = MagicMock()
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
            mock_response.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(return_value=mock_response)

            with pytest.raises(RuntimeError, match="MCP server error: Invalid Request"):
                MCPClient(server_url="http://localhost:8000")

    def test_http_status_error(self):
        """Test handling of HTTP status errors."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock HTTP status error
            mock_client_instance.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "404 Not Found",
                    request=MagicMock(),
                    response=MagicMock(),
                )
            )

            with pytest.raises(RuntimeError, match="HTTP request to MCP server failed"):
                MCPClient(server_url="http://localhost:8000")


@pytest.mark.integration
class TestHTTPEndpointHandling:
    """Test that server URLs are used exactly as provided."""

    def test_endpoint_uses_exact_url(self):
        """Test that the exact server URL is used without modification."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            # Use full endpoint URL
            mcp = MCPClient(server_url="http://localhost:8000/mcp/v1")

            # Verify that post was called with exact URL (no modification)
            # Calls: initialize, initialized notification, tools/list
            calls = mock_client_instance.post.call_args_list
            assert len(calls) == 3
            assert calls[0][0][0] == "http://localhost:8000/mcp/v1"  # initialize
            assert (
                calls[1][0][0] == "http://localhost:8000/mcp/v1"
            )  # initialized notification
            assert calls[2][0][0] == "http://localhost:8000/mcp/v1"  # tools/list

            mcp.close()

    def test_endpoint_trailing_slash_handled(self):
        """Test that trailing slashes in server URL are removed."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock responses
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[mock_response_init, MagicMock(), mock_response_tools]
            )

            # URL with trailing slash
            mcp = MCPClient(server_url="http://localhost:8000/mcp/v1/")

            # Verify trailing slash is removed
            calls = mock_client_instance.post.call_args_list
            assert calls[0][0][0] == "http://localhost:8000/mcp/v1"

            mcp.close()


@pytest.mark.integration
class TestHTTPSSEResponses:
    """Test SSE (Server-Sent Events) response handling."""

    def test_sse_response_parsing(self):
        """Test handling SSE stream responses."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock initialize (JSON response)
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            # Mock tools/list (JSON response)
            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            # Mock tool call (SSE response)
            mock_response_sse = MagicMock()
            mock_response_sse.headers = {"content-type": "text/event-stream"}
            mock_response_sse.raise_for_status = MagicMock()

            # Simulate SSE stream with data lines
            # MCP format: result has "content" array with text items
            async def mock_aiter_lines():
                lines = [
                    'data: {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "SSE result"}]}}',
                    "",
                ]
                for line in lines:
                    yield line

            mock_response_sse.aiter_lines = mock_aiter_lines

            mock_client_instance.post = AsyncMock(
                side_effect=[
                    mock_response_init,
                    MagicMock(),  # initialized notification (no response checked)
                    mock_response_tools,
                    mock_response_sse,
                ]
            )

            mcp = MCPClient(server_url="http://localhost:8000")

            # Call tool which returns SSE response
            result = mcp.call_tool("test_tool", {})

            # Verify SSE response was parsed correctly
            # call_tool extracts the text from content array
            assert result == "SSE result"

            mcp.close()

    def test_session_id_management(self):
        """Test Mcp-Session-Id header handling."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock initialize with session ID in response
            mock_response_init = MagicMock()
            mock_response_init.headers = {
                "content-type": "application/json",
                "Mcp-Session-Id": "test-session-123",
            }
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            # Mock tools/list
            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            mock_client_instance.post = AsyncMock(
                side_effect=[
                    mock_response_init,
                    MagicMock(),  # initialized notification
                    mock_response_tools,
                ]
            )

            mcp = MCPClient(server_url="http://localhost:8000")

            # Verify session ID was captured
            assert mcp._session_id == "test-session-123"

            # Verify subsequent requests include session ID
            calls = mock_client_instance.post.call_args_list
            # tools/list request (3rd call, index 2) should have session ID
            tools_call = calls[2]
            headers = tools_call[1]["headers"]
            assert "Mcp-Session-Id" in headers
            assert headers["Mcp-Session-Id"] == "test-session-123"

            mcp.close()

    def test_sse_with_multiple_events(self):
        """Test SSE stream with multiple events before final response."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock initialize and tools/list
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": []},
            }
            mock_response_tools.raise_for_status = MagicMock()

            # Mock SSE response with notifications before result
            mock_response_sse = MagicMock()
            mock_response_sse.headers = {"content-type": "text/event-stream"}
            mock_response_sse.raise_for_status = MagicMock()

            async def mock_aiter_lines():
                lines = [
                    'data: {"jsonrpc": "2.0", "method": "notification", "params": {"status": "processing"}}',
                    "",
                    'data: {"jsonrpc": "2.0", "method": "notification", "params": {"status": "almost done"}}',
                    "",
                    'data: {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "final result"}]}}',
                    "",
                ]
                for line in lines:
                    yield line

            mock_response_sse.aiter_lines = mock_aiter_lines

            mock_client_instance.post = AsyncMock(
                side_effect=[
                    mock_response_init,
                    MagicMock(),  # initialized notification
                    mock_response_tools,
                    mock_response_sse,
                ]
            )

            mcp = MCPClient(server_url="http://localhost:8000")
            result = mcp.call_tool("test_tool", {})

            # Should return the final result, ignoring notifications
            assert result == "final result"

            mcp.close()

    def test_mixed_json_and_sse_responses(self):
        """Test that client handles both JSON and SSE responses from same server."""
        with patch("aisuite.mcp.client.httpx.AsyncClient") as mock_async_client:
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance

            # Mock initialize (JSON)
            mock_response_init = MagicMock()
            mock_response_init.headers = {"content-type": "application/json"}
            mock_response_init.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {},
            }
            mock_response_init.raise_for_status = MagicMock()

            # Mock tools/list (JSON)
            mock_response_tools = MagicMock()
            mock_response_tools.headers = {"content-type": "application/json"}
            mock_response_tools.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {"name": "fast_tool", "description": "Fast", "inputSchema": {}},
                        {"name": "slow_tool", "description": "Slow", "inputSchema": {}},
                    ]
                },
            }
            mock_response_tools.raise_for_status = MagicMock()

            # Mock fast tool call (JSON response)
            mock_response_fast = MagicMock()
            mock_response_fast.headers = {"content-type": "application/json"}
            mock_response_fast.json.return_value = {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"content": [{"type": "text", "text": "fast"}]},
            }
            mock_response_fast.raise_for_status = MagicMock()

            # Mock slow tool call (SSE response)
            mock_response_slow = MagicMock()
            mock_response_slow.headers = {"content-type": "text/event-stream"}
            mock_response_slow.raise_for_status = MagicMock()

            async def mock_aiter_lines():
                lines = [
                    'data: {"jsonrpc": "2.0", "id": 4, "result": {"content": [{"type": "text", "text": "slow"}]}}',
                    "",
                ]
                for line in lines:
                    yield line

            mock_response_slow.aiter_lines = mock_aiter_lines

            mock_client_instance.post = AsyncMock(
                side_effect=[
                    mock_response_init,
                    MagicMock(),  # initialized notification
                    mock_response_tools,
                    mock_response_fast,
                    mock_response_slow,
                ]
            )

            mcp = MCPClient(server_url="http://localhost:8000")

            # Call fast tool (JSON response)
            result1 = mcp.call_tool("fast_tool", {})
            assert result1 == "fast"

            # Call slow tool (SSE response)
            result2 = mcp.call_tool("slow_tool", {})
            assert result2 == "slow"

            mcp.close()
