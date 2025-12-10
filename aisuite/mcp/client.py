"""
MCP Client for aisuite.

This module provides the MCPClient class that connects to MCP servers and
exposes their tools as Python callables compatible with aisuite's tool system.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import httpx
except ImportError as e:
    if "mcp" in str(e):
        raise ImportError(
            "MCP support requires the 'mcp' package. "
            "Install it with: pip install 'aisuite[mcp]' or pip install mcp"
        )
    elif "httpx" in str(e):
        raise ImportError(
            "HTTP transport requires the 'httpx' package. "
            "Install it with: pip install httpx"
        )
    raise

from .tool_wrapper import create_mcp_tool_wrapper
from .config import MCPConfig, validate_mcp_config, get_transport_type


class MCPClient:
    """
    Client for connecting to MCP servers and using their tools with aisuite.

    This class manages the connection to an MCP server, discovers available tools,
    and creates Python callable wrappers that work seamlessly with aisuite's
    existing tool calling infrastructure.

    Example:
        >>> # Connect to an MCP server
        >>> mcp = MCPClient(
        ...     command="npx",
        ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/path"]
        ... )
        >>>
        >>> # Get tools and use with aisuite
        >>> import aisuite as ai
        >>> client = ai.Client()
        >>> response = client.chat.completions.create(
        ...     model="openai:gpt-4o",
        ...     messages=[{"role": "user", "content": "List files"}],
        ...     tools=mcp.get_callable_tools(),
        ...     max_turns=2
        ... )

    The MCPClient handles:
    - Starting and managing the MCP server process
    - Performing the MCP handshake
    - Discovering available tools
    - Creating callable wrappers for tools
    - Executing tool calls via the MCP protocol
    """

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        server_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        name: Optional[str] = None,
    ):
        """
        Initialize the MCP client and connect to an MCP server.

        Supports both stdio and HTTP transports. Provide either stdio parameters
        (command) OR HTTP parameters (server_url), but not both.

        Args:
            command: Command to start the MCP server (e.g., "npx", "python") - for stdio transport
            args: Arguments to pass to the command (e.g., ["-y", "server-package"]) - for stdio transport
            env: Optional environment variables for the server process - for stdio transport
            server_url: Base URL of the MCP server (e.g., "http://localhost:8000") - for HTTP transport
            headers: Optional HTTP headers (e.g., for authentication) - for HTTP transport
            timeout: Request timeout in seconds - for HTTP transport (default: 30.0)
            name: Optional name for this MCP client (used for logging and prefixing)

        Raises:
            ImportError: If the mcp or httpx package is not installed
            ValueError: If both stdio and HTTP parameters are provided, or neither
            RuntimeError: If connection to the MCP server fails
        """
        # Validate transport parameters
        has_stdio = command is not None
        has_http = server_url is not None

        if not (has_stdio ^ has_http):
            raise ValueError(
                "Must provide exactly one transport: either 'command' (stdio) or 'server_url' (HTTP)."
            )

        # Store parameters based on transport type
        if has_stdio:
            self.server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env,
            )
            self.name = name or command
            # Stdio-specific state
            self._session: Optional[ClientSession] = None
            self._read = None
            self._write = None
            self._stdio_context = None
        else:  # HTTP
            self.server_url = server_url
            self.headers = headers or {}
            self.timeout = timeout
            self.name = name or server_url
            # HTTP-specific state (initialized in _async_connect_http)
            self._http_client = None
            self._request_id = 0
            self._session_id: Optional[str] = None  # MCP session ID from server

        # Shared state
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize connection
        self._connect()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MCPClient":
        """
        Create an MCPClient from a configuration dictionary.

        This method validates the config and creates an MCPClient instance.
        It supports both stdio and HTTP transports.

        Args:
            config: MCP configuration dictionary

        Returns:
            MCPClient instance

        Raises:
            ValueError: If configuration is invalid

        Example (stdio):
            >>> config = {
            ...     "type": "mcp",
            ...     "name": "filesystem",
            ...     "command": "npx",
            ...     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/docs"]
            ... }
            >>> mcp = MCPClient.from_config(config)

        Example (HTTP):
            >>> config = {
            ...     "type": "mcp",
            ...     "name": "api-server",
            ...     "server_url": "http://localhost:8000",
            ...     "headers": {"Authorization": "Bearer token"}
            ... }
            >>> mcp = MCPClient.from_config(config)
        """
        # Validate and normalize config
        validated_config = validate_mcp_config(config)

        # Determine transport type
        transport = get_transport_type(validated_config)

        if transport == "stdio":
            return cls(
                command=validated_config["command"],
                args=validated_config.get("args", []),
                env=validated_config.get("env"),
                name=validated_config["name"],
            )
        else:  # http
            return cls(
                server_url=validated_config["server_url"],
                headers=validated_config.get("headers"),
                timeout=validated_config.get("timeout", 30.0),
                name=validated_config["name"],
            )

    @staticmethod
    def get_tools_from_config(config: Dict[str, Any]) -> List[Callable]:
        """
        Convenience method to create MCPClient and get callable tools from config.

        This is a helper that combines from_config() and get_callable_tools()
        in a single call. It respects the config's allowed_tools and use_tool_prefix
        settings.

        Args:
            config: MCP configuration dictionary

        Returns:
            List of callable tool wrappers

        Example:
            >>> config = {
            ...     "type": "mcp",
            ...     "name": "filesystem",
            ...     "command": "npx",
            ...     "args": ["..."],
            ...     "allowed_tools": ["read_file"],
            ...     "use_tool_prefix": True
            ... }
            >>> tools = MCPClient.get_tools_from_config(config)
            >>> # Returns callable tools filtered and prefixed per config
        """
        # Validate config first
        validated_config = validate_mcp_config(config)

        # Create client
        client = MCPClient.from_config(validated_config)

        # Get tools with config settings
        tools = client.get_callable_tools(
            allowed_tools=validated_config.get("allowed_tools"),
            use_tool_prefix=validated_config.get("use_tool_prefix", False),
        )

        return tools

    def _connect(self):
        """
        Establish connection to the MCP server.

        This method:
        1. Creates an event loop if needed
        2. Detects transport type (stdio or HTTP)
        3. Establishes connection via appropriate transport
        4. Performs the MCP initialization handshake
        5. Caches the available tools

        Note: Automatically handles Jupyter/IPython environments where an event loop
        is already running by using nest_asyncio.
        """
        # Get or create event loop
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

        # Enable nested event loops for Jupyter/IPython compatibility
        # This allows run_until_complete() to work in environments where
        # an event loop is already running (like Jupyter notebooks)
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            # nest_asyncio not available - will work fine in regular Python
            # but may fail in Jupyter. User should install: pip install nest-asyncio
            pass

        # Detect transport type and run appropriate async connection
        if hasattr(self, "server_url"):
            # HTTP transport
            self._event_loop.run_until_complete(self._async_connect_http())
        else:
            # Stdio transport
            self._event_loop.run_until_complete(self._async_connect())

    async def _async_connect(self):
        """Async connection initialization for stdio transport."""
        # Start the MCP server and store the context manager
        self._stdio_context = stdio_client(self.server_params)
        self._read, self._write = await self._stdio_context.__aenter__()

        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

        # Initialize connection
        await self._session.initialize()

        # List available tools and cache them
        tools_result = await self._session.list_tools()

        # Convert Tool objects to dicts for easier handling
        if hasattr(tools_result, "tools"):
            self._tools_cache = [
                {
                    "name": tool.name,
                    "description": (
                        tool.description if hasattr(tool, "description") else ""
                    ),
                    "inputSchema": (
                        tool.inputSchema if hasattr(tool, "inputSchema") else {}
                    ),
                }
                for tool in tools_result.tools
            ]
        else:
            self._tools_cache = []

    async def _parse_sse_response(
        self, response: httpx.Response, request_id: int
    ) -> Dict[str, Any]:
        """
        Parse SSE stream and extract JSON-RPC response.

        SSE format per spec:
            data: {"jsonrpc": "2.0", "id": 1, "result": {...}}

            data: {"jsonrpc": "2.0", "method": "notification", ...}

        The server may send multiple events (notifications, requests) before
        sending the final response. We collect events until we find the
        response matching our request_id.

        Args:
            response: HTTP response with text/event-stream content type
            request_id: The JSON-RPC request ID to match

        Returns:
            Response result dictionary

        Raises:
            RuntimeError: If server returns an error or no matching response found
        """
        result = None

        async for line in response.aiter_lines():
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(":"):
                continue

            # Parse SSE data field
            if line.startswith("data: "):
                data = line[6:]  # Remove 'data: ' prefix

                try:
                    message = json.loads(data)

                    # Check if this is the response to our request
                    if message.get("id") == request_id:
                        if "error" in message:
                            error = message["error"]
                            raise RuntimeError(
                                f"MCP server error: {error.get('message', 'Unknown error')} "
                                f"(code: {error.get('code', 'unknown')})"
                            )
                        result = message.get("result", {})
                        # Found our response, can stop parsing
                        break

                    # Note: Server may send other notifications/requests
                    # which we ignore for now (future enhancement for bidirectional comms)

                except json.JSONDecodeError:
                    # Invalid JSON in SSE data, skip this event
                    continue

        if result is None:
            raise RuntimeError(
                f"No response received in SSE stream for request {request_id}"
            )

        return result

    async def _send_http_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send JSON-RPC request to MCP server via HTTP.

        Args:
            method: JSON-RPC method name
            params: Optional parameters

        Returns:
            Response result

        Raises:
            RuntimeError: If HTTP request fails or server returns an error
        """
        # Increment request ID
        self._request_id += 1

        # Build JSON-RPC 2.0 request
        request_data = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }

        if params:
            request_data["params"] = params

        # Use the exact server URL provided by the user
        url = self.server_url.rstrip("/")

        # Build headers: MCP requires Accept header with both content types
        # Merge with any user-provided headers and session ID
        request_headers = {
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            request_headers["Mcp-Session-Id"] = self._session_id
        if self.headers:
            request_headers.update(self.headers)

        try:
            response = await self._http_client.post(
                url, json=request_data, headers=request_headers
            )
            response.raise_for_status()

            # Check for MCP session ID in response headers
            if "Mcp-Session-Id" in response.headers and not self._session_id:
                self._session_id = response.headers["Mcp-Session-Id"]

            # Check Content-Type to determine response format
            content_type = response.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                # Handle JSON response (simple request-response)
                result = response.json()

                # Check for JSON-RPC error
                if "error" in result:
                    error = result["error"]
                    raise RuntimeError(
                        f"MCP server error: {error.get('message', 'Unknown error')} "
                        f"(code: {error.get('code', 'unknown')})"
                    )

                return result.get("result", {})

            elif "text/event-stream" in content_type:
                # Handle SSE stream response
                return await self._parse_sse_response(response, request_data["id"])

            else:
                raise RuntimeError(
                    f"Unexpected Content-Type from MCP server: {content_type}"
                )

        except httpx.HTTPError as e:
            raise RuntimeError(
                f"HTTP request to MCP server failed: {type(e).__name__}: {str(e)}"
            )

    async def _send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ):
        """
        Send a JSON-RPC notification (no response expected).

        Notifications are JSON-RPC messages without an ID field.
        Per the spec, the server should not send a response.

        Args:
            method: JSON-RPC method name
            params: Optional parameters
        """
        # Build JSON-RPC notification (no id field)
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if params:
            notification["params"] = params

        # Build headers
        url = self.server_url.rstrip("/")
        request_headers = {
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            request_headers["Mcp-Session-Id"] = self._session_id
        if self.headers:
            request_headers.update(self.headers)

        try:
            # Send notification - don't wait for/expect a response
            await self._http_client.post(
                url, json=notification, headers=request_headers
            )
            # Note: We don't check response for notifications
        except httpx.HTTPError:
            # Notifications may timeout or fail, which is acceptable
            pass

    async def _async_connect_http(self):
        """Async connection initialization for HTTP transport."""
        # Create HTTP client
        self._http_client = httpx.AsyncClient(timeout=self.timeout)

        # Send initialize request
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
            "clientInfo": {"name": "aisuite-mcp-client", "version": "1.0.0"},
        }

        await self._send_http_request("initialize", init_params)

        # Send initialized notification (required by MCP spec)
        await self._send_notification("notifications/initialized")

        # List available tools
        tools_result = await self._send_http_request("tools/list")

        # Cache tools
        self._tools_cache = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
            for tool in tools_result.get("tools", [])
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools from the MCP server.

        Returns:
            List of tool schemas in MCP format

        Example:
            >>> tools = mcp.list_tools()
            >>> for tool in tools:
            ...     print(tool['name'], '-', tool['description'])
        """
        if self._tools_cache is None:
            raise RuntimeError("Not connected to MCP server")
        return self._tools_cache

    def get_callable_tools(
        self,
        allowed_tools: Optional[List[str]] = None,
        use_tool_prefix: bool = False,
    ) -> List[Callable]:
        """
        Get all MCP tools as Python callables compatible with aisuite.

        This is the primary method for using MCP tools with aisuite. It returns
        a list of callable wrappers that can be passed directly to the `tools`
        parameter of `client.chat.completions.create()`.

        Args:
            allowed_tools: Optional list of tool names to include. If None, all tools are included.
            use_tool_prefix: If True, prefix tool names with "{client_name}__"

        Returns:
            List of callable tool wrappers

        Example:
            >>> # Get all tools
            >>> mcp_tools = mcp.get_callable_tools()
            >>>
            >>> # Get specific tools only
            >>> mcp_tools = mcp.get_callable_tools(allowed_tools=["read_file"])
            >>>
            >>> # Get tools with name prefixing
            >>> mcp_tools = mcp.get_callable_tools(use_tool_prefix=True)
            >>> # Tools will be named "filesystem__read_file", etc.
        """
        all_tools = self.list_tools()

        # Filter tools if allowed_tools is specified
        if allowed_tools is not None:
            all_tools = [t for t in all_tools if t["name"] in allowed_tools]

        # Create wrappers
        wrappers = []
        for tool in all_tools:
            wrapper = create_mcp_tool_wrapper(self, tool["name"], tool)

            # Apply prefix if requested
            if use_tool_prefix:
                original_name = wrapper.__name__
                wrapper.__name__ = f"{self.name}__{original_name}"

            wrappers.append(wrapper)

        return wrappers

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a specific MCP tool by name as a Python callable.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Callable wrapper for the tool, or None if not found

        Example:
            >>> read_file = mcp.get_tool("read_file")
            >>> write_file = mcp.get_tool("write_file")
            >>> tools = [read_file, write_file]
        """
        tools = self.list_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return create_mcp_tool_wrapper(self, tool_name, tool)
        return None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute an MCP tool call.

        This method is called by MCPToolWrapper when the LLM requests a tool.
        It handles the async MCP protocol communication and returns the result.
        Automatically routes to the appropriate transport (stdio or HTTP).

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            The result from the MCP tool execution

        Raises:
            RuntimeError: If not connected or tool call fails
        """
        # Detect transport type and route to appropriate method
        if hasattr(self, "_http_client") and self._http_client is not None:
            # HTTP transport
            if self._http_client is None:
                raise RuntimeError("Not connected to MCP server (HTTP)")
            result = self._event_loop.run_until_complete(
                self._async_call_tool_http(tool_name, arguments)
            )
        else:
            # Stdio transport
            if self._session is None:
                raise RuntimeError("Not connected to MCP server (stdio)")
            result = self._event_loop.run_until_complete(
                self._async_call_tool(tool_name, arguments)
            )
        return result

    async def _async_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Async implementation of tool calling for stdio transport.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        result = await self._session.call_tool(tool_name, arguments)

        # Extract content from MCP result
        # MCP returns results in various formats, we try to extract the most useful content
        if hasattr(result, "content"):
            if isinstance(result.content, list) and len(result.content) > 0:
                # Get first content item
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return content_item.text
                elif hasattr(content_item, "data"):
                    return content_item.data
                return str(content_item)
            return result.content

        # If no content attribute, return the whole result
        return str(result)

    async def _async_call_tool_http(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """
        Async implementation of tool calling for HTTP transport.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        params = {"name": tool_name, "arguments": arguments}

        result = await self._send_http_request("tools/call", params)

        # Extract content from MCP result (HTTP format)
        # Similar to stdio, but result is already a dict
        if "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                # Get first content item
                content_item = content[0]
                if isinstance(content_item, dict):
                    if "text" in content_item:
                        return content_item["text"]
                    elif "data" in content_item:
                        return content_item["data"]
                return str(content_item)
            return content

        # If no content field, return the whole result
        return json.dumps(result)

    def close(self):
        """
        Close the connection to the MCP server.

        Works for both stdio and HTTP transports. It's recommended to use
        the MCPClient as a context manager to ensure proper cleanup, but
        this method can be called manually if needed.

        Example:
            >>> mcp = MCPClient(command="npx", args=["server"])
            >>> try:
            ...     # Use mcp
            ...     pass
            ... finally:
            ...     mcp.close()
        """
        # Check if we need to cleanup (either stdio or HTTP)
        needs_cleanup = (hasattr(self, "_session") and self._session is not None) or (
            hasattr(self, "_http_client") and self._http_client is not None
        )

        if needs_cleanup:
            self._event_loop.run_until_complete(self._async_close())

    async def _async_close(self):
        """Async cleanup for both stdio and HTTP transports."""
        # Cleanup stdio transport
        try:
            if hasattr(self, "_session") and self._session:
                await self._session.__aexit__(None, None, None)
        except RuntimeError as e:
            # Suppress anyio cancel scope errors that occur in Jupyter/nest_asyncio environments
            # This is a known incompatibility between nest_asyncio and anyio task groups
            if "cancel scope" not in str(e).lower():
                raise
        except Exception:
            pass  # Ignore other errors during session cleanup

        try:
            if hasattr(self, "_stdio_context") and self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except RuntimeError as e:
            # Suppress anyio cancel scope errors that occur in Jupyter/nest_asyncio environments
            # This is a known incompatibility between nest_asyncio and anyio task groups
            if "cancel scope" not in str(e).lower():
                raise
        except Exception:
            pass  # Ignore other errors during stdio cleanup

        # Cleanup HTTP transport
        try:
            if hasattr(self, "_http_client") and self._http_client:
                await self._http_client.aclose()
        except Exception:
            pass  # Ignore errors during HTTP client cleanup

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        num_tools = len(self._tools_cache) if self._tools_cache else 0
        if hasattr(self, "server_url"):
            return f"MCPClient(server_url={self.server_url!r}, tools={num_tools})"
        else:
            return (
                f"MCPClient(command={self.server_params.command!r}, tools={num_tools})"
            )
