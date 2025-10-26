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
except ImportError:
    raise ImportError(
        "MCP support requires the 'mcp' package. "
        "Install it with: pip install 'aisuite[mcp]' or pip install mcp"
    )

from .tool_wrapper import create_mcp_tool_wrapper


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
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the MCP client and connect to an MCP server.

        Args:
            command: Command to start the MCP server (e.g., "npx", "python")
            args: Arguments to pass to the command (e.g., ["-y", "server-package"])
            env: Optional environment variables for the server process

        Raises:
            ImportError: If the mcp package is not installed
            RuntimeError: If connection to the MCP server fails
        """
        self.server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize connection
        self._connect()

    def _connect(self):
        """
        Establish connection to the MCP server.

        This method:
        1. Creates an event loop if needed
        2. Starts the MCP server process
        3. Performs the MCP initialization handshake
        4. Caches the available tools
        """
        # Get or create event loop
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

        # Run the async connection
        self._event_loop.run_until_complete(self._async_connect())

    async def _async_connect(self):
        """Async connection initialization."""
        # Start the MCP server
        self._read, self._write = await stdio_client(self.server_params).__aenter__()

        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

        # Initialize connection
        await self._session.initialize()

        # List available tools and cache them
        tools_result = await self._session.list_tools()
        self._tools_cache = tools_result.tools if hasattr(tools_result, 'tools') else []

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

    def get_callable_tools(self) -> List[Callable]:
        """
        Get all MCP tools as Python callables compatible with aisuite.

        This is the primary method for using MCP tools with aisuite. It returns
        a list of callable wrappers that can be passed directly to the `tools`
        parameter of `client.chat.completions.create()`.

        Returns:
            List of callable tool wrappers

        Example:
            >>> mcp_tools = mcp.get_callable_tools()
            >>> response = client.chat.completions.create(
            ...     model="openai:gpt-4o",
            ...     messages=messages,
            ...     tools=mcp_tools,
            ...     max_turns=2
            ... )
        """
        tools = self.list_tools()
        return [
            create_mcp_tool_wrapper(self, tool["name"], tool)
            for tool in tools
        ]

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

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            The result from the MCP tool execution

        Raises:
            RuntimeError: If not connected or tool call fails
        """
        if self._session is None:
            raise RuntimeError("Not connected to MCP server")

        # Run the async tool call
        result = self._event_loop.run_until_complete(
            self._async_call_tool(tool_name, arguments)
        )
        return result

    async def _async_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Async implementation of tool calling.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        result = await self._session.call_tool(tool_name, arguments)

        # Extract content from MCP result
        # MCP returns results in various formats, we try to extract the most useful content
        if hasattr(result, 'content'):
            if isinstance(result.content, list) and len(result.content) > 0:
                # Get first content item
                content_item = result.content[0]
                if hasattr(content_item, 'text'):
                    return content_item.text
                elif hasattr(content_item, 'data'):
                    return content_item.data
                return str(content_item)
            return result.content

        # If no content attribute, return the whole result
        return str(result)

    def close(self):
        """
        Close the connection to the MCP server.

        It's recommended to use the MCPClient as a context manager to ensure
        proper cleanup, but this method can be called manually if needed.

        Example:
            >>> mcp = MCPClient(command="npx", args=["server"])
            >>> try:
            ...     # Use mcp
            ...     pass
            ... finally:
            ...     mcp.close()
        """
        if self._session is not None:
            self._event_loop.run_until_complete(self._async_close())

    async def _async_close(self):
        """Async cleanup."""
        if self._session:
            await self._session.__aexit__(None, None, None)
        if self._read and self._write:
            # The stdio_client context manager handles cleanup
            pass

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
        return f"MCPClient(command={self.server_params.command!r}, tools={num_tools})"
