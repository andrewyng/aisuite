"""
MCP (Model Context Protocol) integration for aisuite.

This module provides support for using MCP servers and their tools with aisuite's
unified interface for AI providers.

MCP allows AI applications to connect to external data sources and tools through
a standardized protocol. This integration makes MCP tools available as Python
callables that work seamlessly with aisuite's existing tool calling infrastructure.

Example:
    >>> from aisuite import Client
    >>> from aisuite.mcp import MCPClient
    >>>
    >>> # Connect to an MCP server
    >>> mcp = MCPClient(
    ...     command="npx",
    ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/docs"]
    ... )
    >>>
    >>> # Use MCP tools with any provider
    >>> client = Client()
    >>> response = client.chat.completions.create(
    ...     model="openai:gpt-4o",
    ...     messages=[{"role": "user", "content": "Read README.md"}],
    ...     tools=mcp.get_callable_tools(),
    ...     max_turns=2
    ... )
"""

from .client import MCPClient

__all__ = ["MCPClient"]
