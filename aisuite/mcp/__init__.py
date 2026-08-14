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

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .client import MCPClient

__all__ = ["MCPClient"]


def __getattr__(name: str):
    """Import MCPClient on first access (PEP 562).

    `aisuite.mcp.client` needs the optional `mcp` package, while sibling modules
    such as `aisuite.mcp.config` are pure typing helpers with no third-party
    dependencies. Importing MCPClient eagerly here made the whole subpackage
    unimportable without the extra, so `from .mcp.config import is_mcp_config`
    failed too and left callers with a NameError instead of the intended
    "install the mcp extra" ImportError (#369).
    """
    if name == "MCPClient":
        from .client import MCPClient

        return MCPClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
