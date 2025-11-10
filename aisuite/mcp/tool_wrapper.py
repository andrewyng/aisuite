"""
MCP Tool Wrapper for aisuite.

This module provides the MCPToolWrapper class, which creates Python callable
wrappers around MCP tools that are compatible with aisuite's existing tool
calling infrastructure.
"""

from typing import Any, Callable, Dict, Optional
import asyncio
import inspect
from .schema_converter import (
    mcp_schema_to_annotations,
    extract_parameter_descriptions,
    build_docstring,
)


class MCPToolWrapper:
    """
    A callable wrapper around an MCP tool that makes it compatible with aisuite.

    This class wraps an MCP tool and exposes it as a Python callable with proper
    type annotations and docstrings that aisuite's Tools class can inspect and use.

    The wrapper sets the following attributes that aisuite's Tools class reads:
    - __name__: The tool name
    - __doc__: The tool description and parameter documentation
    - __annotations__: Python type annotations for parameters

    When called, the wrapper executes the MCP tool via the MCP protocol.

    Example:
        >>> wrapper = MCPToolWrapper(mcp_client, "read_file", tool_schema)
        >>> result = wrapper(path="/path/to/file")
    """

    def __init__(
        self,
        mcp_client: "MCPClient",  # Forward reference to avoid circular import
        tool_name: str,
        tool_schema: Dict[str, Any],
    ):
        """
        Initialize the MCP tool wrapper.

        Args:
            mcp_client: The MCPClient instance that manages the connection
            tool_name: Name of the MCP tool
            tool_schema: MCP tool schema definition
        """
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.schema = tool_schema

        # Set attributes that aisuite's Tools class will inspect
        self.__name__ = tool_name

        # Build docstring from MCP schema
        description = tool_schema.get("description", "")
        input_schema = tool_schema.get("inputSchema", {})
        param_descriptions = extract_parameter_descriptions(input_schema)
        self.__doc__ = build_docstring(description, param_descriptions)

        # Convert MCP JSON Schema to Python type annotations
        self.__annotations__ = mcp_schema_to_annotations(input_schema)

        # Create a proper signature for inspect.signature() to read
        # This allows aisuite's Tools class to introspect the parameters
        self.__signature__ = self._create_signature(input_schema)

        # Store the original MCP inputSchema for direct use by Tools class
        # This avoids lossy round-trip conversion through Python type annotations
        # and preserves all JSON Schema details (arrays, nested objects, etc.)
        self.__mcp_input_schema__ = input_schema

    def _create_signature(self, input_schema: Dict[str, Any]) -> inspect.Signature:
        """
        Create a signature for this wrapper based on MCP tool schema.

        This allows inspect.signature() to see the proper parameters with
        type annotations, rather than just **kwargs.
        """
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        parameters = []
        for param_name, annotation in self.__annotations__.items():
            # Create parameter with annotation and default
            if param_name in required:
                # Required parameter (no default)
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                )
            else:
                # Optional parameter (with None default)
                param = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=annotation,
                )
            parameters.append(param)

        return inspect.Signature(parameters, return_annotation=Any)

    def __call__(self, **kwargs) -> Any:
        """
        Execute the MCP tool with the given arguments.

        This method is called by aisuite's tool execution loop when the LLM
        requests this tool.

        Args:
            **kwargs: Tool arguments as keyword arguments

        Returns:
            The result from the MCP tool execution
        """
        # Filter out None values - only pass parameters that have actual values
        # This prevents passing null to MCP tools that expect specific types
        # (e.g., a tool expecting number won't accept null, it wants the param omitted)
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Call the MCP client's tool execution method
        # The MCP client handles the async MCP protocol communication
        return self.mcp_client.call_tool(self.tool_name, filtered_kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the wrapper."""
        return f"MCPToolWrapper(name={self.tool_name!r})"


def create_mcp_tool_wrapper(
    mcp_client: "MCPClient",
    tool_name: str,
    tool_schema: Dict[str, Any],
) -> Callable:
    """
    Factory function to create an MCP tool wrapper.

    Args:
        mcp_client: The MCPClient instance
        tool_name: Name of the tool
        tool_schema: MCP tool schema

    Returns:
        Callable wrapper for the MCP tool
    """
    return MCPToolWrapper(mcp_client, tool_name, tool_schema)
