"""
MCP configuration validation and normalization.

This module provides utilities for validating and normalizing MCP tool
configuration dictionaries passed to aisuite's chat completion API.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class MCPConfig(TypedDict, total=False):
    """Type definition for MCP tool configuration."""

    # Required fields
    type: Literal["mcp"]
    name: str

    # Transport: stdio
    command: str
    args: List[str]
    env: Dict[str, str]
    cwd: str

    # Transport: http
    server_url: str
    headers: Dict[str, str]

    # Tool filtering
    allowed_tools: List[str]

    # Namespacing
    use_tool_prefix: bool

    # Safety limits
    timeout_seconds: int
    response_bytes_cap: int

    # Connection behavior
    lazy_connect: bool


# Default values
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_RESPONSE_BYTES_CAP = 10 * 1024 * 1024  # 10 MB
DEFAULT_USE_TOOL_PREFIX = False
DEFAULT_LAZY_CONNECT = False


def validate_mcp_config(config: Dict[str, Any]) -> MCPConfig:
    """
    Validate and normalize an MCP tool configuration.

    This function:
    1. Validates required fields are present
    2. Auto-detects transport type (stdio vs http)
    3. Validates transport-specific required fields
    4. Sets defaults for optional fields
    5. Returns a normalized config dict

    Args:
        config: Raw MCP configuration dictionary

    Returns:
        Validated and normalized MCP configuration

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> config = {
        ...     "type": "mcp",
        ...     "name": "filesystem",
        ...     "command": "npx",
        ...     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/docs"]
        ... }
        >>> validated = validate_mcp_config(config)
        >>> validated['timeout_seconds']
        30
    """
    # Check type field
    if config.get("type") != "mcp":
        raise ValueError(f"Invalid config type: {config.get('type')}. Expected 'mcp'")

    # Check name field (required)
    if "name" not in config:
        raise ValueError(
            "MCP config must have 'name' field. "
            "Example: {'type': 'mcp', 'name': 'my_server', ...}"
        )

    name = config["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"MCP 'name' must be a non-empty string, got: {name}")

    # Auto-detect transport type
    has_stdio = "command" in config
    has_http = "server_url" in config

    if not (has_stdio ^ has_http):
        raise ValueError(
            "MCP config must have either 'command' or 'server_url'."
            "Use one or the other to specify transport type."
        )

    # Validate stdio transport
    if has_stdio:
        if not isinstance(config["command"], str):
            raise ValueError(
                f"MCP 'command' must be a string, got: {type(config['command'])}"
            )

        # args is optional but should be a list if present
        if "args" in config and not isinstance(config["args"], list):
            raise ValueError(f"MCP 'args' must be a list, got: {type(config['args'])}")

        # env is optional but should be a dict if present
        if "env" in config and not isinstance(config["env"], dict):
            raise ValueError(f"MCP 'env' must be a dict, got: {type(config['env'])}")

    # Validate http transport
    if has_http:
        if not isinstance(config["server_url"], str):
            raise ValueError(
                f"MCP 'server_url' must be a string, got: {type(config['server_url'])}"
            )

        # Validate URL format
        server_url = config["server_url"]
        if not (server_url.startswith("http://") or server_url.startswith("https://")):
            raise ValueError(
                f"MCP 'server_url' must start with http:// or https://, got: {server_url}"
            )

        # headers is optional but should be a dict if present
        if "headers" in config and not isinstance(config["headers"], dict):
            raise ValueError(
                f"MCP 'headers' must be a dict, got: {type(config['headers'])}"
            )

        # timeout is optional but should be a number if present
        if "timeout" in config:
            if not isinstance(config["timeout"], (int, float)):
                raise ValueError(
                    f"MCP 'timeout' must be a number, got: {type(config['timeout'])}"
                )
            if config["timeout"] <= 0:
                raise ValueError(
                    f"MCP 'timeout' must be positive, got: {config['timeout']}"
                )

    # Validate optional fields
    if "allowed_tools" in config:
        if not isinstance(config["allowed_tools"], list):
            raise ValueError(
                f"MCP 'allowed_tools' must be a list, got: {type(config['allowed_tools'])}"
            )
        if not all(isinstance(t, str) for t in config["allowed_tools"]):
            raise ValueError("MCP 'allowed_tools' must be a list of strings")

    if "use_tool_prefix" in config:
        if not isinstance(config["use_tool_prefix"], bool):
            raise ValueError(
                f"MCP 'use_tool_prefix' must be a boolean, got: {type(config['use_tool_prefix'])}"
            )

    if "timeout_seconds" in config:
        if not isinstance(config["timeout_seconds"], (int, float)):
            raise ValueError(
                f"MCP 'timeout_seconds' must be a number, got: {type(config['timeout_seconds'])}"
            )
        if config["timeout_seconds"] <= 0:
            raise ValueError(
                f"MCP 'timeout_seconds' must be positive, got: {config['timeout_seconds']}"
            )

    if "response_bytes_cap" in config:
        if not isinstance(config["response_bytes_cap"], int):
            raise ValueError(
                f"MCP 'response_bytes_cap' must be an integer, got: {type(config['response_bytes_cap'])}"
            )
        if config["response_bytes_cap"] <= 0:
            raise ValueError(
                f"MCP 'response_bytes_cap' must be positive, got: {config['response_bytes_cap']}"
            )

    # Create normalized config with defaults
    normalized: MCPConfig = {
        "type": "mcp",
        "name": config["name"],
    }

    # Copy transport fields
    if has_stdio:
        normalized["command"] = config["command"]
        normalized["args"] = config.get("args", [])
        if "env" in config:
            normalized["env"] = config["env"]
        if "cwd" in config:
            normalized["cwd"] = config["cwd"]
    else:  # has_http
        normalized["server_url"] = config["server_url"]
        if "headers" in config:
            normalized["headers"] = config["headers"]
        if "timeout" in config:
            normalized["timeout"] = config["timeout"]

    # Copy optional fields with defaults
    if "allowed_tools" in config:
        normalized["allowed_tools"] = config["allowed_tools"]

    normalized["use_tool_prefix"] = config.get(
        "use_tool_prefix", DEFAULT_USE_TOOL_PREFIX
    )
    normalized["timeout_seconds"] = config.get(
        "timeout_seconds", DEFAULT_TIMEOUT_SECONDS
    )
    normalized["response_bytes_cap"] = config.get(
        "response_bytes_cap", DEFAULT_RESPONSE_BYTES_CAP
    )
    normalized["lazy_connect"] = config.get("lazy_connect", DEFAULT_LAZY_CONNECT)

    return normalized


def is_mcp_config(obj: Any) -> bool:
    """
    Check if an object is an MCP config dictionary.

    Args:
        obj: Object to check

    Returns:
        True if obj is a dict with type="mcp", False otherwise

    Example:
        >>> is_mcp_config({"type": "mcp", "name": "test"})
        True
        >>> is_mcp_config(lambda: None)
        False
    """
    return isinstance(obj, dict) and obj.get("type") == "mcp"


def get_transport_type(config: MCPConfig) -> Literal["stdio", "http"]:
    """
    Determine the transport type from a validated MCP config.

    Args:
        config: Validated MCP configuration

    Returns:
        "stdio" or "http"
    """
    if "command" in config:
        return "stdio"
    else:
        return "http"
