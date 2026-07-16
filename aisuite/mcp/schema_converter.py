"""
Schema conversion utilities for MCP tools.

This module provides functionality to convert MCP JSON Schema tool definitions
to Python type annotations that are compatible with aisuite's existing Tools class.
"""

from typing import Any, Dict, List, Optional, Union, get_args, get_origin
import inspect


def json_schema_to_python_type(schema: Dict[str, Any]) -> type:
    """
    Convert a JSON Schema type definition to a Python type annotation.

    Args:
        schema: JSON Schema type definition (e.g., {"type": "string"})

    Returns:
        Python type annotation (e.g., str, int, List[str], etc.)
    """
    schema_type = schema.get("type")

    # Handle null/None
    if schema_type == "null":
        return type(None)

    # Handle basic types
    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    if schema_type in type_mapping:
        base_type = type_mapping[schema_type]

        # Handle arrays with item type
        if schema_type == "array" and "items" in schema:
            item_type = json_schema_to_python_type(schema["items"])
            return List[item_type]

        return base_type

    # Handle anyOf/oneOf (union types)
    if "anyOf" in schema or "oneOf" in schema:
        union_schemas = schema.get("anyOf", schema.get("oneOf", []))
        types = [json_schema_to_python_type(s) for s in union_schemas]
        if len(types) == 1:
            return types[0]
        return Union[tuple(types)]

    # Default to Any if we can't determine the type
    return Any


def mcp_schema_to_annotations(input_schema: Dict[str, Any]) -> Dict[str, type]:
    """
    Convert MCP tool input schema to Python type annotations.

    MCP tools use JSON Schema for their input parameters. This function
    converts those schemas to Python type annotations that can be used
    by aisuite's Tools class.

    Args:
        input_schema: MCP tool input schema (JSON Schema format)

    Returns:
        Dictionary mapping parameter names to Python types

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "location": {"type": "string"},
        ...         "count": {"type": "integer"}
        ...     },
        ...     "required": ["location"]
        ... }
        >>> annotations = mcp_schema_to_annotations(schema)
        >>> annotations
        {'location': <class 'str'>, 'count': typing.Optional[int]}
    """
    annotations = {}

    if input_schema.get("type") != "object":
        return annotations

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    for param_name, param_schema in properties.items():
        param_type = json_schema_to_python_type(param_schema)

        # Make optional if not in required list
        if param_name not in required:
            param_type = Optional[param_type]

        annotations[param_name] = param_type

    return annotations


def create_function_signature(
    func_name: str, annotations: Dict[str, type], docstring: Optional[str] = None
) -> inspect.Signature:
    """
    Create a function signature from parameter annotations.

    Args:
        func_name: Name of the function
        annotations: Dictionary mapping parameter names to types
        docstring: Optional docstring for the function

    Returns:
        inspect.Signature object
    """
    parameters = []

    for param_name, param_type in annotations.items():
        # Check if it's an Optional type
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if type(None) in args:
                # It's Optional, set default to None
                parameters.append(
                    inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=param_type,
                    )
                )
            else:
                parameters.append(
                    inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=param_type,
                    )
                )
        else:
            # Required parameter
            parameters.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=param_type,
                )
            )

    return inspect.Signature(parameters)


def extract_parameter_descriptions(input_schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract parameter descriptions from MCP schema.

    Args:
        input_schema: MCP tool input schema

    Returns:
        Dictionary mapping parameter names to their descriptions
    """
    descriptions = {}
    properties = input_schema.get("properties", {})

    for param_name, param_schema in properties.items():
        if "description" in param_schema:
            descriptions[param_name] = param_schema["description"]

    return descriptions


def build_docstring(
    tool_description: str, parameter_descriptions: Dict[str, str]
) -> str:
    """
    Build a Python docstring from MCP tool description and parameter descriptions.

    Args:
        tool_description: Overall description of the tool
        parameter_descriptions: Dictionary of parameter descriptions

    Returns:
        Formatted docstring
    """
    lines = [tool_description, ""]

    if parameter_descriptions:
        lines.append("Args:")
        for param_name, param_desc in parameter_descriptions.items():
            lines.append(f"    {param_name}: {param_desc}")

    return "\n".join(lines)
