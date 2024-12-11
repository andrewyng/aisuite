from typing import Callable, Dict, Any, Type, Optional
from pydantic import BaseModel, create_model, Field, ValidationError
import inspect
import json


class ToolManager:
    def __init__(self):
        self._tools = {}

    # Add a tool function with or without a Pydantic model.
    def add_tool(self, func: Callable, param_model: Optional[Type[BaseModel]] = None):
        """Register a tool function with metadata. If no param_model is provided, infer from function signature."""
        if param_model:
            tool_spec = self._convert_to_tool_spec(func, param_model)
        else:
            tool_spec, param_model = self._infer_from_signature(func)

        self._tools[func.__name__] = {
            "function": func,
            "param_model": param_model,
            "spec": tool_spec,
        }

    def tools(self, format="openai") -> list:
        """Return tools in the specified format (default OpenAI)."""
        if format == "openai":
            return self._convert_to_openai_format()
        return [tool["spec"] for tool in self._tools.values()]

    def _convert_to_tool_spec(
        self, func: Callable, param_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Convert the function and its Pydantic model to a unified tool specification."""
        type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}

        properties = {}
        for field_name, field in param_model.model_fields.items():
            field_type = field.annotation

            # Handle enum types
            if hasattr(field_type, "__members__"):  # Check if it's an enum
                enum_values = [
                    member.value if hasattr(member, "value") else member.name
                    for member in field_type
                ]
                properties[field_name] = {
                    "type": "string",
                    "enum": enum_values,
                    "description": field.description or "",
                }
                # Convert enum default value to string if it exists
                if str(field.default) != "PydanticUndefined":
                    properties[field_name]["default"] = (
                        field.default.value
                        if hasattr(field.default, "value")
                        else field.default
                    )
            else:
                properties[field_name] = {
                    "type": type_mapping.get(field_type, str(field_type)),
                    "description": field.description or "",
                }
                # Add default if it exists and isn't PydanticUndefined
                if str(field.default) != "PydanticUndefined":
                    properties[field_name]["default"] = field.default

        return {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [
                    name
                    for name, field in param_model.model_fields.items()
                    if field.is_required and str(field.default) == "PydanticUndefined"
                ],
            },
        }

    def _infer_from_signature(
        self, func: Callable
    ) -> tuple[Dict[str, Any], Type[BaseModel]]:
        """Infer parameters(required and optional) and requirements directly from the function signature."""
        signature = inspect.signature(func)
        fields = {}
        required_fields = []

        # Get function's docstring
        docstring = inspect.getdoc(func) or " "

        for param_name, param in signature.parameters.items():
            # Check if a type annotation is missing
            if param.annotation == inspect._empty:
                raise TypeError(
                    f"Parameter '{param_name}' in function '{func.__name__}' must have a type annotation."
                )

            # Determine field type and optionality
            param_type = param.annotation
            if param.default == inspect._empty:
                fields[param_name] = (param_type, ...)
                required_fields.append(param_name)
            else:
                fields[param_name] = (param_type, Field(default=param.default))

        # Dynamically create a Pydantic model based on inferred fields
        param_model = create_model(f"{func.__name__.capitalize()}Params", **fields)

        # Convert inferred model to a tool spec format
        tool_spec = self._convert_to_tool_spec(func, param_model)

        # Update the tool spec with the docstring
        tool_spec["description"] = docstring

        return tool_spec, param_model

    def _convert_to_openai_format(self) -> list:
        """Convert tools to OpenAI's format."""
        return [
            {"type": "function", "function": tool["spec"]}
            for tool in self._tools.values()
        ]

    def execute_tool(self, tool_calls) -> tuple[list, list]:
        """Executes registered tools based on the tool calls from the model.

        Args:
            tool_calls: List of tool calls from the model

        Returns:
            List of tuples containing (result, result_message) for each tool call
        """
        results = []
        messages = []

        # Handle single tool call or list of tool calls
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if tool_name not in self._tools:
                raise ValueError(f"Tool '{tool_name}' not registered.")

            tool = self._tools[tool_name]
            tool_func = tool["function"]
            param_model = tool["param_model"]

            # Validate and parse the arguments with Pydantic if a model exists
            try:
                validated_args = param_model(**arguments)
                result = tool_func(**validated_args.model_dump())
                results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result),
                        "tool_call_id": tool_call.id,  # Include the tool call ID in the response
                    }
                )
            except ValidationError as e:
                raise ValueError(f"Error in tool '{tool_name}' parameters: {e}")

        return results, messages
