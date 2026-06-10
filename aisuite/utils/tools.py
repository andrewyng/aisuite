from typing import Callable, Dict, Any, Type, Optional, get_origin, get_args, Union
from pydantic import BaseModel, create_model, Field, ValidationError
import inspect
import json
from docstring_parser import parse


def _preview_tool_result(value: Any, max_chars: int = 2000) -> str:
    preview_value = _truncate_preview_value(value)
    try:
        rendered = json.dumps(preview_value, sort_keys=True)
    except TypeError:
        rendered = str(preview_value)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 3] + "..."


def _truncate_preview_value(value: Any, max_string_chars: int = 600) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        return value[: max_string_chars - 3] + "..."
    if isinstance(value, list):
        items = [_truncate_preview_value(item, max_string_chars) for item in value[:20]]
        if len(value) > 20:
            items.append(f"... {len(value) - 20} more")
        return items
    if isinstance(value, tuple):
        return _truncate_preview_value(list(value), max_string_chars)
    if isinstance(value, dict):
        return {
            key: _truncate_preview_value(item, max_string_chars)
            for key, item in value.items()
        }
    return value


class Tools:
    def __init__(self, tools: list[Callable] = None):
        self._tools = {}
        self.last_policy_events = []
        self.last_tool_events = []
        if tools:
            for tool in tools:
                self._add_tool(tool)

    def _active_trace_context(self):
        from ..agents.context import get_active_run_context

        return get_active_run_context()

    def _emit_tool_trace_event(self, event_type: str, data: dict[str, Any]) -> None:
        context = self._active_trace_context()
        if not context or not context.trace_sinks or not context.trace_id:
            return
        from ..tracing.sinks import TraceEvent, emit_event

        emit_event(
            context.trace_sinks,
            TraceEvent(
                event_type=event_type,
                trace_id=context.trace_id,
                agent_name=context.agent_name,
                run_name=context.run_name,
                parent_run_id=context.parent_run_id,
                group_id=context.group_id,
                tags=list(context.tags),
                metadata=dict(context.metadata),
                data=data,
            ),
        )

    def _artifactized_trace_value(self, value: Any) -> tuple[Any, list[dict[str, Any]]]:
        context = self._active_trace_context()
        if not context or context.artifact_store is None:
            return value, []
        from ..agents.artifacts import artifactize_value, collect_artifactized_fields

        artifactized = artifactize_value(value, context.artifact_store)
        return artifactized, collect_artifactized_fields(artifactized)

    # Add a tool function with or without a Pydantic model.
    def _add_tool(self, func: Callable, param_model: Optional[Type[BaseModel]] = None):
        """Register a tool function with metadata. If no param_model is provided, infer from function signature."""
        from ..agents.types import ToolMetadata

        # Check if this is an MCP tool with original schema
        if hasattr(func, "__mcp_input_schema__") and func.__mcp_input_schema__:
            # Use the original MCP schema directly to preserve all JSON Schema details
            tool_spec = self._convert_mcp_schema_to_tool_spec(func)
            # Create Pydantic model from MCP schema for validation
            param_model = self._create_pydantic_model_from_mcp_schema(func)
        elif param_model:
            tool_spec = self._convert_to_tool_spec(func, param_model)
        else:
            tool_spec, param_model = self.__infer_from_signature(func)

        metadata = getattr(func, "__aisuite_tool_metadata__", None)
        if metadata is not None and metadata.name is None:
            metadata.name = func.__name__

        self._tools[func.__name__] = {
            "function": func,
            "param_model": param_model,
            "spec": tool_spec,
            "metadata": metadata,
        }

    # Return tools in the specified format (default OpenAI).
    def tools(self, format="openai") -> list:
        """Return tools in the specified format (default OpenAI)."""
        if format == "openai":
            return self.__convert_to_openai_format()
        return [tool["spec"] for tool in self._tools.values()]

    def _unwrap_optional(self, field_type: Type) -> tuple[Type, bool]:
        """
        Unwrap Optional[T] to get the base type T.

        Returns:
            tuple: (base_type, is_optional)
        """
        # Check if it's Optional (Union with None)
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Optional[T] is Union[T, None]
            if type(None) in args:
                # Get the non-None type
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    return non_none_types[0], True
        return field_type, False

    # Convert the function and its Pydantic model to a unified tool specification.
    def _convert_to_tool_spec(
        self, func: Callable, param_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Convert the function and its Pydantic model to a unified tool specification."""
        type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}

        properties = {}
        for field_name, field in param_model.model_fields.items():
            field_type = field.annotation

            # Unwrap Optional[T] to get base type T
            field_type, is_optional = self._unwrap_optional(field_type)

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

    def __extract_param_descriptions(self, func: Callable) -> dict[str, str]:
        """Extract parameter descriptions from function docstring.

        Args:
            func: The function to extract parameter descriptions from

        Returns:
            Dictionary mapping parameter names to their descriptions
        """
        docstring = inspect.getdoc(func) or ""
        parsed_docstring = parse(docstring)

        param_descriptions = {}
        for param in parsed_docstring.params:
            param_descriptions[param.arg_name] = param.description or ""

        return param_descriptions

    def _convert_mcp_schema_to_tool_spec(self, func: Callable) -> Dict[str, Any]:
        """
        Convert MCP tool with original inputSchema to tool spec.

        This preserves the original JSON Schema from MCP without round-trip conversion,
        avoiding information loss for complex types like arrays and nested objects.

        Args:
            func: MCP tool wrapper with __mcp_input_schema__ attribute

        Returns:
            Tool specification compatible with OpenAI format
        """
        input_schema = func.__mcp_input_schema__

        return {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": input_schema,  # Use original schema directly!
        }

    def _create_pydantic_model_from_mcp_schema(self, func: Callable) -> Type[BaseModel]:
        """
        Create a Pydantic model from MCP inputSchema for parameter validation.

        This is needed for the execute() method to validate tool call arguments.

        Args:
            func: MCP tool wrapper with __mcp_input_schema__ attribute

        Returns:
            Pydantic model for parameter validation
        """
        from ..mcp.schema_converter import mcp_schema_to_annotations

        input_schema = func.__mcp_input_schema__
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Get type annotations from MCP schema
        annotations = mcp_schema_to_annotations(input_schema)

        fields = {}
        for param_name, param_type in annotations.items():
            param_schema = properties.get(param_name, {})
            description = param_schema.get("description", "")

            if param_name in required:
                fields[param_name] = (param_type, Field(..., description=description))
            else:
                fields[param_name] = (
                    param_type,
                    Field(default=None, description=description),
                )

        return create_model(f"{func.__name__.capitalize()}Params", **fields)

    def __infer_from_signature(
        self, func: Callable
    ) -> tuple[Dict[str, Any], Type[BaseModel]]:
        """Infer parameters(required and optional) and requirements directly from the function signature."""
        signature = inspect.signature(func)
        fields = {}
        required_fields = []

        # Get function's docstring and parse parameter descriptions
        param_descriptions = self.__extract_param_descriptions(func)
        docstring = inspect.getdoc(func) or ""

        # Parse the docstring to get the main function description
        parsed_docstring = parse(docstring)
        function_description = parsed_docstring.short_description or ""
        if parsed_docstring.long_description:
            function_description += "\n\n" + parsed_docstring.long_description

        for param_name, param in signature.parameters.items():
            # Check if a type annotation is missing
            if param.annotation == inspect._empty:
                raise TypeError(
                    f"Parameter '{param_name}' in function '{func.__name__}' must have a type annotation."
                )

            # Determine field type and optionality
            param_type = param.annotation
            description = param_descriptions.get(param_name, "")

            if param.default == inspect._empty:
                fields[param_name] = (param_type, Field(..., description=description))
                required_fields.append(param_name)
            else:
                fields[param_name] = (
                    param_type,
                    Field(default=param.default, description=description),
                )

        # Dynamically create a Pydantic model based on inferred fields
        param_model = create_model(f"{func.__name__.capitalize()}Params", **fields)

        # Convert inferred model to a tool spec format
        tool_spec = self._convert_to_tool_spec(func, param_model)

        # Update the tool spec with the parsed function description instead of raw docstring
        tool_spec["description"] = function_description

        return tool_spec, param_model

    def __convert_to_openai_format(self) -> list:
        """Convert tools to OpenAI's format."""
        return [
            {"type": "function", "function": tool["spec"]}
            for tool in self._tools.values()
        ]

    def results_to_messages(self, results: list, message: any) -> list:
        """Converts results to messages."""
        # if message is empty return empty list
        if not message or len(results) == 0:
            return []

        messages = []
        # Iterate over results and match with tool calls from the message
        for result in results:
            # Find matching tool call from message.tool_calls
            for tool_call in message.tool_calls:
                if tool_call.id == result["tool_call_id"]:
                    messages.append(
                        {
                            "role": "tool",
                            "name": result["name"],
                            "content": json.dumps(result["content"]),
                            "tool_call_id": tool_call.id,
                        }
                    )
                    break

        return messages

    def execute(self, tool_calls) -> list:
        """Executes registered tools based on the tool calls from the model.

        Args:
            tool_calls: List of tool calls from the model

        Returns:
            List of results from executing each tool call
        """
        results = []

        # Handle single tool call or list of tool calls
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            # Handle both dictionary and object-style tool calls
            if isinstance(tool_call, dict):
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
            else:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

            # Ensure arguments is a dict
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

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
            except ValidationError as e:
                raise ValueError(f"Error in tool '{tool_name}' parameters: {e}")

        return results

    def _evaluate_tool_policy(
        self,
        tool_policy,
        tool_policy_context,
        tool_name: str,
        arguments: dict,
        tool_metadata=None,
    ):
        if tool_policy is None:
            return None

        from ..agents import ToolPolicyContext, ToolPolicyDecision

        base_context = tool_policy_context or {}
        context = ToolPolicyContext(
            agent_name=base_context.get("agent_name", ""),
            tool_name=tool_name,
            arguments=arguments,
            run_name=base_context.get("run_name"),
            trace_id=base_context.get("trace_id"),
            parent_run_id=base_context.get("parent_run_id"),
            group_id=base_context.get("group_id"),
            tags=list(base_context.get("tags", [])),
            metadata=dict(base_context.get("metadata", {})),
            messages=list(base_context.get("messages", [])),
            tool_metadata=tool_metadata,
        )

        if hasattr(tool_policy, "evaluate"):
            raw_decision = tool_policy.evaluate(context)
        else:
            raw_decision = tool_policy(context)

        if isinstance(raw_decision, ToolPolicyDecision):
            return raw_decision
        if isinstance(raw_decision, bool):
            return ToolPolicyDecision(allowed=raw_decision)
        raise TypeError(
            "Tool policy must return a bool or ToolPolicyDecision instance."
        )

    def execute_tool(
        self, tool_calls, tool_policy=None, tool_policy_context=None
    ) -> tuple[list, list]:
        """Executes registered tools based on the tool calls from the model.

        Args:
            tool_calls: List of tool calls from the model

        Returns:
            List of tuples containing (result, result_message) for each tool call
        """
        results = []
        messages = []
        self.last_policy_events = []
        self.last_tool_events = []

        # Handle single tool call or list of tool calls
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        for tool_call in tool_calls:
            # Handle both dictionary and object-style tool calls
            if isinstance(tool_call, dict):
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                tool_call_id = tool_call["id"]
            else:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_call_id = tool_call.id

            # Ensure arguments is a dict
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in self._tools:
                raise ValueError(f"Tool '{tool_name}' not registered.")

            tool = self._tools[tool_name]
            tool_func = tool["function"]
            param_model = tool["param_model"]
            tool_metadata = tool.get("metadata")
            tool_metadata_dict = (
                tool_metadata.to_dict() if tool_metadata is not None else None
            )

            # Validate and parse the arguments with Pydantic if a model exists
            try:
                validated_args = param_model(**arguments)
                validated_args_dict = validated_args.model_dump()
                trace_arguments, argument_artifacts = self._artifactized_trace_value(
                    validated_args_dict
                )
                decision = self._evaluate_tool_policy(
                    tool_policy,
                    tool_policy_context,
                    tool_name,
                    validated_args_dict,
                    tool_metadata,
                )
                if decision is not None:
                    policy_event = {
                        "tool_name": tool_name,
                        "allowed": decision.allowed,
                        "reason": decision.reason,
                        "metadata": decision.metadata,
                    }
                    if tool_metadata_dict is not None:
                        policy_event["tool_metadata"] = tool_metadata_dict
                    self.last_policy_events.append(policy_event)
                if decision is not None and not decision.allowed:
                    tool_event = {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "arguments": trace_arguments,
                        "allowed": False,
                        "reason": decision.reason,
                        "metadata": decision.metadata,
                    }
                    if argument_artifacts:
                        tool_event["argument_artifacts"] = argument_artifacts
                    if tool_metadata_dict is not None:
                        tool_event["tool_metadata"] = tool_metadata_dict
                    self.last_tool_events.append(tool_event)
                    self._emit_tool_trace_event("tool.denied", tool_event)
                    result = {
                        "error": "Tool call denied by policy",
                        "reason": decision.reason,
                    }
                else:
                    tool_event = {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "arguments": trace_arguments,
                        "allowed": True,
                        "reason": decision.reason if decision else None,
                        "metadata": decision.metadata if decision else {},
                    }
                    if argument_artifacts:
                        tool_event["argument_artifacts"] = argument_artifacts
                    if tool_metadata_dict is not None:
                        tool_event["tool_metadata"] = tool_metadata_dict
                    self.last_tool_events.append(tool_event)
                    self._emit_tool_trace_event("tool.allowed", tool_event)
                    self._emit_tool_trace_event("tool.started", tool_event)
                    try:
                        result = tool_func(**validated_args_dict)
                    except Exception as exc:
                        failed_event = {
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "status": "failed",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                        if tool_metadata_dict is not None:
                            failed_event["tool_metadata"] = tool_metadata_dict
                        self.last_tool_events.append(failed_event)
                        self._emit_tool_trace_event("tool.failed", failed_event)
                        raise
                    result_event = {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "status": "success",
                        "result_preview": _preview_tool_result(result),
                    }
                    artifactized_result, result_artifacts = (
                        self._artifactized_trace_value(result)
                    )
                    if result_artifacts:
                        result_event["result_artifacts"] = result_artifacts
                        result_event["result_preview"] = _preview_tool_result(
                            artifactized_result
                        )
                    if tool_metadata_dict is not None:
                        result_event["tool_metadata"] = tool_metadata_dict
                    self.last_tool_events.append(result_event)
                    self._emit_tool_trace_event("tool.completed", result_event)
                results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result),
                        "tool_call_id": tool_call_id,
                    }
                )
            except ValidationError as e:
                raise ValueError(f"Error in tool '{tool_name}' parameters: {e}")

        return results, messages
