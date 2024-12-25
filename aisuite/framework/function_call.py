from typing import Dict, Any, Callable
from pydantic import BaseModel, Field
import inspect
from typing import get_type_hints, get_args, get_origin


class FunctionCall(BaseModel):
    name: str = Field(..., description="The name of the function to call")
    arguments: Dict[str, Any] = Field(
        ..., description="The arguments to pass to the function"
    )


class Argument(BaseModel):
    type: str
    description: str
    # enum: Optional[list[str]] = None


class InputSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Argument]
    required: list[str] = []


class FunctionCallingSchema(BaseModel):
    name: str
    description: str
    input_schema: InputSchema

    def evaluate(self, function: Callable) -> bool:
        """Required fields are present InputSchema"""
        return all(
            field in self.input_schema.properties
            for field in self.input_schema.required
        )


def _get_type_info(type_hint) -> dict:
    """Helper function to get type information from type hints"""
    origin = get_origin(type_hint)
    if origin is None:
        if type_hint == str:
            return {"type": "string"}
        elif type_hint == int:
            return {"type": "integer"}
        elif type_hint == float:
            return {"type": "number"}
        elif type_hint == bool:
            return {"type": "boolean"}
    elif origin == list:
        item_type = get_args(type_hint)[0]
        return {"type": "array", "items": _get_type_info(item_type)}
    return {"type": "string"}  # default fallback


def generate_function_calling_schema(function: Callable) -> dict:
    """Construct the function calling schema from the function signature"""
    sig = inspect.signature(function)
    type_hints = get_type_hints(function)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        type_info = _get_type_info(param_type)

        param_doc = ""
        if function.__doc__:
            # Try to extract parameter documentation from docstring
            param_lines = [
                line.strip()
                for line in function.__doc__.split("\n")
                if f":param {param_name}:" in line
            ]
            if param_lines:
                param_doc = param_lines[0].split(":param {param_name}:")[1].strip()

        properties[param_name] = Argument(
            type=type_info["type"],
            description=param_doc or f"Parameter {param_name}",
            **{k: v for k, v in type_info.items() if k != "type"},
        )

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    input_schema = InputSchema(properties=properties)
    input_schema.required = required  # Add required field to the parameters schema

    schema = FunctionCallingSchema(
        name=function.__name__,
        description=function.__doc__ or "",
        input_schema=input_schema,
    )

    return schema.model_dump()
