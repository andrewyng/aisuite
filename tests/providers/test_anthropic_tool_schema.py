"""Keep MCP JSON Schema references and constraints when sending tools to Claude."""

from copy import deepcopy

from pydantic import BaseModel, ConfigDict

from aisuite.mcp.tool_wrapper import MCPToolWrapper
from aisuite.providers.anthropic_provider import AnthropicMessageConverter
from aisuite.utils.tools import Tools


class Address(BaseModel):
    city: str


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    address: Address


def test_mcp_tool_schema_keeps_definitions_referenced_by_parameters():
    schema = Delivery.model_json_schema()
    original = deepcopy(schema)
    wrapper = MCPToolWrapper(
        object(),
        "schedule_delivery",
        {
            "description": "Schedule a delivery",
            "inputSchema": schema,
        },
    )
    openai_tools = Tools([wrapper]).tools()

    converted = AnthropicMessageConverter().convert_tool_spec(openai_tools)[0]
    parameters = converted["input_schema"]
    reference = parameters["properties"]["address"]["$ref"]
    assert reference == "#/$defs/Address"
    assert parameters["$defs"]["Address"] == original["$defs"]["Address"]
    assert schema == original


def test_tool_schema_keeps_root_constraints():
    schema = Delivery.model_json_schema()
    schema["minProperties"] = 1
    tool = {
        "type": "function",
        "function": {
            "name": "schedule_delivery",
            "description": "Schedule a delivery",
            "parameters": schema,
        },
    }

    converted = AnthropicMessageConverter().convert_tool_spec([tool])[0]

    assert converted["input_schema"]["additionalProperties"] is False
    assert converted["input_schema"]["minProperties"] == 1
    assert converted["input_schema"] == schema
