import unittest
from pydantic import BaseModel
from typing import Dict, Literal, Optional
from aisuite.utils.tools import Tools  # Import your ToolManager class
from enum import Enum


# Define a sample tool function and Pydantic model for testing
class TemperatureUnit(str, Enum):
    CELSIUS = "Celsius"
    FAHRENHEIT = "Fahrenheit"


class TemperatureParamsV2(BaseModel):
    location: str
    unit: TemperatureUnit = TemperatureUnit.CELSIUS


class TemperatureParams(BaseModel):
    location: str
    unit: str = "Celsius"


def get_current_temperature(location: str, unit: str = "Celsius") -> Dict[str, str]:
    """Gets the current temperature for a specific location and unit."""
    return {"location": location, "unit": unit, "temperature": "72"}


def missing_annotation_tool(location, unit="Celsius"):
    """Tool function without type annotations."""
    return {"location": location, "unit": unit, "temperature": "72"}


def get_current_temperature_v2(
    location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS
) -> Dict[str, str]:
    """Gets the current temperature for a specific location and unit (with enum support)."""
    return {"location": location, "unit": unit, "temperature": "72"}


def analyze_items(
    tags: list[str],
    scores: dict[str, int],
    mode: Literal["fast", "thorough"] = "fast",
    note: Optional[str] = None,
    limit: int | None = None,
) -> Dict[str, object]:
    """Analyze items using complex argument types."""
    return {
        "tags": tags,
        "scores": scores,
        "mode": mode,
        "note": note,
        "limit": limit,
    }


class TestToolManager(unittest.TestCase):
    def setUp(self):
        self.tool_manager = Tools()

    def test_add_tool_with_pydantic_model(self):
        """Test adding a tool with an explicit Pydantic model."""
        self.tool_manager._add_tool(get_current_temperature, TemperatureParams)

        expected_tool_spec = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Gets the current temperature for a specific location and unit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "",
                            },
                            "unit": {
                                "type": "string",
                                "description": "",
                                "default": "Celsius",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        tools = self.tool_manager.tools()
        self.assertIn(
            "get_current_temperature", [tool["function"]["name"] for tool in tools]
        )
        assert (
            tools == expected_tool_spec
        ), f"Expected {expected_tool_spec}, but got {tools}"

    def test_add_tool_with_signature_inference(self):
        """Test adding a tool and inferring parameters from the function signature."""
        self.tool_manager._add_tool(get_current_temperature)
        # Expected output from tool_manager.tools() when called with OpenAI format
        expected_tool_spec = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Gets the current temperature for a specific location and unit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "",  # No description provided in function signature
                            },
                            "unit": {
                                "type": "string",
                                "description": "",
                                "default": "Celsius",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        tools = self.tool_manager.tools()
        print(tools)
        self.assertIn(
            "get_current_temperature", [tool["function"]["name"] for tool in tools]
        )
        assert (
            tools == expected_tool_spec
        ), f"Expected {expected_tool_spec}, but got {tools}"

    def test_add_tool_missing_annotation_raises_exception(self):
        """Test that adding a tool with missing type annotations raises a TypeError."""
        with self.assertRaises(TypeError):
            self.tool_manager._add_tool(missing_annotation_tool)

    def test_execute_tool_valid_parameters(self):
        """Test executing a registered tool with valid parameters."""
        self.tool_manager._add_tool(get_current_temperature, TemperatureParams)
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "get_current_temperature",
                "arguments": {"location": "San Francisco", "unit": "Celsius"},
            },
        }
        result, result_message = self.tool_manager.execute_tool(tool_call)

        # Assuming result is returned as a list with a single dictionary
        result_dict = result[0] if isinstance(result, list) else result

        # Check that the result matches expected output
        self.assertEqual(result_dict["location"], "San Francisco")
        self.assertEqual(result_dict["unit"], "Celsius")
        self.assertEqual(result_dict["temperature"], "72")

    def test_execute_tool_invalid_parameters(self):
        """Test that executing a tool with invalid parameters raises a ValueError."""
        self.tool_manager._add_tool(get_current_temperature, TemperatureParams)
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "get_current_temperature",
                "arguments": {"location": 123},  # Invalid type for location
            },
        }

        with self.assertRaises(ValueError) as context:
            self.tool_manager.execute_tool(tool_call)

        # Verify the error message contains information about the validation error
        self.assertIn(
            "Error in tool 'get_current_temperature' parameters", str(context.exception)
        )

    def test_add_tool_with_enum(self):
        """Test adding a tool with an enum parameter."""
        self.tool_manager._add_tool(get_current_temperature_v2, TemperatureParamsV2)

        expected_tool_spec = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature_v2",
                    "description": "Gets the current temperature for a specific location and unit (with enum support).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["Celsius", "Fahrenheit"],
                                "description": "",
                                "default": "Celsius",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        tools = self.tool_manager.tools()
        assert (
            tools == expected_tool_spec
        ), f"Expected {expected_tool_spec}, but got {tools}"

    def test_add_tool_with_complex_annotations_uses_json_schema_types(self):
        """Test complex annotations generate valid JSON Schema types."""
        self.tool_manager._add_tool(analyze_items)

        parameters = self.tool_manager.tools()[0]["function"]["parameters"]
        properties = parameters["properties"]

        self.assertEqual(properties["tags"]["type"], "array")
        self.assertEqual(properties["tags"]["items"], {"type": "string"})
        self.assertEqual(properties["scores"]["type"], "object")
        self.assertEqual(
            properties["scores"]["additionalProperties"], {"type": "integer"}
        )
        self.assertEqual(properties["mode"]["type"], "string")
        self.assertEqual(properties["mode"]["enum"], ["fast", "thorough"])
        self.assertEqual(properties["mode"]["default"], "fast")
        self.assertEqual(properties["note"]["type"], "string")
        self.assertIsNone(properties["note"]["default"])
        self.assertEqual(properties["limit"]["type"], "integer")
        self.assertIsNone(properties["limit"]["default"])
        self.assertEqual(parameters["required"], ["tags", "scores"])

        for schema in properties.values():
            self.assertNotIn("list[", str(schema))
            self.assertNotIn("dict[", str(schema))
            self.assertNotIn("| None", str(schema))

    def test_execute_tool_with_complex_annotations(self):
        """Test Pydantic validation still executes complex-typed tools."""
        self.tool_manager._add_tool(analyze_items)
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "analyze_items",
                "arguments": {
                    "tags": ["urgent", "finance"],
                    "scores": {"urgent": 10, "finance": 7},
                    "mode": "thorough",
                    "limit": 3,
                },
            },
        }

        result, _ = self.tool_manager.execute_tool(tool_call)

        self.assertEqual(
            result[0],
            {
                "tags": ["urgent", "finance"],
                "scores": {"urgent": 10, "finance": 7},
                "mode": "thorough",
                "note": None,
                "limit": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
