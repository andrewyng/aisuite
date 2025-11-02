import unittest
from pydantic import BaseModel
from typing import Dict
from aisuite.utils.tools import Tools  # Import your ToolManager class
from enum import Enum
import dataclasses
from datetime import datetime
from markupsafe import Markup


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


@dataclasses.dataclass
class WeatherReport:
    temperature: int
    temperature_unit: TemperatureUnit = TemperatureUnit.CELSIUS
    last_forecast_datetime: datetime = datetime(2024, 1, 1, 11, 5, 5)
    forecast: Markup = Markup(
        """
    <h1>Weather Report for Today</h1>
    <section>It's looking really good.</section>
    """
    )


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


class WeatherReportParams(BaseModel):
    location: str


def get_weather_report_complex_type(location: str) -> WeatherReport:
    """Gets a detailed weather report for a location"""
    return WeatherReport(
        20,
        TemperatureUnit.CELSIUS,
    )


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

    def test_execute_tool_with_complex_type(self):
        """Test executing a registered tool with valid parameters."""
        self.tool_manager._add_tool(
            get_weather_report_complex_type, WeatherReportParams
        )
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "get_weather_report_complex_type",
                "arguments": {"location": "San Francisco"},
            },
        }
        result, result_message = self.tool_manager.execute_tool(tool_call)

        # Assuming result is returned as a list with a single dictionary
        result_obj = result[0] if isinstance(result, list) else result
        self.assertIsInstance(result_obj, WeatherReport)

        # Check that the result matches expected output
        self.assertEqual(
            result_obj.forecast,
            Markup(
                "\n    <h1>Weather Report for Today</h1>\n    <section>It's looking really good.</section>\n    "
            ),
        )
        self.assertEqual(
            result_obj.last_forecast_datetime, datetime(2024, 1, 1, 11, 5, 5)
        )


if __name__ == "__main__":
    unittest.main()
