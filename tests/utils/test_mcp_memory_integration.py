"""
Integration test simulating the memory MCP server schema that was causing BadRequestError.

This test verifies that the fix for List[dict] schema conversion works correctly.
"""

import unittest
from aisuite.utils.tools import Tools


class MockMemoryMCPTool:
    """Mock memory MCP tool with the exact schema that was failing."""

    def __init__(self):
        self.__name__ = "create_entities"
        self.__doc__ = "Create multiple entities in the knowledge graph"

        # This is the exact schema from the memory MCP server that was causing:
        # "Invalid schema for function 'create_entities': 'typing.List[dict]' is not valid"
        self.__mcp_input_schema__ = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "List of entities to create",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the entity",
                            },
                            "entityType": {
                                "type": "string",
                                "description": "The type of the entity",
                            },
                            "observations": {
                                "type": "array",
                                "description": "An array of observation contents",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "entityType", "observations"],
                    },
                }
            },
            "required": ["entities"],
        }

    def __call__(self, **kwargs):
        """Mock execution."""
        return {"created": len(kwargs.get("entities", [])), "status": "success"}


class TestMCPMemoryIntegration(unittest.TestCase):
    """Test that the memory server schema works correctly."""

    def test_memory_create_entities_schema(self):
        """Test that create_entities schema is converted correctly for OpenAI."""
        tool_manager = Tools()
        memory_tool = MockMemoryMCPTool()

        # This should not raise an error anymore
        tool_manager._add_tool(memory_tool)

        # Get the OpenAI format tools
        tools = tool_manager.tools()

        self.assertEqual(len(tools), 1)
        tool_spec = tools[0]["function"]

        # Verify the structure matches OpenAI expectations
        self.assertEqual(tool_spec["name"], "create_entities")
        self.assertEqual(tool_spec["parameters"]["type"], "object")
        self.assertIn("entities", tool_spec["parameters"]["properties"])

        # Verify the array type is preserved correctly (not 'typing.List[dict]')
        entities_param = tool_spec["parameters"]["properties"]["entities"]
        self.assertEqual(entities_param["type"], "array")
        self.assertIn("items", entities_param)
        self.assertEqual(entities_param["items"]["type"], "object")

        # Verify nested array (observations) is also preserved
        observations = entities_param["items"]["properties"]["observations"]
        self.assertEqual(observations["type"], "array")
        self.assertEqual(observations["items"]["type"], "string")

    def test_memory_tool_openai_format_validation(self):
        """Verify the output format would be accepted by OpenAI API."""
        tool_manager = Tools()
        memory_tool = MockMemoryMCPTool()
        tool_manager._add_tool(memory_tool)

        tools = tool_manager.tools()
        tool_spec = tools[0]["function"]

        # Check that there are no Python type strings in the schema
        import json

        schema_json = json.dumps(tool_spec["parameters"])

        # These should NOT appear in valid OpenAI JSON Schema
        self.assertNotIn("typing.", schema_json)
        self.assertNotIn("List[", schema_json)
        self.assertNotIn("Dict[", schema_json)

        # Only valid JSON Schema types should appear
        self.assertIn('"type": "array"', schema_json)
        self.assertIn('"type": "object"', schema_json)
        self.assertIn('"type": "string"', schema_json)

    def test_memory_tool_execution(self):
        """Test that the tool can be executed with proper validation."""
        tool_manager = Tools()
        memory_tool = MockMemoryMCPTool()
        tool_manager._add_tool(memory_tool)

        # Simulate a tool call from the LLM
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "create_entities",
                "arguments": {
                    "entities": [
                        {
                            "name": "MCP",
                            "entityType": "Protocol",
                            "observations": [
                                "Enables LLM tool calling",
                                "Uses JSON Schema",
                            ],
                        },
                        {
                            "name": "aisuite",
                            "entityType": "Library",
                            "observations": ["Unified API", "Multi-provider support"],
                        },
                    ]
                },
            },
        }

        # This should execute successfully
        results, messages = tool_manager.execute_tool(tool_call)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["created"], 2)
        self.assertEqual(results[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
