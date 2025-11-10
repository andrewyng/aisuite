import unittest
from typing import Dict, Any, List
from aisuite.utils.tools import Tools


class MockMCPToolWrapper:
    """Mock MCP tool wrapper for testing schema preservation."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.__name__ = name
        self.__doc__ = description
        self.__mcp_input_schema__ = input_schema

    def __call__(self, **kwargs):
        """Mock execution."""
        return {"result": "success", "args": kwargs}


class TestToolsMCPSchema(unittest.TestCase):
    """Test suite for MCP schema handling in Tools class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool_manager = Tools()

    def test_mcp_tool_with_simple_types(self):
        """Test MCP tool with simple types (string, integer, boolean)."""
        input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer", "description": "User age"},
                "active": {"type": "boolean", "description": "Is active"},
            },
            "required": ["name"],
        }

        tool = MockMCPToolWrapper("test_simple", "A simple test tool", input_schema)
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        self.assertEqual(len(tools), 1)

        # Verify the schema was preserved exactly
        tool_spec = tools[0]["function"]
        self.assertEqual(tool_spec["name"], "test_simple")
        self.assertEqual(tool_spec["description"], "A simple test tool")
        self.assertEqual(tool_spec["parameters"], input_schema)

    def test_mcp_tool_with_array_of_objects(self):
        """Test MCP tool with array of objects (List[dict])."""
        input_schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                        },
                        "required": ["name", "type"],
                    },
                    "description": "List of entities to create",
                }
            },
            "required": ["entities"],
        }

        tool = MockMCPToolWrapper(
            "create_entities", "Create multiple entities", input_schema
        )
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        tool_spec = tools[0]["function"]

        # Verify complex array schema is preserved
        self.assertEqual(
            tool_spec["parameters"]["properties"]["entities"]["type"], "array"
        )
        self.assertIn("items", tool_spec["parameters"]["properties"]["entities"])
        self.assertEqual(
            tool_spec["parameters"]["properties"]["entities"]["items"]["type"], "object"
        )

    def test_mcp_tool_with_nested_objects(self):
        """Test MCP tool with nested object structures."""
        input_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                }
            },
            "required": ["user"],
        }

        tool = MockMCPToolWrapper(
            "create_user", "Create user with address", input_schema
        )
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        tool_spec = tools[0]["function"]

        # Verify nested structure is preserved
        self.assertEqual(tool_spec["parameters"], input_schema)
        self.assertIn(
            "address", tool_spec["parameters"]["properties"]["user"]["properties"]
        )

    def test_mcp_tool_with_array_of_strings(self):
        """Test MCP tool with array of simple types (List[str])."""
        input_schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags",
                }
            },
            "required": ["tags"],
        }

        tool = MockMCPToolWrapper("add_tags", "Add tags to item", input_schema)
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        tool_spec = tools[0]["function"]

        # Verify array of strings is preserved
        self.assertEqual(tool_spec["parameters"]["properties"]["tags"]["type"], "array")
        self.assertEqual(
            tool_spec["parameters"]["properties"]["tags"]["items"]["type"], "string"
        )

    def test_mcp_tool_detection(self):
        """Test that MCP tools are properly detected via __mcp_input_schema__ attribute."""
        input_schema = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        }

        tool = MockMCPToolWrapper("mcp_tool", "An MCP tool", input_schema)

        # Verify the attribute exists
        self.assertTrue(hasattr(tool, "__mcp_input_schema__"))
        self.assertEqual(tool.__mcp_input_schema__, input_schema)

    def test_mcp_tool_with_optional_parameters(self):
        """Test MCP tool with mix of required and optional parameters."""
        input_schema = {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "description": "Required parameter",
                },
                "optional_param": {
                    "type": "integer",
                    "description": "Optional parameter",
                },
            },
            "required": ["required_param"],
        }

        tool = MockMCPToolWrapper(
            "mixed_params", "Tool with mixed params", input_schema
        )
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        tool_spec = tools[0]["function"]

        # Verify required fields are correct
        self.assertEqual(tool_spec["parameters"]["required"], ["required_param"])
        self.assertIn("required_param", tool_spec["parameters"]["properties"])
        self.assertIn("optional_param", tool_spec["parameters"]["properties"])

    def test_mcp_schema_preserves_additional_fields(self):
        """Test that additional JSON Schema fields are preserved (enum, format, etc.)."""
        input_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                    "description": "Status value",
                },
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "Email address",
                },
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Count value",
                },
            },
            "required": ["status"],
        }

        tool = MockMCPToolWrapper(
            "advanced_schema", "Tool with advanced schema", input_schema
        )
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        tool_spec = tools[0]["function"]

        # Verify enum is preserved
        self.assertIn("enum", tool_spec["parameters"]["properties"]["status"])
        self.assertEqual(
            tool_spec["parameters"]["properties"]["status"]["enum"],
            ["active", "inactive", "pending"],
        )

        # Verify format is preserved
        self.assertIn("format", tool_spec["parameters"]["properties"]["email"])
        self.assertEqual(
            tool_spec["parameters"]["properties"]["email"]["format"], "email"
        )

        # Verify min/max are preserved
        self.assertIn("minimum", tool_spec["parameters"]["properties"]["count"])
        self.assertIn("maximum", tool_spec["parameters"]["properties"]["count"])

    def test_mcp_tool_execution_with_validation(self):
        """Test that MCP tools can be executed and parameters are validated."""
        input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name"},
                "count": {"type": "integer", "description": "Count"},
            },
            "required": ["name"],
        }

        tool = MockMCPToolWrapper(
            "validate_tool", "Tool for validation test", input_schema
        )
        self.tool_manager._add_tool(tool)

        # Test valid execution
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "validate_tool",
                "arguments": {"name": "test", "count": 5},
            },
        }

        results, messages = self.tool_manager.execute_tool(tool_call)
        self.assertEqual(len(results), 1)
        self.assertIn("result", results[0])

    def test_mcp_tool_with_empty_schema(self):
        """Test MCP tool with no parameters."""
        input_schema = {"type": "object", "properties": {}, "required": []}

        tool = MockMCPToolWrapper("no_params", "Tool with no params", input_schema)
        self.tool_manager._add_tool(tool)

        tools = self.tool_manager.tools()
        tool_spec = tools[0]["function"]

        self.assertEqual(tool_spec["parameters"]["properties"], {})
        self.assertEqual(tool_spec["parameters"]["required"], [])

    def test_multiple_mcp_tools(self):
        """Test adding multiple MCP tools to the manager."""
        schema1 = {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
            "required": ["param1"],
        }

        schema2 = {
            "type": "object",
            "properties": {"param2": {"type": "integer"}},
            "required": ["param2"],
        }

        tool1 = MockMCPToolWrapper("tool1", "First tool", schema1)
        tool2 = MockMCPToolWrapper("tool2", "Second tool", schema2)

        self.tool_manager._add_tool(tool1)
        self.tool_manager._add_tool(tool2)

        tools = self.tool_manager.tools()
        self.assertEqual(len(tools), 2)

        tool_names = [tool["function"]["name"] for tool in tools]
        self.assertIn("tool1", tool_names)
        self.assertIn("tool2", tool_names)

    def test_mcp_tool_schema_not_modified(self):
        """Test that the original schema is not modified during processing."""
        original_schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    },
                }
            },
            "required": ["data"],
        }

        # Create a copy to verify immutability
        import copy

        schema_copy = copy.deepcopy(original_schema)

        tool = MockMCPToolWrapper(
            "immutable_tool", "Test immutability", original_schema
        )
        self.tool_manager._add_tool(tool)

        # Verify original schema wasn't modified
        self.assertEqual(original_schema, schema_copy)

    def test_backward_compatibility_non_mcp_tools(self):
        """Test that regular Python functions still work (backward compatibility)."""

        def regular_function(name: str, age: int = 25) -> Dict[str, Any]:
            """A regular Python function."""
            return {"name": name, "age": age}

        # Regular functions don't have __mcp_input_schema__
        self.assertFalse(hasattr(regular_function, "__mcp_input_schema__"))

        # Should still work with the Tools class
        self.tool_manager._add_tool(regular_function)

        tools = self.tool_manager.tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["function"]["name"], "regular_function")


if __name__ == "__main__":
    unittest.main()
