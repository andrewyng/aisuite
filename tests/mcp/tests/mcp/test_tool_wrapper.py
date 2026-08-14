"""Unit tests for MCPToolWrapper's signature construction."""

import inspect
import unittest

from aisuite.mcp.tool_wrapper import MCPToolWrapper


class TestCreateSignatureParameterOrdering(unittest.TestCase):
    """Regression tests for a schema-order bug in _create_signature.

    inspect.Signature (like Python's own `def` syntax) requires every
    non-default parameter to precede any default parameter. JSON Schema's
    `required` list is independent of `properties` order, so a tool schema
    listing an optional property before a required one used to raise
    "ValueError: non-default argument follows default argument" while
    constructing the wrapper. This is not a rare shape: most interaction
    tools in the official @playwright/mcp server list an optional
    human-readable `element` description before the required `target`
    element reference (browser_click, browser_type, browser_hover,
    browser_drag, browser_take_screenshot, browser_evaluate,
    browser_select_option all use this pattern).
    """

    def test_optional_before_required_does_not_raise(self):
        schema = {
            "name": "browser_click",
            "description": "Click an element",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Human-readable element description",
                    },
                    "target": {"type": "string", "description": "Element target ref"},
                },
                "required": ["target"],
            },
        }

        # Should not raise ValueError: non-default argument follows default argument
        wrapper = MCPToolWrapper(
            mcp_client=None, tool_name="browser_click", tool_schema=schema
        )

        parameters = list(wrapper.__signature__.parameters.values())
        self.assertEqual([p.name for p in parameters], ["target", "element"])
        self.assertEqual(parameters[0].default, inspect.Parameter.empty)
        self.assertIsNone(parameters[1].default)

    def test_multiple_required_and_optional_are_each_grouped_and_stable(self):
        schema = {
            "name": "browser_drag",
            "description": "Drag from one element to another",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "startElement": {"type": "string"},
                    "startTarget": {"type": "string"},
                    "endElement": {"type": "string"},
                    "endTarget": {"type": "string"},
                },
                "required": ["startTarget", "endTarget"],
            },
        }

        wrapper = MCPToolWrapper(
            mcp_client=None, tool_name="browser_drag", tool_schema=schema
        )

        parameters = list(wrapper.__signature__.parameters.values())
        # Required params keep their relative order, then optional params
        # keep theirs -- this is a stable sort, not a semantic reordering.
        self.assertEqual(
            [p.name for p in parameters],
            ["startTarget", "endTarget", "startElement", "endElement"],
        )

    def test_all_required_or_all_optional_schemas_still_work(self):
        all_required_schema = {
            "name": "tool_a",
            "description": "",
            "inputSchema": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
        }
        all_optional_schema = {
            "name": "tool_b",
            "description": "",
            "inputSchema": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": [],
            },
        }

        wrapper_a = MCPToolWrapper(
            mcp_client=None, tool_name="tool_a", tool_schema=all_required_schema
        )
        wrapper_b = MCPToolWrapper(
            mcp_client=None, tool_name="tool_b", tool_schema=all_optional_schema
        )

        self.assertEqual(
            [p.name for p in wrapper_a.__signature__.parameters.values()], ["a", "b"]
        )
        self.assertEqual(
            [p.name for p in wrapper_b.__signature__.parameters.values()], ["a", "b"]
        )


if __name__ == "__main__":
    unittest.main()
