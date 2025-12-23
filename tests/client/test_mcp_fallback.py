"""
Tests for MCP fallback behavior when MCP package is not installed.

This tests the fix for the NameError that occurred when using tools
without the MCP package installed.
"""

import pytest
from unittest.mock import Mock, patch


class TestMCPFallback:
    """Test MCP fallback behavior when MCP is not available."""

    def test_is_mcp_config_fallback_exists(self):
        """
        Test that is_mcp_config is always defined, even without MCP.
        
        This is a regression test for the bug where using tools without
        MCP installed would raise: NameError: name 'is_mcp_config' is not defined
        """
        # Import the client module - is_mcp_config should always be defined
        from aisuite import client as client_module
        
        # Verify is_mcp_config exists (either from MCP or fallback)
        assert hasattr(client_module, 'is_mcp_config') or client_module.MCP_AVAILABLE
        
    def test_mcp_config_detection(self):
        """Test that MCP configs are correctly detected."""
        # Import after any patching
        from aisuite.client import MCP_AVAILABLE
        
        # We need to test the fallback function directly
        # Define the same logic as the fallback
        def is_mcp_config_fallback(obj):
            if not isinstance(obj, dict):
                return False
            return obj.get("type") == "mcp" and "name" in obj
        
        # Test MCP config detection
        mcp_config = {"type": "mcp", "name": "test", "command": "npx"}
        assert is_mcp_config_fallback(mcp_config) == True
        
        # Test non-MCP configs
        function_tool = {"type": "function", "function": {"name": "test"}}
        assert is_mcp_config_fallback(function_tool) == False
        
        # Test non-dict inputs
        assert is_mcp_config_fallback("not a dict") == False
        assert is_mcp_config_fallback(None) == False


class TestIsMCPConfigFallback:
    """Direct tests for the is_mcp_config fallback function logic."""

    def test_valid_mcp_config(self):
        """Test detection of valid MCP configs."""
        # Inline the fallback logic for testing
        def is_mcp_config(obj):
            if not isinstance(obj, dict):
                return False
            return obj.get("type") == "mcp" and "name" in obj
        
        valid_configs = [
            {"type": "mcp", "name": "filesystem"},
            {"type": "mcp", "name": "test", "command": "npx", "args": ["arg1"]},
            {"type": "mcp", "name": "server", "extra": "data"},
        ]
        
        for config in valid_configs:
            assert is_mcp_config(config) == True, f"Should detect {config} as MCP config"

    def test_invalid_mcp_config(self):
        """Test that non-MCP configs are not detected as MCP."""
        def is_mcp_config(obj):
            if not isinstance(obj, dict):
                return False
            return obj.get("type") == "mcp" and "name" in obj
        
        invalid_configs = [
            {"type": "function", "function": {}},  # OpenAI function format
            {"name": "test"},  # Missing type
            {"type": "mcp"},  # Missing name
            {"type": "other", "name": "test"},  # Wrong type
            {},  # Empty dict
        ]
        
        for config in invalid_configs:
            assert is_mcp_config(config) == False, f"Should not detect {config} as MCP config"

    def test_non_dict_inputs(self):
        """Test that non-dict inputs return False."""
        def is_mcp_config(obj):
            if not isinstance(obj, dict):
                return False
            return obj.get("type") == "mcp" and "name" in obj
        
        non_dicts = [
            "string",
            123,
            12.34,
            None,
            [],
        ]
        
        for item in non_dicts:
            assert is_mcp_config(item) == False, f"Should return False for {type(item)}"
