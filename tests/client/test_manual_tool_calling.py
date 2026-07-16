"""Test for manual tool calling without max_turns parameter.

This test ensures that tool schemas are properly passed to providers
when max_turns is not specified (manual tool calling mode).

Regression test for: https://github.com/andrewyng/aisuite/pull/266
"""

from unittest.mock import Mock, patch
import pytest
from aisuite import Client


@pytest.fixture
def mock_provider():
    """Create a mock provider that can receive tool schemas."""
    provider = Mock()
    # Simulate a response with tool_calls
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = (
        None  # Set to None for tool call responses
    )
    mock_response.choices[0].message.tool_calls = [
        Mock(
            id="test_tool_call_id",
            function=Mock(name="test_function", arguments='{"arg": "value"}'),
            type="function",
        )
    ]
    mock_response.choices[0].finish_reason = "tool_calls"
    provider.chat_completions_create.return_value = mock_response
    return provider


def test_manual_tool_calling_preserves_tools_in_kwargs(mock_provider):
    """Test that tools parameter is passed to provider when max_turns is not specified.

    When using manual tool calling (no max_turns parameter), the tools schema
    should be passed to the provider so the LLM can see the available tools.

    This is a regression test for a bug where kwargs.pop("tools") was removing
    tools from kwargs before they could be passed to the provider.
    """
    client = Client()

    # Patch the provider factory to return our mock
    with patch(
        "aisuite.provider.ProviderFactory.create_provider", return_value=mock_provider
    ):
        messages = [{"role": "user", "content": "What time is it?"}]

        # Manual tool schema (OpenAI format)
        manual_tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # Call without max_turns (manual tool calling mode)
        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=messages,
            tools=manual_tool_schema,
        )

        # Verify provider was called
        assert mock_provider.chat_completions_create.called

        # Get the kwargs that were passed to the provider
        call_kwargs = mock_provider.chat_completions_create.call_args.kwargs

        # CRITICAL: tools must be in kwargs passed to provider
        assert "tools" in call_kwargs, (
            "tools parameter was not passed to provider. "
            "This breaks manual tool calling where LLM needs to see available tools."
        )
        assert call_kwargs["tools"] == manual_tool_schema

        # Verify response has tool_calls
        assert response.choices[0].message.tool_calls is not None


def test_auto_tool_calling_does_not_pass_tools_to_provider(mock_provider):
    """Test that tools parameter is NOT in kwargs when max_turns is specified.

    When using automatic tool execution (with max_turns), the tools are
    handled by aisuite's _tool_runner and should not be passed to the provider.
    """
    client = Client()

    # Create a callable tool function
    def get_current_time():
        """Get the current time"""
        return "12:00 PM"

    # Mock the _tool_runner to avoid actual execution
    with patch(
        "aisuite.provider.ProviderFactory.create_provider", return_value=mock_provider
    ):
        final_response = Mock()
        final_response.choices = [Mock()]
        final_response.choices[0].message = Mock()
        final_response.choices[0].message.content = "It is 12:00 PM"
        final_response.choices[0].message.tool_calls = None

        # Make _tool_runner return immediately
        with patch.object(
            client.chat.completions,
            "_tool_runner",
            return_value=final_response,
        ) as mock_tool_runner:
            messages = [{"role": "user", "content": "What time is it?"}]

            # Call with max_turns (automatic tool execution mode)
            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=messages,
                tools=[get_current_time],  # Callable function
                max_turns=5,
            )

            # Verify _tool_runner was called
            assert mock_tool_runner.called

            # Get kwargs passed to _tool_runner
            tool_runner_kwargs = mock_tool_runner.call_args.kwargs

            # tools should NOT be in kwargs passed to provider
            # because _tool_runner handles them separately
            assert "tools" not in tool_runner_kwargs
            assert response is final_response


def test_manual_tool_calling_converts_callables_to_specs(mock_provider):
    """Manual mode with callable tools: the provider receives OpenAI-format specs.

    Callables (including MCP-derived tools) can't be serialized into a provider
    request, so the manual path must convert them to schema dicts.
    """
    client = Client()

    def get_current_time():
        """Get the current time"""
        return "12:00 PM"

    with patch(
        "aisuite.provider.ProviderFactory.create_provider", return_value=mock_provider
    ):
        client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=[get_current_time],  # callable, no max_turns
        )

        specs = mock_provider.chat_completions_create.call_args.kwargs["tools"]
        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "get_current_time"


def test_manual_tool_calling_with_mcp_configs(mock_provider):
    """Test that MCP config dicts are properly processed for manual tool calling.

    When passing MCP configs in manual mode, they should be converted to
    callable tools, then back to schemas for the provider.
    """
    # This is a simplified test - full MCP testing is in tests/mcp/
    # Just verify the flow doesn't break with dict-based tools
    client = Client()

    with patch(
        "aisuite.provider.ProviderFactory.create_provider", return_value=mock_provider
    ):
        # Mock _process_mcp_configs to avoid needing actual MCP setup
        with patch.object(
            client.chat.completions,
            "_process_mcp_configs",
            return_value=([], []),  # Return empty tools and clients
        ):
            messages = [{"role": "user", "content": "Test"}]

            # Call with empty tools (after MCP processing)
            client.chat.completions.create(
                model="openai:gpt-4o",
                messages=messages,
                tools=[],  # Empty after MCP processing
            )

            # Should not raise an error
            assert mock_provider.chat_completions_create.called
