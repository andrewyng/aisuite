"""
Real LLM End-to-End Tests for HTTP MCP Integration.

These tests make ACTUAL API calls to LLM providers (OpenAI, Anthropic) to verify
that HTTP-based MCP tools work correctly with real models. Unlike test_http_transport.py
which mocks HTTP responses, these tests verify the complete integration stack.

⚠️ WARNING: These tests will make real API calls and incur costs!
   - Each test costs ~$0.01-0.05 depending on the model
   - Tests are marked with @pytest.mark.llm
   - Tests are skipped if API keys are not present

MCP Server Used:
   - Context7 (https://mcp.context7.com/mcp)
   - Public HTTP MCP server for library documentation
   - No authentication required (optional API key for higher limits)
   - Tools: resolve-library-id, get-library-docs

Requirements:
    - API keys in .env file:
        OPENAI_API_KEY=your-key
        ANTHROPIC_API_KEY=your-key
    - pytest-asyncio, python-dotenv

Running:
    # Run ONLY HTTP LLM tests (⚠️ costs money):
    pytest tests/mcp/test_http_llm_e2e.py -v -m llm

    # Skip LLM tests (default, free):
    pytest tests/mcp/ -v -m "integration and not llm"
"""

import pytest
import os
from aisuite import Client


# Helper functions to check if we have API keys
def has_openai_key():
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key():
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


@pytest.mark.llm
@pytest.mark.integration
class TestOpenAIWithHTTPMCP:
    """Test OpenAI models with HTTP MCP tools (Context7)."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_gpt4o_resolves_library_via_http_mcp(self):
        """Test GPT-4o can resolve library names using HTTP MCP."""
        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": 'Use resolve-library-id to resolve the library name "requests" and tell me the library ID.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "allowed_tools": ["resolve-library-id"],
                }
            ],
            max_turns=3,
        )

        # Verify the LLM used the HTTP MCP tool
        content = response.choices[0].message.content.lower()
        # Should mention requests or library ID
        assert any(
            keyword in content
            for keyword in ["requests", "library", "id", "pypi", "python"]
        ), f"Expected library resolution info in response, got: {content}"

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_gpt4o_gets_library_docs_via_http_mcp(self):
        """Test GPT-4o can get library documentation using HTTP MCP."""
        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": 'First use resolve-library-id to get the ID for "requests", then use get-library-docs to fetch its documentation.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "allowed_tools": ["resolve-library-id", "get-library-docs"],
                }
            ],
            max_turns=5,
        )

        # Verify the LLM got documentation
        content = response.choices[0].message.content.lower()
        # Should mention documentation or requests library
        assert any(
            keyword in content
            for keyword in ["documentation", "requests", "http", "api", "library"]
        ), f"Expected documentation content in response, got: {content}"

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_gpt4o_mixed_tools_http(self):
        """Test GPT-4o with both HTTP MCP tools and regular Python functions."""

        # Define a Python function
        def get_current_year() -> str:
            """Get the current year."""
            from datetime import datetime

            return str(datetime.now().year)

        client = Client()

        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": 'First use get_current_year to get the year, then use resolve-library-id to resolve "requests".',
                }
            ],
            tools=[
                get_current_year,  # Python function
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "allowed_tools": ["resolve-library-id"],
                },  # HTTP MCP
            ],
            max_turns=5,
        )

        # Verify both tools were used
        content = response.choices[0].message.content.lower()
        # Should mention the year (from Python function)
        assert any(
            str(y) in content for y in [2024, 2025, 2026]
        ), f"Expected year in response, got: {content}"
        # Should mention requests or library (from HTTP MCP tool)
        assert any(
            keyword in content for keyword in ["requests", "library", "id"]
        ), f"Expected library info in response, got: {content}"


@pytest.mark.llm
@pytest.mark.integration
class TestAnthropicWithHTTPMCP:
    """Test Anthropic Claude models with HTTP MCP tools (Context7)."""

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_claude_resolves_library_via_http_mcp(self):
        """Test Claude can resolve library names using HTTP MCP."""
        client = Client()

        response = client.chat.completions.create(
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": 'Use resolve-library-id to resolve the library name "flask" and tell me the library ID.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "allowed_tools": ["resolve-library-id"],
                }
            ],
            max_turns=3,
        )

        # Verify Claude used the HTTP MCP tool
        content = response.choices[0].message.content.lower()
        assert any(
            keyword in content
            for keyword in ["flask", "library", "id", "pypi", "python"]
        ), f"Expected library resolution info in response, got: {content}"

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_claude_gets_library_docs_via_http_mcp(self):
        """Test Claude can get library documentation using HTTP MCP."""
        client = Client()

        response = client.chat.completions.create(
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": 'Use resolve-library-id to get the ID for "flask", then use get-library-docs to fetch documentation.',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "allowed_tools": ["resolve-library-id", "get-library-docs"],
                }
            ],
            max_turns=5,
        )

        # Verify Claude got documentation
        content = response.choices[0].message.content.lower()
        assert any(
            keyword in content
            for keyword in ["documentation", "flask", "web", "library"]
        ), f"Expected documentation content in response, got: {content}"

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_claude_mixed_tools_http(self):
        """Test Claude with both HTTP MCP tools and regular Python functions."""

        def get_language() -> str:
            """Get the primary programming language."""
            return "Python"

        client = Client()

        response = client.chat.completions.create(
            model="anthropic:claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": 'Use get_language to get the language, then use resolve-library-id to resolve "django". Tell me both.',
                }
            ],
            tools=[
                get_language,  # Python function
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "allowed_tools": ["resolve-library-id"],
                },  # HTTP MCP
            ],
            max_turns=5,
        )

        # Verify both tools were used
        content = response.choices[0].message.content.lower()
        # Should mention Python (from Python function)
        assert "python" in content, f"Expected Python in response, got: {content}"
        # Should mention django or library (from HTTP MCP tool)
        assert any(
            keyword in content for keyword in ["django", "library", "id"]
        ), f"Expected library info in response, got: {content}"


@pytest.mark.llm
@pytest.mark.integration
class TestHTTPMCPConfigDict:
    """Test HTTP MCP with config dict format."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_http_mcp_config_dict_format(self):
        """Test that HTTP MCP works with config dict format."""
        client = Client()

        # Using config dict format (not explicit MCPClient)
        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": 'Use resolve-library-id to resolve the library name "numpy".',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "timeout": 60.0,  # Test timeout parameter
                    "allowed_tools": ["resolve-library-id"],
                }
            ],
            max_turns=3,
        )

        # Verify it worked
        content = response.choices[0].message.content.lower()
        assert any(
            keyword in content
            for keyword in ["numpy", "library", "id", "pypi", "python"]
        ), f"Expected library resolution info in response, got: {content}"


@pytest.mark.llm
@pytest.mark.integration
class TestHTTPMCPWithHeaders:
    """Test HTTP MCP with custom headers."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_http_mcp_with_headers(self):
        """Test that HTTP MCP accepts custom headers (Context7 supports optional API key)."""
        client = Client()

        # Context7 doesn't require auth for basic usage, but supports it
        response = client.chat.completions.create(
            model="openai:gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": 'Use resolve-library-id to resolve "pandas".',
                }
            ],
            tools=[
                {
                    "type": "mcp",
                    "name": "context7",
                    "server_url": "https://mcp.context7.com/mcp",
                    "headers": {
                        "User-Agent": "aisuite-test"
                    },  # Custom header (optional)
                    "allowed_tools": ["resolve-library-id"],
                }
            ],
            max_turns=3,
        )

        # Verify it worked with headers
        content = response.choices[0].message.content.lower()
        assert any(
            keyword in content
            for keyword in ["pandas", "library", "id", "data"]
        ), f"Expected library info in response, got: {content}"
