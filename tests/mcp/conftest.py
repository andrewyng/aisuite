"""
Pytest fixtures for MCP integration tests.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, that's okay
    pass


@pytest.fixture
def temp_test_dir():
    """
    Create a temporary directory with test files for filesystem MCP server.

    This fixture creates a temp directory with sample files that can be used
    to test the Anthropic filesystem MCP server.

    Yields:
        str: Real path to the temporary test directory (resolves symlinks)

    Example:
        >>> def test_mcp(temp_test_dir):
        ...     mcp = MCPClient(
        ...         command="npx",
        ...         args=["-y", "@modelcontextprotocol/server-filesystem", temp_test_dir]
        ...     )
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve real path to handle symlinks (e.g., /var -> /private/var on macOS)
        real_tmpdir = os.path.realpath(tmpdir)

        # Create test files
        test_file = Path(real_tmpdir) / "test.txt"
        test_file.write_text("Hello from MCP test!")

        readme = Path(real_tmpdir) / "README.md"
        readme.write_text("# Test Directory\n\nThis is a test README file.")

        data_file = Path(real_tmpdir) / "data.json"
        data_file.write_text('{"key": "value", "number": 42}')

        # Create a subdirectory
        subdir = Path(real_tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested file content")

        yield real_tmpdir


@pytest.fixture
def skip_if_no_npx():
    """
    Skip test if npx is not available.

    This fixture checks if npx (Node.js package executor) is installed,
    which is required to run the Anthropic filesystem MCP server.

    Raises:
        pytest.skip: If npx is not found in PATH
    """
    import shutil

    if not shutil.which("npx"):
        pytest.skip(
            "npx not found. Install Node.js to run MCP integration tests. "
            "See: https://nodejs.org/"
        )
