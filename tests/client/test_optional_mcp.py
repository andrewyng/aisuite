import subprocess
import sys
import textwrap


def test_tool_processing_without_mcp_extra():
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def import_without_mcp(name, *args, **kwargs):
            if name == "mcp" or name.startswith("mcp."):
                raise ImportError("No module named 'mcp'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_mcp

        from aisuite.client import Client, Completions, MCP_AVAILABLE

        tools = [{"type": "function", "function": {"name": "ping"}}]
        completions = Completions(Client())
        processed_tools, mcp_clients = completions._process_mcp_configs(tools)

        assert MCP_AVAILABLE is False
        assert processed_tools == tools
        assert mcp_clients == []

        try:
            completions._process_mcp_configs([{"type": "mcp", "name": "example"}])
        except ImportError as exc:
            assert "pip install 'aisuite[mcp]'" in str(exc)
        else:
            raise AssertionError("MCP config should require the optional MCP package")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
