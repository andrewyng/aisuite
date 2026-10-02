"""Regression tests for #369 (bug 2): a missing `mcp` extra must raise an
actionable ImportError, not NameError.

`aisuite/client.py` imported `is_mcp_config` and `MCPClient` in one try/except.
Only `MCPClient` needs the optional `mcp` package, but its failure abandoned the
whole block, so `is_mcp_config` never bound. The `if not MCP_AVAILABLE` branch
then called it and raised `NameError: name 'is_mcp_config' is not defined`
instead of telling the user to install the extra.

The pre-existing `test_mcp_not_installed_raises_error` could not catch this: it
patches `aisuite.client.MCP_AVAILABLE` to False while `is_mcp_config` stays
bound (the test env has `mcp` installed), so it exercises the branch with the
name available. These tests simulate the import failure itself, so they are
meaningful whether or not `mcp` is present.
"""

import builtins
import importlib
import sys

import pytest


def _reload_without_mcp(monkeypatch):
    """Reimport aisuite.mcp + aisuite.client with `import mcp` failing."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only block the ABSOLUTE `import mcp`. Relative imports inside the
        # package (`from .mcp.config import ...`) arrive here as name="mcp.config"
        # with level=1 and must keep working — that is the whole point of the fix.
        if level == 0 and (name == "mcp" or name.startswith("mcp.")):
            raise ImportError("No module named 'mcp'")
        return real_import(name, globals, locals, fromlist, level)

    for mod in [m for m in list(sys.modules) if m == "mcp" or m.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    # Drop the cached MCP client module so client.py re-executes its import.
    monkeypatch.delitem(sys.modules, "aisuite.mcp.client", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    mcp_pkg = importlib.import_module("aisuite.mcp")
    importlib.reload(mcp_pkg)
    return importlib.reload(importlib.import_module("aisuite.client"))


@pytest.fixture
def client_without_mcp(monkeypatch):
    module = _reload_without_mcp(monkeypatch)
    yield module
    # Restore the real modules for the rest of the session.
    monkeypatch.undo()
    importlib.reload(importlib.import_module("aisuite.mcp"))
    importlib.reload(importlib.import_module("aisuite.client"))


MCP_TOOL = {"type": "mcp", "name": "filesystem", "command": "npx", "args": ["server"]}


def test_is_mcp_config_binds_without_the_mcp_extra(client_without_mcp):
    """The dependency-free helper must survive a missing optional dependency."""
    assert client_without_mcp.MCP_AVAILABLE is False
    assert hasattr(client_without_mcp, "is_mcp_config"), (
        "is_mcp_config must stay bound when the mcp extra is absent, "
        "otherwise the not-available branch raises NameError"
    )
    assert client_without_mcp.is_mcp_config(MCP_TOOL) is True


def test_mcp_config_without_extra_raises_actionable_importerror(client_without_mcp):
    completions = client_without_mcp.Completions.__new__(client_without_mcp.Completions)

    with pytest.raises(ImportError, match=r"pip install"):
        completions._process_mcp_configs([MCP_TOOL])


def test_non_mcp_tools_are_untouched_without_the_extra(client_without_mcp):
    """A caller with no MCP configs must not be penalised for the missing extra."""
    completions = client_without_mcp.Completions.__new__(client_without_mcp.Completions)

    def my_tool():
        pass

    processed, mcp_clients = completions._process_mcp_configs([my_tool])
    assert processed == [my_tool]
    assert mcp_clients == []
