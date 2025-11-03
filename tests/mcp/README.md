# MCP Integration Tests

This directory contains integration tests for aisuite's MCP (Model Context Protocol) support.

## Prerequisites

To run these tests, you need:

1. **Node.js and npx** - Required to run the Anthropic filesystem MCP server
   - Install from: https://nodejs.org/
   - Verify with: `npx --version`

2. **Python test dependencies**:
   ```bash
   pip install pytest pytest-asyncio python-dotenv
   ```

3. **MCP package** (should already be installed if you have aisuite[mcp]):
   ```bash
   pip install 'aisuite[mcp]'
   ```

4. **Environment variables** (for e2e tests that mock LLM calls):
   Create a `.env` file in the project root with your API keys:
   ```bash
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   ```
   Note: E2E tests mock LLM responses, so API keys won't be charged, but providers validate keys on initialization.

## Running Tests

### Run all MCP integration tests:
```bash
pytest tests/mcp/ -v -m integration
```

### Run specific test file:
```bash
# MCPClient tests
pytest tests/mcp/test_client.py -v -m integration

# End-to-end tests
pytest tests/mcp/test_e2e.py -v -m integration
```

### Run a specific test:
```bash
pytest tests/mcp/test_client.py::TestMCPClientConnection::test_connect_to_filesystem_server -v
```

### Skip integration tests (if no Node.js):
```bash
pytest tests/mcp/ -v -m "not integration"
```

## Test Structure

### `test_client.py` - MCPClient Integration Tests
Tests the `MCPClient` class with a real MCP server:
- Connection to Anthropic filesystem server
- Listing tools
- Calling tools
- Tool filtering (`allowed_tools`)
- Tool prefixing (`use_tool_prefix`)
- `from_config()` method
- Context manager support

### `test_e2e.py` - End-to-End Tests
Tests the complete flow with `client.chat.completions.create()`:
- Config dict format
- Mixing MCP configs with Python functions
- Multiple MCP servers with prefixing
- Automatic cleanup
- Error handling

### `conftest.py` - Test Fixtures
- `temp_test_dir` - Creates temp directory with test files
- `skip_if_no_npx` - Skips tests if npx not available

## What Gets Tested

These tests use the **real** `@modelcontextprotocol/server-filesystem` MCP server from Anthropic, which:
- Provides file system access tools (read_file, write_file, list_directory, etc.)
- Is installed automatically via `npx -y @modelcontextprotocol/server-filesystem`
- Runs in a temporary test directory for isolation

The tests verify:
1. ✅ Connection to real MCP servers
2. ✅ Tool discovery and schema parsing
3. ✅ Tool execution and result handling
4. ✅ Config dict → callable conversion
5. ✅ Tool filtering and prefixing
6. ✅ Integration with aisuite's tool system
7. ✅ Proper resource cleanup
8. ✅ Error handling

## CI/CD

### GitHub Actions
If running in CI without Node.js:
```yaml
- name: Run tests
  run: pytest tests/mcp/ -v -m "not integration"
```

With Node.js:
```yaml
- name: Setup Node.js
  uses: actions/setup-node@v3
  with:
    node-version: '18'

- name: Run integration tests
  run: pytest tests/mcp/ -v -m integration
```

## Notes

- Tests are marked with `@pytest.mark.integration` to allow selective running
- Tests use mocking for LLM API calls to avoid costs
- Each test creates isolated temp directories for file operations
- MCP servers are started fresh for each test
- Cleanup is automatic via fixtures and context managers

## Troubleshooting

**Error: "npx not found"**
- Install Node.js from https://nodejs.org/

**Error: "MCP package not installed"**
- Run: `pip install 'aisuite[mcp]'`

**Tests hang or timeout**
- Check Node.js/npx is working: `npx --version`
- Check MCP server can be installed: `npx -y @modelcontextprotocol/server-filesystem --help`

**Import errors**
- Make sure you're running from the project root
- Install test dependencies: `pip install pytest pytest-asyncio`
