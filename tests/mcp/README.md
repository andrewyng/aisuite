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
   EXA_API_KEY=your-key-here  # Optional: for Exa MCP tests
   ```
   Note: E2E tests mock LLM responses, so API keys won't be charged, but providers validate keys on initialization.

## Running Tests

### Run all MCP integration tests (mocked LLM, free):
```bash
pytest tests/mcp/ -v -m "integration and not llm"
```

### Run specific test file:
```bash
# MCPClient tests
pytest tests/mcp/test_client.py -v -m integration

# End-to-end tests (mocked LLM)
pytest tests/mcp/test_e2e.py -v -m integration

# Real LLM tests with stdio (⚠️ costs money, requires API keys)
pytest tests/mcp/test_llm_e2e.py -v -m llm

# Real LLM tests with HTTP (⚠️ costs money, requires API keys)
pytest tests/mcp/test_http_llm_e2e.py -v -m llm
```

### Run ONLY real LLM tests (⚠️ costs ~$0.50):
```bash
pytest tests/mcp/ -v -m llm
```

### Run ALL tests including LLM (⚠️ costs money):
```bash
pytest tests/mcp/ -v -m integration
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

### `test_e2e.py` - End-to-End Tests (Mocked LLM)
Tests the complete flow with `client.chat.completions.create()`:
- Config dict format
- Mixing MCP configs with Python functions
- Multiple MCP servers with prefixing
- Automatic cleanup
- Error handling
- **Note:** LLM responses are mocked, so no API calls are made

### `test_llm_e2e.py` - Real LLM End-to-End Tests with stdio (⚠️ Costs Money)
Tests with **actual API calls** to verify stdio MCP works with real LLMs:
- OpenAI GPT-4o reading files via stdio MCP
- Anthropic Claude reading files via stdio MCP
- Mixed tools (stdio MCP + Python functions)
- Multiple MCP servers with prefixing
- **Uses:** `@modelcontextprotocol/server-filesystem` (stdio)
- **Note:** These tests make real API calls (~$0.05-0.10 per test)
- **Marked with:** `@pytest.mark.llm`
- **Skipped if:** API keys not present in .env

### `test_http_llm_e2e.py` - Real LLM End-to-End Tests with HTTP (⚠️ Costs Money)
Tests with **actual API calls** to verify HTTP MCP works with real LLMs:
- OpenAI GPT-4o using HTTP MCP tools (Context7 and Exa)
- Anthropic Claude using HTTP MCP tools (Context7 and Exa)
- Mixed tools (HTTP MCP + Python functions)
- Config dict format with HTTP transport
- Custom headers support (including Authorization headers for Exa)
- **Uses:**
  - Context7 HTTP MCP server (`https://mcp.context7.com/mcp`)
    - Tools: `resolve-library-id`, `get-library-docs` (library documentation)
  - Exa HTTP MCP server (`https://mcp.exa.ai/mcp`)
    - Tools: `web_search_exa`, `get_code_context_exa` (web search and code context)
    - Requires: EXA_API_KEY in .env
- **Note:** These tests make real API calls (~$0.05-0.10 per test)
- **Marked with:** `@pytest.mark.llm`
- **Skipped if:** API keys not present in .env

### `conftest.py` - Test Fixtures
- `temp_test_dir` - Creates temp directory with test files
- `skip_if_no_npx` - Skips tests if npx not available

## What Gets Tested

### stdio Transport Tests
Use the **real** `@modelcontextprotocol/server-filesystem` MCP server from Anthropic, which:
- Provides file system access tools (read_file, write_file, list_directory, etc.)
- Is installed automatically via `npx -y @modelcontextprotocol/server-filesystem`
- Runs in a temporary test directory for isolation

### HTTP Transport Tests
Use **real** HTTP MCP servers:

1. **Context7** (`https://mcp.context7.com/mcp`):
   - Provides library documentation tools
   - Tools: `resolve-library-id`, `get-library-docs`
   - No installation required (hosted service)
   - No authentication required (optional API key for higher rate limits)

2. **Exa** (`https://mcp.exa.ai/mcp`):
   - Provides web search and code context tools
   - Tools: `web_search_exa`, `get_code_context_exa`, `deep_researcher`
   - No installation required (hosted service)
   - Requires: EXA_API_KEY (via Authorization header)

The tests verify:
1. ✅ Connection to real MCP servers (stdio and HTTP)
2. ✅ Tool discovery and schema parsing
3. ✅ Tool execution and result handling
4. ✅ Config dict → callable conversion
5. ✅ Tool filtering and prefixing
6. ✅ Integration with aisuite's tool system
7. ✅ Proper resource cleanup
8. ✅ Error handling
9. ✅ HTTP transport with headers and timeout

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
- Most tests use mocking for LLM API calls to avoid costs
- Real LLM tests are marked with `@pytest.mark.llm` and can be skipped
- Each test creates isolated temp directories for file operations
- MCP servers are started fresh for each test
- Cleanup is automatic via fixtures and context managers

## Test Markers

- `@pytest.mark.integration` - All MCP tests (includes both mocked and real LLM)
- `@pytest.mark.llm` - Real LLM tests only (makes actual API calls, costs money)

To run tests without LLM costs:
```bash
pytest tests/mcp/ -v -m "integration and not llm"
```

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
