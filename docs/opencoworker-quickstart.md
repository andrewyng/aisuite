# OpenCoworker quickstart

[OpenCoworker] is a desktop AI agent that can not only chat, but also do deep research and carry out tasks for you on your computer. It can read files (with permission) to gain context, read/send messages (slack, email, etc.), and create real deliverables like PDF reports, documents, spreadsheets. It also supports scheduled automations, such as providing you a daily news summary.

## 1. Install

| Platform | Download |
|---|---|
| macOS 13+ (Apple Silicon, M1 or later) | [OpenCoworker-macos-arm64.dmg](https://github.com/andrewyng/aisuite/releases/latest/download/OpenCoworker-macos-arm64.dmg) |
| Windows 10/11 (x64) | [OpenCoworker-windows-setup.exe](https://github.com/andrewyng/aisuite/releases/latest/download/OpenCoworker-windows-setup.exe) |

NOTE: On Windows, SmartScreen may warn on first run: choose **More info → Run anyway** (the build isn't Authenticode-signed yet).

## 2. Connect a model

Pick a provider and paste your API key — OpenAI, Anthropic (Claude), or Google (Gemini) — or select **Ollama** to run fully local with no key at all. Keys are stored on your machine and sent only to the provider you chose; there is no OpenCoworker server.

You can connect several providers and switch models per conversation.

## 3. Give it work

Grant access to a folder (read-only or read-write) and ask in plain language. Some first tasks that show what it can do:

- *"Organize this folder — group files by type and project, and rename the screenshots based on what's in them."*
- *"Read the five vendor proposals in this folder and build me a comparison spreadsheet: pricing, terms, deadlines, red flags."*
- *"Go through the receipts in this folder and produce a monthly expense report with totals by category."*

Everything the agent produces is saved to a scratch folder by default, but you can ask it to save it to a folder of your choice. The built-in viewer previews documents, spreadsheets, images, and PDFs without leaving the app, and risky actions (shell commands, file writes outside granted folders) ask for your approval first.

## 4. Automations

Ask for recurring work and OpenCoworker schedules it:

> *"Every weekday at 7am, search the web for news about my industry and write a one-page brief to my briefings folder."*

Scheduled automations run while the app is running (enable *Launch at login* in settings to keep them going).

## 5. Extend with MCP

OpenCoworker speaks the Model Context Protocol. Add servers in **Manage → Integrations** using the same `mcpServers` JSON format as Claude Desktop and Cursor — stdio or HTTP, with per-tool approval controls.

## Privacy

Your API keys, your conversations, and your files stay on your machine. OpenCoworker has no backend — model calls go directly from your computer to the provider you configured.

## For developers

OpenCoworker's source lives in this repository under [`platform/`](../platform/). It is built using aisuite (see the [Agents quickstart](agents-quickstart.md)) — a working reference for building your own agent harness.
