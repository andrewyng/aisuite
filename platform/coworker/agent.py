"""Engine assembly from an Agent (Code / Chat / …).

Wires the agent's base tools + permissions + AGENTS.md (workspace agents) + memory +
the skill catalog (progressive disclosure) + load_skill into a TurnEngine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .agents import Agent, AgentContext, code_agent
from .automation import scheduling_tools
from .config import load_config
from .connectors import (
    connector_list,
    load_settings,
    make_integration_tools,
    make_send_message_tool,
)
from .engine import Approver, TurnEngine
from .memory import MemoryStore, Scope, format_memories, memory_tools
from .permissions import Mode, PermissionEngine
from .project import load_agents_md
from .roots import RootDir, normalize_roots, render_context
from .providers import ProviderClient, ProviderRouter
from .secrets import SecretStore, state_dir
from .skills import SkillLoader, skill_catalog_text, skill_tools
from .tools import ToolRegistry
from .tools.directories import request_directory_tool
from .web import make_web_fetch_tool, make_web_search_tool
from .tools.shell import LocalExecutor
from .tools.todo import TodoList


def _enabled_connector_tools(secrets: SecretStore) -> tuple[set[str], set[str]]:
    connectors = {c["name"]: c for c in connector_list(secrets)}
    enabled_connectors = {
        name
        for name, c in connectors.items()
        if c.get("connected") and c.get("enabled")
    }
    enabled_tools = {
        tool["name"]
        for c in connectors.values()
        if c.get("name") in enabled_connectors
        for tool in c.get("tools", [])
        if tool.get("enabled")
    }
    return enabled_connectors, enabled_tools


def _skill_dirs(workspace: Optional[Path]) -> list[Path]:
    dirs = [state_dir() / "skills"]
    if workspace is not None:
        dirs.append(workspace / ".coworker" / "skills")
    return dirs


def build_engine(
    *,
    agent: Agent,
    workspace: Optional[str | Path] = None,
    model: str = "gpt-5.5",
    mode: Mode = Mode.INTERACTIVE,
    approver: Optional[Approver] = None,
    provider: Optional[ProviderClient] = None,
    allowed_commands: Optional[list[str]] = None,
    max_iterations: Optional[int] = None,
    model_settings: Optional[dict[str, Any]] = None,
    memory_store: Optional[MemoryStore] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    extra_tools: Optional[list[Any]] = None,
    secrets: Optional[SecretStore] = None,
    task_store: Optional[Any] = None,
    session_id: Optional[str] = None,
    audit_sink: Optional[Any] = None,
    roots: Optional[list] = None,
    directory_requester: Optional[Any] = None,
) -> TurnEngine:
    ws = Path(workspace).expanduser().resolve() if workspace else None
    if agent.needs_workspace and ws is None:
        raise ValueError(f"agent '{agent.name}' requires a workspace")

    # The session's directories. Explicit `roots` (orphan Cowork: scratch + added folders) wins;
    # otherwise the single workspace is the sole writable root. One shared, mutable list flows to
    # the file tools, the permission engine, and the context injector so add/remove is seen by all.
    if roots:
        root_list: list[RootDir] = normalize_roots(roots)
    elif ws is not None:
        root_list = [RootDir(path=ws, writable=True)]
    else:
        root_list = []

    config = load_config(ws)
    executor = (
        LocalExecutor(cwd=ws) if (agent.needs_workspace and ws is not None) else None
    )
    todo = TodoList()
    context = AgentContext(
        workspace=ws, executor=executor, todo=todo, roots=root_list or None
    )

    registry = ToolRegistry()
    registry.register_all(agent.build_tools(context))
    # MCP / connector tools (supplied by the manager) carry their own metadata + schema.
    if extra_tools:
        registry.register_all(extra_tools)
    # Connectors are Cowork-facing tools. MyHelper keeps the messaging reply path used by
    # inbound Telegram/Slack super-agent sessions.
    secrets = secrets or SecretStore()
    if agent.name in ("cowork", "myhelper") and any(
        s.enabled for s in load_settings(secrets).values()
    ):
        registry.register(make_send_message_tool(secrets))
    # Orphan surfaces can ask the user mid-task for access to another folder (read-only/-write).
    if agent.name in ("cowork", "myhelper") and root_list:
        registry.register(request_directory_tool())
    if agent.name == "cowork":
        enabled_connectors, enabled_tools = _enabled_connector_tools(secrets)
        registry.register_all(
            make_integration_tools(
                secrets,
                enabled_connectors=enabled_connectors,
                enabled_tools=enabled_tools,
                roots=root_list or None,
            )
        )
    # Web search + fetch: research tools for every agent (keyless DuckDuckGo default).
    registry.register(make_web_search_tool(secrets))
    registry.register(make_web_fetch_tool())
    # Scheduling: Cowork + MyHelper can set up scheduled tasks (origin = this session).
    if (
        task_store is not None
        and ws is not None
        and agent.name in ("cowork", "myhelper")
    ):
        origin = {
            "surface": agent.name,
            "session_id": session_id or "",
            "workspace": str(ws),
            "agent": agent.name,
        }
        registry.register_all(
            scheduling_tools(task_store, origin=origin, default_workspace=str(ws))
        )

    instructions = agent.system_prompt
    if ws is not None:
        conventions = load_agents_md(ws)
        if conventions:
            instructions = f"{instructions}\n\n{conventions}"

    if memory_store is not None:
        registry.register_all(
            memory_tools(memory_store, workspace=str(ws) if ws else None)
        )
        remembered = memory_store.list(scope=Scope.GLOBAL)
        if ws is not None:
            remembered += memory_store.list(scope=Scope.WORKSPACE, workspace=str(ws))
        block = format_memories(remembered)
        if block:
            instructions = f"{instructions}\n\n{block}"

    skill_loader = SkillLoader(_skill_dirs(ws))
    registry.register_all(skill_tools(skill_loader))
    catalog = skill_catalog_text(skill_loader)
    if catalog:
        instructions = f"{instructions}\n\n{catalog}"

    permissions = PermissionEngine(
        workspace_root=ws or (root_list[0].path if root_list else Path.cwd()),
        mode=mode,
        allowed_commands=allowed_commands or config.allowed_commands,
        auto_allow_tools=set(config.auto_allow),
        roots=root_list or None,
    )
    # Tell the agent, each turn, which directories it has and their access (orphan Cowork can gain
    # folders mid-session) — appended to the latest user message since mid-thread system messages
    # aren't reliable across providers. Multi-dir surfaces (Cowork/MyHelper) only.
    context_provider = (
        (lambda: render_context(root_list))
        if root_list and agent.name in ("cowork", "myhelper")
        else None
    )
    # Route by the model's `provider:` prefix (OpenAI default, Ollama, …). The manager normally
    # passes its shared router; this fallback covers the TUI / direct build_engine() callers.
    provider = provider or ProviderRouter(secrets, default_provider="openai")

    engine = TurnEngine(
        provider=provider,
        registry=registry,
        permissions=permissions,
        model=model,
        instructions=instructions,
        approver=approver,
        max_iterations=(
            max_iterations if max_iterations is not None else config.max_iterations
        ),
        model_settings=model_settings,
        messages=messages,
        audit_sink=audit_sink,
        context_provider=context_provider,
        directory_requester=directory_requester,
    )
    engine.executor = executor  # type: ignore[attr-defined]
    engine.todo = todo  # type: ignore[attr-defined]
    engine.agent_name = agent.name  # type: ignore[attr-defined]
    engine.roots = root_list  # type: ignore[attr-defined]  # shared list; Slice C mutates in place
    engine.audit_context = {
        "session_id": session_id or "",
        "agent": agent.name,
        "workspace": str(ws) if ws else "",
    }
    engine.skill_loader = skill_loader  # type: ignore[attr-defined]
    return engine


def build_code_engine(**kwargs: Any) -> TurnEngine:
    """Back-compat shim: build the Code agent's engine."""
    return build_engine(agent=code_agent(), **kwargs)
