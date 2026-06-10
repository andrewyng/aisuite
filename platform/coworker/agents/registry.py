"""Built-in agent registry."""

from __future__ import annotations

from .base import Agent
from .chat import chat_agent
from .code import code_agent
from .cowork import cowork_agent
from .myhelper import myhelper_agent

_BUILDERS = {
    "code": code_agent,
    "chat": chat_agent,
    "cowork": cowork_agent,
    "myhelper": myhelper_agent,
}


def get_agent(name: str) -> Agent:
    builder = _BUILDERS.get(name) or _BUILDERS["code"]
    return builder()


def list_agents() -> list[dict]:
    # Session surfaces shown in the sidebar. MyHelper is the persistent helper (its own
    # always-on surface via /ws/superagent), not an ad-hoc session, so it's not listed here.
    return [
        {"name": "code", "title": "Code", "needs_workspace": True},
        {"name": "chat", "title": "Chat", "needs_workspace": False},
        {"name": "cowork", "title": "Coworker", "needs_workspace": True},
    ]
