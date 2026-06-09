"""Model-provider registry — descriptors + a factory, mirroring the connector
(`connectors/descriptors.py`) and web-search (`web/providers.py`) patterns.

A `ProviderDescriptor` declares a provider's UI config `fields` (rendered dynamically by the
GUI, same `to_dict()` shape connectors use) and a `build(profile, secrets)` factory that returns
a `ProviderClient`. The `ProviderRouter` selects a descriptor by the `provider:` prefix of a
model string and builds (and caches) its client from the matching SecretStore profile.

Today: `openai` (the default) and `ollama` (OpenAI-compatible `/v1`). Azure (an `AzureOpenAI`
client) and Bedrock/Anthropic/Gemini (new wire formats) slot in as additional descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .base import ProviderClient
from .openai_provider import OpenAIProvider

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ProviderField:
    """One config input for a provider, rendered by the GUI (mirrors connectors' `Field`)."""

    key: str
    label: str
    secret: bool = False
    required: bool = True
    help: str = ""
    placeholder: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "secret": self.secret,
            "required": self.required,
            "help": self.help,
            "placeholder": self.placeholder,
        }


@dataclass(frozen=True)
class ProviderDescriptor:
    """A model provider: its UI fields + a factory that builds its `ProviderClient`."""

    name: str
    title: str
    needs_key: bool
    fields: list[ProviderField]
    build: Callable[[dict[str, Any], Any], ProviderClient] = field(repr=False)
    recommended_model: Optional[str] = None  # pre-filled in the UI; auto-added on configure

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "needs_key": self.needs_key,
            "fields": [f.to_dict() for f in self.fields],
            "recommended_model": self.recommended_model,
        }


def _normalize_ollama_url(url: Optional[str]) -> str:
    """Accept `http://host:11434` or `.../v1` and return an OpenAI-compatible base URL.

    Ollama serves its OpenAI-compatible API under `/v1`; the native API lives at the root, so we
    always target `<root>/v1`.
    """
    base = (url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if not base:
        base = DEFAULT_OLLAMA_URL
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def _build_openai(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # Key resolution stays in OpenAIProvider/resolve_api_key (explicit → env → SecretStore),
    # so we just hand it the SecretStore. `profile` is unused but kept for a uniform signature.
    return OpenAIProvider(secrets=secrets)


def _build_ollama(profile: dict[str, Any], secrets: Any) -> ProviderClient:
    # Ollama's OpenAI-compatible endpoint ignores the key but the SDK requires a non-empty
    # string, so we pass a placeholder. `base_url` comes from the stored profile (or the default).
    base_url = _normalize_ollama_url((profile or {}).get("base_url"))
    return OpenAIProvider(api_key="ollama", base_url=base_url)


DESCRIPTORS: list[ProviderDescriptor] = [
    ProviderDescriptor(
        name="openai",
        title="OpenAI",
        needs_key=True,
        fields=[
            ProviderField(
                "api_key",
                "OpenAI API key",
                secret=True,
                placeholder="sk-…",
                help="Stored locally (0600). Never sent to the model.",
            ),
        ],
        build=_build_openai,
    ),
    ProviderDescriptor(
        name="ollama",
        title="Ollama (local models)",
        needs_key=False,
        fields=[
            ProviderField(
                "base_url",
                "Ollama server URL",
                secret=False,
                required=False,
                placeholder=DEFAULT_OLLAMA_URL,
                help="Where `ollama serve` is listening. The OpenAI-compatible /v1 path is added automatically.",
            ),
        ],
        build=_build_ollama,
        # Reliable native tool-calling + strong coding quality (verified). Pull with
        # `ollama pull qwen3-coder:30b`.
        recommended_model="qwen3-coder:30b",
    ),
]

_BY_NAME = {d.name: d for d in DESCRIPTORS}


def provider_descriptors() -> list[ProviderDescriptor]:
    return list(DESCRIPTORS)


def provider_names() -> list[str]:
    return [d.name for d in DESCRIPTORS]


def get_descriptor(name: str) -> Optional[ProviderDescriptor]:
    return _BY_NAME.get(name)


def build_provider_client(name: str, profile: dict[str, Any], secrets: Any) -> ProviderClient:
    """Build a `ProviderClient` for `name` from its stored profile. Unknown → OpenAI default."""
    descriptor = _BY_NAME.get(name) or _BY_NAME["openai"]
    return descriptor.build(profile or {}, secrets)
