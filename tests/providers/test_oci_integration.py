"""
Author: L. Saetta
Date last modified: 2026-07-29
License: MIT
Description: Opt-in live integration tests for OCI Generative AI chat completions.
"""

import asyncio
import os
from pathlib import Path

import pytest

import aisuite as ai

pytestmark = [pytest.mark.integration, pytest.mark.llm]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "openai.gpt-5.5"


def load_environment_file(dotenv_path: Path) -> None:
    """Load simple ``KEY=VALUE`` entries without overriding shell variables.

    Args:
        dotenv_path: Location of the local environment file.
    """
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@pytest.fixture(scope="module", autouse=True)
def enable_oci_integration_tests() -> None:
    """Load local credentials only after the caller explicitly opts in."""
    if os.getenv("AISUITE_RUN_OCI_INTEGRATION") != "1":
        pytest.skip(
            "Set AISUITE_RUN_OCI_INTEGRATION=1 to run live OCI integration tests."
        )

    dotenv_path = REPOSITORY_ROOT / ".env"
    if dotenv_path.is_file():
        load_environment_file(dotenv_path)

    missing_variables = [
        variable
        for variable in ("OCI_GENAI_REGION", "OCI_GENAI_API_KEY")
        if not os.getenv(variable)
    ]
    if missing_variables:
        pytest.skip("Missing OCI credentials: " + ", ".join(missing_variables))


@pytest.fixture(scope="module")
def model() -> str:
    """Return the OCI model name used by the integration tests.

    Returns:
        An OCI OpenAI-compatible model identifier without the provider prefix.
    """
    return os.getenv("OCI_GENAI_TEST_MODEL", DEFAULT_MODEL)


@pytest.fixture(scope="module")
def client() -> ai.Client:
    """Create an aisuite client configured from OCI environment variables.

    Returns:
        An aisuite client ready to call OCI.
    """
    return ai.Client()


def assert_completion(response) -> None:
    """Validate the minimum OpenAI-shaped completion contract.

    Args:
        response: Completion response returned by aisuite.
    """
    assert response.choices
    content = response.choices[0].message.content
    assert isinstance(content, str)
    assert content.strip()


def test_sync_chat_completion(client: ai.Client, model: str) -> None:
    """OCI returns a non-empty synchronous chat completion."""
    response = client.chat.completions.create(
        model=f"oci:{model}",
        messages=[
            {"role": "user", "content": "Reply with exactly: OCI integration OK"}
        ],
        max_completion_tokens=20,
    )

    assert_completion(response)


def test_async_chat_completion(client: ai.Client, model: str) -> None:
    """OCI returns a non-empty asynchronous chat completion."""
    response = asyncio.run(
        client.chat.completions.acreate(
            model=f"oci:{model}",
            messages=[{"role": "user", "content": "Reply with exactly: async OK"}],
            max_completion_tokens=20,
        )
    )

    assert_completion(response)


def test_streaming_chat_completion(client: ai.Client, model: str) -> None:
    """OCI yields at least one non-empty content chunk when streaming."""
    stream = client.chat.completions.create(
        model=f"oci:{model}",
        messages=[{"role": "user", "content": "Reply with exactly: stream OK"}],
        max_completion_tokens=20,
        stream=True,
    )

    content = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in stream
        if chunk.choices and chunk.choices[0].delta.content
    )
    assert content.strip()
