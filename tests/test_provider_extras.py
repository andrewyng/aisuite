"""Each provider extra must declare the third-party packages its provider imports.

Regression test for `huggingface = []`: the extra existed but was empty, so
`pip install 'aisuite[huggingface]'` installed neither `huggingface_hub` nor
`requests`, and the first call to a `huggingface:` model raised

    ImportError: Could not import module aisuite.providers.huggingface_provider:
    No module named 'requests'

CI did not catch it because both packages arrive transitively through the dev
dependencies (sentence-transformers, datasets, chromadb), so they are always
present in a development checkout and never in a user's install.
"""

import ast
import sys
from pathlib import Path

import pytest

try:  # tomllib is stdlib on 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = pytest.importorskip(
        "tomli", reason="needs tomllib (3.11+) or tomli to read pyproject.toml"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PROVIDERS = REPO_ROOT / "aisuite" / "providers"

# Import name -> distribution name, where the two differ. Both of these are
# guaranteed by the distribution already listed in the extra: boto3 depends on
# botocore, and the cerebras_cloud_sdk distribution ships the `cerebras` package.
IMPORT_TO_DISTRIBUTION = {
    "botocore": "boto3",
    "cerebras": "cerebras_cloud_sdk",
}


def _normalize(name: str) -> str:
    return name.replace("-", "_").lower()


def _load_poetry_config() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["tool"]["poetry"]


POETRY = _load_poetry_config()
EXTRAS = POETRY["extras"]
REQUIRED_DEPENDENCIES = {
    _normalize(name)
    for name, spec in POETRY["dependencies"].items()
    if name != "python" and not (isinstance(spec, dict) and spec.get("optional"))
}

# Extras named after a provider module are the ones this contract applies to.
# `all`, `mcp` and `postgres` are not providers.
PROVIDER_EXTRAS = sorted(
    extra for extra in EXTRAS if (PROVIDERS / f"{extra}_provider.py").is_file()
)


def _module_level_third_party_imports(path: Path) -> set[str]:
    """Top-level import names in `path`, excluding stdlib and aisuite itself.

    Only module-scope imports count: an import inside a function or a
    `TYPE_CHECKING` block does not run on a plain install.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in sys.stdlib_module_names and m != "aisuite"}


def test_provider_extras_are_discovered():
    """Guard against the parametrization below silently covering nothing."""
    assert "huggingface" in PROVIDER_EXTRAS
    assert len(PROVIDER_EXTRAS) > 5


@pytest.mark.parametrize("extra", PROVIDER_EXTRAS)
def test_extra_declares_its_provider_imports(extra):
    provider = PROVIDERS / f"{extra}_provider.py"
    declared = {_normalize(pkg) for pkg in EXTRAS[extra]} | REQUIRED_DEPENDENCIES

    undeclared = sorted(
        imported
        for imported in _module_level_third_party_imports(provider)
        if _normalize(IMPORT_TO_DISTRIBUTION.get(imported, imported)) not in declared
    )

    assert not undeclared, (
        f"{provider.name} imports {undeclared} at module scope, but "
        f"`{extra} = {EXTRAS[extra]}` does not install them. "
        f"`pip install 'aisuite[{extra}]'` would fail at first use."
    )
