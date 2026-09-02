"""Guards the dashboard's imports against silent breakage.

`app.py` is only ever executed by Streamlit Cloud, so a stale import there —
a loader function renamed or removed — passes lint and unit tests and only
surfaces as an ImportError on the live site. Ruff cannot catch it: it checks
each file in isolation and never resolves a name against the module it is
imported from.

These tests parse `app.py` (rather than importing it — importing would run the
whole page, including its S3 reads) and verify every name it pulls from the
project actually exists.
"""
import ast
import importlib
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "src" / "dashboard" / "app.py"


def _project_imports() -> list[tuple[str, str]]:
    """Every (module, name) that app.py imports from the project's own package."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.")
        for alias in node.names
    ]


def test_app_file_is_parseable():
    assert _project_imports(), "no project imports found — check APP_PATH"


@pytest.mark.parametrize("module_name,symbol", _project_imports())
def test_imported_symbol_exists(module_name: str, symbol: str):
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), (
        f"app.py imports '{symbol}' from {module_name}, which does not define it — "
        "the dashboard would fail to start"
    )


def test_data_loader_imports_cleanly():
    """Import-time failures in the loader (bad config symbol, missing dep) would
    take the dashboard down before any callback runs."""
    importlib.import_module("src.dashboard.utils.data_loader")
