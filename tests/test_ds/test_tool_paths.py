from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hotpot.ds import _paths
from hotpot.ds.tools import analyze_structure_composition
from hotpot.ds.tools.electronic_state import assess_quality, build_database
from hotpot.ds.tools.oxidation import validate as validate_oxidation
from hotpot.ds.tools.spin import validate as validate_spin


TOOLS_ROOT = Path(analyze_structure_composition.__file__).parent


def test_tool_defaults_use_canonical_dataset_layout() -> None:
    assert validate_oxidation.SOURCE_ROOT == _paths.SOURCES_ROOT / "oxidation"
    assert validate_spin.SOURCE_ROOT == _paths.SOURCES_ROOT / "spin"
    assert assess_quality.DEFAULT_DB == (
        _paths.SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3"
    )
    assert assess_quality.DEFAULT_OUTPUT == (
        _paths.AUDITS_ROOT / "csd-electronic-state-quality.json"
    )
    assert build_database.DEFAULT_SCHEMA == (
        _paths.DATASET_ROOT / "schemas" / "csd-electronic-state.sql"
    )
    assert build_database.DEFAULT_OUTPUT == (
        _paths.SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3"
    )
    assert build_database.DEFAULT_AUDIT_OUTPUT.parent == _paths.AUDITS_ROOT
    assert build_database.DEFAULT_PROVENANCE_OUTPUT.parent == _paths.AUDITS_ROOT


def test_composition_source_tables_use_canonical_source_root(monkeypatch) -> None:
    paths = []

    def empty_table(path):
        paths.append(path)
        return []

    monkeypatch.setattr(analyze_structure_composition, "read_csv", empty_table)
    assert analyze_structure_composition.target_metals() == {}
    assert len(paths) == 6
    assert all(path.is_relative_to(_paths.SOURCES_ROOT) for path in paths)


def test_tools_contain_no_repository_level_storage_paths() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in TOOLS_ROOT.rglob("*.py")
    )
    assert "benchmark/" not in sources
    assert "datasets/" not in sources
    assert "structure_store" not in sources


@pytest.mark.parametrize(
    "module",
    [
        "hotpot.ds.tools.analyze_structure_composition",
        "hotpot.ds.tools.oxidation.validate",
        "hotpot.ds.tools.spin.validate",
        "hotpot.ds.tools.electronic_state.assess_quality",
        "hotpot.ds.tools.electronic_state.build_database",
    ],
)
def test_tool_module_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
