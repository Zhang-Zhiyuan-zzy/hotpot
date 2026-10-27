"""Keep every Markdown ``Verified by`` reference resolvable."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = re.compile(
    r"(?P<path>tests/[A-Za-z0-9_./-]+\.py)::(?P<name>test_[A-Za-z0-9_]+)"
)
DOCUMENTS = (
    ROOT / "README.md",
    *sorted((ROOT / "doc").rglob("*.md")),
    *sorted((ROOT / "hotpot").rglob("*.md")),
)


def _references() -> tuple[tuple[Path, str, Path], ...]:
    references = []
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for comment in re.findall(r"<!--\s*Verified by(.*?)-->", text, re.DOTALL):
            matches = tuple(REFERENCE.finditer(comment))
            assert matches, f"unparseable Verified by annotation in {document}"
            references.extend(
                (ROOT / match.group("path"), match.group("name"), document)
                for match in matches
            )
    return tuple(references)


@pytest.mark.parametrize(
    ("test_path", "test_name", "document"),
    _references(),
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_verified_documentation_reference_resolves_to_a_test_function(
    test_path: Path,
    test_name: str,
    document: Path,
) -> None:
    assert test_path.is_file(), f"{document} references missing {test_path}"
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert test_name in functions, f"{document} references missing {test_name}"
