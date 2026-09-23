"""Import-surface and dependency-direction tests for the geometry package."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Iterator, Tuple

import pytest

from hotpot.cheminfo import geometry


GEOMETRY_DIR = Path(__file__).parents[2] / "hotpot" / "cheminfo" / "geometry"
PACKAGE_MODULES = ("settings", "object", "relation", "convert")


def _imports_outside_type_checking(tree: ast.AST) -> Iterator[Tuple[int, str]]:
    """Yield imported module names that execute outside TYPE_CHECKING guards."""

    class RuntimeImportVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.type_checking_depth = 0
            self.imports = []

        def visit_If(self, node: ast.If) -> None:
            guarded = (
                isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            )
            if guarded:
                self.type_checking_depth += 1
            for statement in node.body:
                self.visit(statement)
            if guarded:
                self.type_checking_depth -= 1
            for statement in node.orelse:
                self.visit(statement)

        def visit_Import(self, node: ast.Import) -> None:
            if not self.type_checking_depth:
                self.imports.extend((node.lineno, alias.name) for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if not self.type_checking_depth:
                module_name = node.module or ""
                for alias in node.names:
                    imported_name = ".".join(
                        part for part in (module_name, alias.name) if part
                    )
                    self.imports.append((node.lineno, imported_name))

    visitor = RuntimeImportVisitor()
    visitor.visit(tree)
    return iter(visitor.imports)


def test_root_exports_exact_child_union_without_duplicates():
    child_modules = [
        importlib.import_module("hotpot.cheminfo.geometry.{}".format(name))
        for name in PACKAGE_MODULES
    ]
    child_exports = [name for module in child_modules for name in module.__all__]

    assert len(child_exports) == 64
    assert len(child_exports) == len(set(child_exports))
    assert tuple(geometry.__all__) == tuple(child_exports)

    for module in child_modules:
        for name in module.__all__:
            assert getattr(geometry, name) is getattr(module, name)


@pytest.mark.parametrize(
    "module_name",
    (
        "hotpot.cheminfo.geometry",
        "hotpot.cheminfo.core",
        "hotpot.cheminfo.forcefields",
    ),
)
def test_module_imports_in_fresh_subprocess(module_name):
    subprocess.run(
        [sys.executable, "-c", "import importlib; importlib.import_module({!r})".format(module_name)],
        cwd=str(Path(__file__).parents[2]),
        check=True,
    )


@pytest.mark.parametrize(
    "module_name, forbidden",
    (
        (
            "object",
            {
                "relation",
                "convert",
                "core",
                "forcefields",
                "graph",
                "networkx",
            },
        ),
        ("relation", {"convert", "core", "forcefields"}),
        ("convert", {"core", "forcefields"}),
    ),
)
def test_runtime_import_dependency_direction(module_name, forbidden):
    source_path = GEOMETRY_DIR / "{}.py".format(module_name)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    violations = [
        (line_number, imported)
        for line_number, imported in _imports_outside_type_checking(tree)
        if set(imported.split(".")) & forbidden
    ]

    assert violations == []
