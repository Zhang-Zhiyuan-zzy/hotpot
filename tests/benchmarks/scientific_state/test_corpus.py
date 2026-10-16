"""Corpus protocol tests for scientific-state dry resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from .corpus import (
    DryStateSourceKind,
    load_corpus,
    load_structure_registry,
    load_topology_manifest,
)


def test_load_corpus_preserves_order_ids_and_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "states.smi"
    encoded = b"# held-out records\nO water\n[Eu].N\n\n"
    source.write_bytes(encoded)

    corpus = load_corpus(source)

    assert corpus.source == source.resolve()
    assert corpus.source_sha256 == hashlib.sha256(encoded).hexdigest()
    assert tuple(
        (
            case.index,
            case.case_id,
            case.source_kind,
            case.source,
            case.format,
        )
        for case in corpus.cases
    ) == (
        (2, "water", DryStateSourceKind.SMILES, "O", "smi"),
        (
            3,
            "case-000003",
            DryStateSourceKind.SMILES,
            "[Eu].N",
            "smi",
        ),
    )


@pytest.mark.parametrize("content", ("\n# empty\n", "O same\nN same\n"))
def test_load_corpus_rejects_empty_and_duplicate_identity(
    tmp_path: Path,
    content: str,
) -> None:
    source = tmp_path / "invalid.smi"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_corpus(source)


def test_load_structure_registry_preserves_file_topology_provenance(
    tmp_path: Path,
) -> None:
    structure_root = tmp_path / "store"
    structure = structure_root / "structures/objects/a.mol2"
    structure.parent.mkdir(parents=True)
    structure.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    digest = hashlib.sha256(structure.read_bytes()).hexdigest()
    registry = tmp_path / "structures.csv"
    registry.write_text(
        "structure_id,format,store_path,canonical_file_sha256\n"
        f"structure-a,mol2,structures/objects/a.mol2,{digest}\n",
        encoding="utf-8",
    )

    corpus = load_structure_registry(registry, structure_root)

    case = corpus.cases[0]
    assert case.case_id == "structure-a"
    assert case.source_kind is DryStateSourceKind.FILE
    assert Path(case.source) == structure.resolve()
    assert case.format == "mol2"
    assert case.source_sha256 == digest


def test_load_topology_manifest_requires_explicit_digest_bound_bonds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "complex.smi"
    source.write_text("[Eu].N\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "topology.jsonl"
    manifest.write_text(
        "{"
        '"case_id":"eu-n",'
        '"format":"smi",'
        '"store_path":"complex.smi",'
        f'"source_sha256":"{digest}",'
        '"coordination_bonds":[[0,1]]'
        "}\n",
        encoding="utf-8",
    )

    corpus = load_topology_manifest(manifest, tmp_path)

    assert corpus.cases[0].coordination_bonds == ((0, 1),)
