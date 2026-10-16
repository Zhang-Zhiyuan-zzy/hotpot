"""Corpus protocol tests for scientific-state dry resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from .corpus import load_corpus


def test_load_corpus_preserves_order_ids_and_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "states.smi"
    encoded = b"# held-out records\nO water\n[Eu].N\n\n"
    source.write_bytes(encoded)

    corpus = load_corpus(source)

    assert corpus.source == source.resolve()
    assert corpus.source_sha256 == hashlib.sha256(encoded).hexdigest()
    assert tuple(
        (case.index, case.case_id, case.smiles) for case in corpus.cases
    ) == (
        (2, "water", "O"),
        (3, "case-000003", "[Eu].N"),
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
