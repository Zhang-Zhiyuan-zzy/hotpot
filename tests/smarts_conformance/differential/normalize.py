"""Normalization helpers shared by SMARTS reference-engine adapters.

An embedding always preserves query-atom order. Target atom sets are a
separate representation and are the only tuples whose elements are sorted.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


Embedding = Tuple[int, ...]
TargetAtomSet = Tuple[int, ...]


def normalize_embeddings(
    embeddings: Iterable[Sequence[int]],
    *,
    index_base: int = 0,
) -> Tuple[Embedding, ...]:
    """Return sorted raw embeddings without discarding duplicate mappings.

    Indices inside an embedding remain in query-atom order. Only the outer
    collection is sorted to make the evidence deterministic.
    """

    normalized = [
        tuple(int(atom_index) - index_base for atom_index in embedding)
        for embedding in embeddings
    ]
    return tuple(sorted(normalized))


def unique_embeddings(
    embeddings: Iterable[Sequence[int]],
    *,
    index_base: int = 0,
) -> Tuple[Embedding, ...]:
    """Return distinct query-order embeddings in deterministic order."""

    return tuple(sorted(set(normalize_embeddings(embeddings, index_base=index_base))))


def target_atom_sets(
    embeddings: Iterable[Sequence[int]],
) -> Tuple[TargetAtomSet, ...]:
    """Collapse embeddings by the unordered set of target atoms they use."""

    normalized = {tuple(sorted(map(int, embedding))) for embedding in embeddings}
    return tuple(sorted(normalized))


def embeddings_from_hotpot_hits(hits: object) -> Tuple[Embedding, ...]:
    """Extract query-index ordered embeddings from ``hotpot.search.Hits``."""

    embeddings = []
    for hit in hits:
        for mapping in hit.mappings:
            embeddings.append(tuple(mapping[index] for index in sorted(mapping)))
    return normalize_embeddings(embeddings)


def unique_embeddings_from_hotpot_hits(hits: object) -> Tuple[Embedding, ...]:
    """Choose one query-order embedding from each target-atom-set Hit."""

    representatives = []
    for hit in hits:
        if hit.mappings:
            mapping = hit.mappings[0]
            representatives.append(tuple(mapping[index] for index in sorted(mapping)))
    return normalize_embeddings(representatives)
