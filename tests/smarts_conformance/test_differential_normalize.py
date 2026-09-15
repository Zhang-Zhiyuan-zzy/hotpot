"""Pure tests for differential-result normalization."""

from types import SimpleNamespace

import pytest

from .differential.cases import DifferentialCase
from .differential.normalize import (
    normalize_embeddings,
    target_atom_sets,
    unique_embeddings,
    unique_embeddings_from_hotpot_hits,
)
from .differential.runner import classify


pytestmark = pytest.mark.smarts_core


def test_raw_normalization_preserves_query_order_and_exact_duplicates():
    raw = ((2, 1), (1, 2), (2, 1))

    assert normalize_embeddings(raw) == ((1, 2), (2, 1), (2, 1))


def test_index_base_conversion_does_not_sort_inside_embedding():
    assert normalize_embeddings(((3, 1, 2),), index_base=1) == ((2, 0, 1),)


def test_unique_embeddings_and_target_sets_are_distinct_normalizations():
    raw = ((2, 1), (1, 2), (2, 1))

    assert unique_embeddings(raw) == ((1, 2), (2, 1))
    assert target_atom_sets(raw) == ((1, 2),)


def test_hotpot_unique_mode_uses_one_query_order_mapping_per_grouped_hit():
    hits = (
        SimpleNamespace(mappings=({0: 2, 1: 1}, {0: 1, 1: 2})),
        SimpleNamespace(mappings=({0: 4, 1: 3}, {0: 3, 1: 4})),
    )

    assert unique_embeddings_from_hotpot_hits(hits) == ((2, 1), (4, 3))


def test_cross_engine_classification_treats_native_unique_counts_as_diagnostic():
    case = DifferentialCase(
        "test.enumeration",
        "CC",
        "CCC",
        "safe_intersection",
        True,
        True,
        ("enumeration",),
        "test",
    )
    common = {
        "query_accepted": True,
        "target_accepted": True,
        "matched": True,
        "raw_embedding_count": 4,
        "unique_target_atom_set_count": 2,
    }
    engines = {
        "hotpot": {**common, "engine_unique_embedding_count": 2},
        "rdkit": {**common, "engine_unique_embedding_count": 4},
        "openbabel": {**common, "engine_unique_embedding_count": 3},
    }

    assert classify(case, engines) == "unanimous"
    engines["rdkit"] = {**engines["rdkit"], "raw_embedding_count": 3}
    assert classify(case, engines) == "enumeration_disagreement"
