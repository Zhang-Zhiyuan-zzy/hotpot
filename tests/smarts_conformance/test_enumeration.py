from types import MappingProxyType
from itertools import permutations

import pytest

import hotpot as hp


pytestmark = pytest.mark.smarts_core


def embeddings(hits, query_size):
    return tuple(
        sorted(
            tuple(mapping[query_index] for query_index in range(query_size))
            for hit in hits
            for mapping in hit.mappings
        )
    )


@pytest.mark.smarts_smoke
def test_asymmetric_embeddings_preserve_query_atom_order():
    query = hp.Substructure.from_smarts("CO")
    hits = hp.Searcher(query).search(hp.read_mol("COC", "smi"))

    assert embeddings(hits, 2) == ((0, 1), (2, 1))
    assert [hit.atom_indices for hit in hits] == [
        frozenset((0, 1)),
        frozenset((1, 2)),
    ]


def test_automorphisms_are_retained_but_grouped_by_target_atom_set():
    query = hp.Substructure.from_smarts("CC")
    hits = hp.Searcher(query).search(hp.read_mol("CCC", "smi"))

    assert embeddings(hits, 2) == ((0, 1), (1, 0), (1, 2), (2, 1))
    assert [hit.atom_indices for hit in hits] == [
        frozenset((0, 1)),
        frozenset((1, 2)),
    ]
    assert all(
        isinstance(mapping, MappingProxyType)
        for hit in hits
        for mapping in hit.mappings
    )


def test_disconnected_query_keeps_raw_mapping_distinct_from_atom_set():
    query = hp.Substructure.from_smarts("C.C")
    hits = hp.Searcher(query).search(hp.read_mol("CC", "smi"))

    assert embeddings(hits, 2) == ((0, 1), (1, 0))
    assert [hit.atom_indices for hit in hits] == [frozenset((0, 1))]


def test_symmetric_cycle_has_six_raw_embeddings_and_one_atom_set():
    query = hp.Substructure.from_smarts("CCC")
    hits = hp.Searcher(query).search(hp.read_mol("C1CC1", "smi"))

    assert len(hits) == 1
    assert len(hits[0].mappings) == 6
    assert set(embeddings(hits, 3)) == set(permutations(range(3)))


def test_mapping_views_are_read_only_and_repeatable():
    query = hp.Substructure.from_smarts("CC")
    searcher = hp.Searcher(query)
    molecule = hp.read_mol("CCC", "smi")
    first = searcher.search(molecule)
    second = searcher.search(molecule)

    assert embeddings(first, 2) == embeddings(second, 2)
    with pytest.raises(TypeError):
        first[0].mappings[0][0] = 99


def test_legacy_node_set_iteration_remains_deterministic():
    hits = hp.Searcher(hp.Substructure.from_smarts("CC")).search(
        hp.read_mol("CCC", "smi")
    )
    hits.get_hit = False

    assert list(hits) == [frozenset((0, 1)), frozenset((1, 2))]
    assert hits[0] == frozenset((0, 1))
    assert frozenset((1, 2)) in hits
