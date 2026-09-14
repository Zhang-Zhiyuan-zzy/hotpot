from types import MappingProxyType

import pytest

import hotpot as hp
from hotpot.cheminfo.search.search import Hits, QueryAtom, Searcher, Substructure


def carbon_pair_substructure():
    sub = Substructure()
    sub.add_atom(QueryAtom(atomic_number={6}, map_number=1))
    sub.add_atom(QueryAtom(atomic_number={6}, map_number=2))
    sub.add_bond(0, 1)
    return sub


class SingleUseMatcher:
    def __init__(self, mappings):
        self.mappings = mappings
        self.calls = 0

    def subgraph_monomorphisms_iter(self):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("GraphMatcher iterator was consumed more than once")
        yield from self.mappings


def test_query_atom_map_number_is_metadata_not_match_constraint():
    atom = QueryAtom(atomic_number={6}, map_number=7)

    assert atom.map_number == 7
    assert atom.kwargs == {"atomic_number": {6}}
    with pytest.raises(AttributeError):
        atom.map_number = 8


def test_hits_preserve_all_automorphisms_and_mapped_anchors():
    mol = hp.read_mol("CCC")
    hits = Searcher(carbon_pair_substructure()).search(mol)

    assert [hit.atom_indices for hit in hits] == [frozenset({0, 1}), frozenset({1, 2})]
    for hit in hits:
        assert len(hit.mappings) == 2
        assert all(isinstance(mapping, MappingProxyType) for mapping in hit.mappings)
        assert hit.mapped_atom_indices(0) == tuple(sorted(hit.atom_indices))
        assert hit.mapped_atom_indices(1) == tuple(sorted(hit.atom_indices))
        assert hit.mapped_atoms(0) == tuple(mol.atoms[idx] for idx in sorted(hit.atom_indices))

    assert tuple(tuple(dict(mapping).items()) for mapping in hits[0].mappings) == (
        ((0, 0), (1, 1)),
        ((0, 1), (1, 0)),
    )
    with pytest.raises(TypeError):
        hits[0].mappings[0][0] = 2


def test_hits_materialize_matcher_once_and_keep_legacy_node_sets():
    mol = hp.read_mol("CCC")
    sub = carbon_pair_substructure()
    matcher = SingleUseMatcher([
        {2: 0, 1: 1},
        {1: 0, 2: 1},
        {1: 0, 2: 1},
        {1: 0, 0: 1},
        {0: 0, 1: 1},
    ])
    hits = Hits(sub, mol, matcher, get_hit=False)

    expected = [frozenset({0, 1}), frozenset({1, 2})]
    assert matcher.calls == 1
    assert list(hits) == expected
    assert len(hits) == 2
    assert bool(hits)
    assert hits[0] == expected[0]
    assert expected[1] in hits

    materialized_hits = hits.hits
    assert matcher.calls == 1
    assert [hit.atom_indices for hit in materialized_hits] == expected
    assert tuple(tuple(dict(mapping).items()) for mapping in materialized_hits[1].mappings) == (
        ((0, 1), (1, 2)),
        ((0, 2), (1, 1)),
    )


def test_empty_hits_are_deterministic_and_do_not_reconsume_matcher():
    mol = hp.read_mol("O")
    matcher = SingleUseMatcher([])
    hits = Hits(carbon_pair_substructure(), mol, matcher)

    assert not hits
    assert list(hits) == []
    assert hits.hits == []
    assert matcher.calls == 1
