import pytest

import hotpot as hp
from hotpot.cheminfo.search import QueryAtom, Searcher, Substructure


pytestmark = pytest.mark.smarts_core


def test_query_atom_matching_covers_empty_set_callable_missing_attr_and_type_error():
    atom = hp.read_mol("C", "smi").atoms[0]

    assert QueryAtom().match(atom)
    assert QueryAtom(atomic_number={6}).match(atom)
    assert QueryAtom(predicate=lambda candidate: candidate.atomic_number == 6).match(
        atom
    )
    assert not QueryAtom(attribute_not_present={1}).match(atom)
    with pytest.raises(TypeError):
        QueryAtom(atomic_number={6}).match(object())


def test_query_atom_and_bond_labels_follow_parent_query_indices():
    query = Substructure()
    first = query.add_atom(QueryAtom(atomic_number={6}))
    second = query.add_atom(QueryAtom(atomic_number={8}))
    bond = query.add_bond(first, second, bond_order={1})

    assert first.idx == 0
    assert second.label == "1"
    assert bond.label == "0-1"
    assert bond.a1idx == 0
    assert bond.a2idx == 1
    assert bond.sub is query


def test_substructure_from_molecule_and_smiles_match_the_source_graph():
    molecule = hp.read_mol("CO", "smi")
    from_molecule = Substructure.from_mol(molecule)
    from_smiles = Substructure.from_smiles("CO")

    assert repr(from_molecule) == "Substructure(2 Atoms, 1 Bonds)"
    assert len(Searcher(from_molecule).search(molecule)) == 1
    assert len(Searcher(from_smiles).search(molecule)) == 1


def test_substructure_from_molecule_accepts_index_and_atom_pair_overrides():
    molecule = hp.read_mol("CO", "smi")
    query = Substructure.from_mol(
        molecule,
        addition_atom_attr={0: {"formal_charge": {0}}},
        addition_bond_attr={(0, 1): {"predicate": lambda bond: not bond.is_aromatic}},
    )

    assert len(Searcher(query).search(molecule)) == 1


def test_searcher_rejects_graph_nodes_or_edges_without_hotpot_payloads():
    with pytest.raises(AttributeError, match="QueryAtom"):
        Searcher._node_match({}, {})
    with pytest.raises(AttributeError, match="QueryBond"):
        Searcher._edge_match({}, {})


def test_hits_membership_node_set_helper_and_mapped_atoms():
    molecule = hp.read_mol("CCC", "smi")
    hits = Searcher(Substructure.from_smarts("CC")).search(molecule)

    assert hits._get_nodes_set() == {frozenset((0, 1)), frozenset((1, 2))}
    assert hits[0] in hits
    assert hits[0].mapped_atoms(0) == tuple(
        molecule.atoms[index] for index in hits[0].mapped_atom_indices(0)
    )


def test_substructure_construct_graph_is_cached_and_mutations_invalidate_it():
    query = Substructure()
    first = query.add_atom(QueryAtom(atomic_number={6}))

    one_atom_graph = query.construct_graph()
    assert query.construct_graph() is one_atom_graph

    second = query.add_atom(QueryAtom(atomic_number={6}))
    two_atom_graph = query.construct_graph()
    assert two_atom_graph is not one_atom_graph
    assert set(two_atom_graph) == {0, 1}
    assert query.construct_graph() is two_atom_graph

    query.add_bond(first, second, bond_order={1})
    bonded_graph = query.construct_graph()
    assert bonded_graph is not two_atom_graph
    assert set(bonded_graph.edges) == {(0, 1)}
    assert query.construct_graph() is bonded_graph


def test_searcher_has_match_and_bounded_mapping_iterator_contract():
    molecule = hp.read_mol("CCC", "smi")
    carbon_searcher = Searcher(Substructure.from_smarts("C"))

    assert carbon_searcher.has_match(molecule)
    assert not Searcher(Substructure.from_smarts("O")).has_match(molecule)

    mappings = carbon_searcher.iter_mappings(molecule, max_matches=2)
    materialized = tuple(mappings)

    assert len(materialized) == 2
    assert all(tuple(mapping) == (0,) for mapping in materialized)
    assert len({mapping[0] for mapping in materialized}) == 2
    assert mappings.truncated
    with pytest.raises(TypeError):
        materialized[0][0] = 99


def test_search_max_matches_reports_truncation():
    molecule = hp.read_mol("CCC", "smi")
    searcher = Searcher(Substructure.from_smarts("C"))

    bounded_hits = searcher.search(molecule, max_matches=2)
    complete_hits = searcher.search(molecule)

    assert len(bounded_hits) == 2
    assert bounded_hits.truncated
    assert len(complete_hits) == 3
    assert not complete_hits.truncated


def test_hit_mapped_bonds_follow_each_mapping_not_the_induced_atom_subgraph():
    molecule = hp.read_mol("C1CC1", "smi")
    hit = Searcher(Substructure.from_smarts("CCC")).search(molecule)[0]

    induced_bonds = set(hit.induced_bonds)
    assert len(hit.mappings) == 6
    assert len(induced_bonds) == 3
    assert hit.bonds == hit.mapped_bonds()

    for mapping_index in range(len(hit.mappings)):
        mapped_bonds = set(hit.mapped_bonds(mapping_index))
        assert len(mapped_bonds) == 2
        assert mapped_bonds < induced_bonds
