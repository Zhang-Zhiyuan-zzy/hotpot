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
