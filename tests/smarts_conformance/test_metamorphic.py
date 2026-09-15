import pytest

import hotpot as hp


pytestmark = pytest.mark.smarts_core


def atom_sets(smiles, smarts):
    return tuple(
        hit.atom_indices
        for hit in hp.read_mol(smiles, "smi").search_substructure(smarts)
    )


@pytest.mark.parametrize("smiles", ["C", "CN", "CCO", "C1CCCCC1", "c1ccncc1"])
def test_repeat_execution_is_deterministic(smiles):
    query = "[C,N;!R]"
    assert atom_sets(smiles, query) == atom_sets(smiles, query)


@pytest.mark.parametrize("smiles", ["C", "N", "O", "CN", "CCO", "c1ccncc1"])
def test_or_operand_permutation_preserves_results(smiles):
    assert atom_sets(smiles, "[C,N,O]") == atom_sets(smiles, "[O,C,N]")


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "CC=O"])
def test_commutative_and_permutation_preserves_results(smiles):
    assert atom_sets(smiles, "[C;D1;H3]") == atom_sets(smiles, "[H3;C;D1]")


@pytest.mark.parametrize("smiles", ["C", "N", "CCO", "c1ccccc1"])
def test_double_negation_preserves_results(smiles):
    assert atom_sets(smiles, "[!!#6]") == atom_sets(smiles, "[#6]")


@pytest.mark.parametrize(
    ("left", "right", "query"),
    [
        ("CCO", "OCC", "CO"),
        ("CC(=O)N", "NC(=O)C", "C(=O)N"),
        ("c1ccncc1", "n1ccccc1", "[nH0]"),
    ],
)
def test_equivalent_target_order_preserves_existence(left, right, query):
    assert bool(atom_sets(left, query)) == bool(atom_sets(right, query))


def _embeddings_as_atom_ids(molecule, smarts):
    query = hp.Substructure.from_smarts(smarts)
    hits = hp.Searcher(query).search(molecule)
    return tuple(
        sorted(
            tuple(
                molecule.atoms[mapping[index]].id
                for index in range(len(query.query_atoms))
            )
            for hit in hits
            for mapping in hit.mappings
        )
    )


def test_atom_renumbering_preserves_embeddings_after_stable_identity_mapping():
    original = hp.read_mol("CCOCN", "smi")
    for stable_id, atom in enumerate(original.atoms, start=100):
        atom.id = stable_id
    renumbered = original.copy()
    renumbered._atoms = list(reversed(renumbered._atoms))
    renumbered._update_graph()

    assert [atom.id for atom in original.atoms] != [
        atom.id for atom in renumbered.atoms
    ]
    assert (
        _embeddings_as_atom_ids(original, "CO")
        == _embeddings_as_atom_ids(renumbered, "CO")
        == ((101, 102), (103, 102))
    )
