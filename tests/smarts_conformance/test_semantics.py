import pytest

import hotpot as hp


pytestmark = pytest.mark.smarts_core


def matching_atom_sets(smiles, smarts):
    molecule = hp.read_mol(smiles, "smi")
    return {frozenset(hit.atom_indices) for hit in molecule.search_substructure(smarts)}


@pytest.mark.smarts_smoke
@pytest.mark.parametrize(
    ("smiles", "smarts", "expected"),
    [
        ("CCO", "[#8]", True),
        ("CCN", "[#8]", False),
        ("[NH4+]", "[N+]", True),
        ("N", "[N+]", False),
        ("CCO", "[O;H1]", True),
        ("CC=O", "[O;H1]", False),
        ("CCO", "[C;D1;X4;H3]", True),
        ("CC#N", "[N;X1;v3]", True),
        ("C1CCCCC1", "[C;R]", True),
        ("CCCC", "[C;R]", False),
        ("C1CCCC1", "[C;r5]", True),
        ("C1CCCCC1", "[C;r5]", False),
        ("CC(=O)N", "[#6;$([#6](=[#8])[#7])]", True),
        ("CCN", "[N;!$(N-C=O)]", True),
        ("CC(=O)N", "[N;!$(N-C=O)]", False),
        ("[Na+].[O-]C=O", "[Na+].[O-]", True),
    ],
)
def test_curated_atom_and_recursive_semantics(smiles, smarts, expected):
    assert bool(matching_atom_sets(smiles, smarts)) is expected


@pytest.mark.parametrize(
    ("smiles", "smarts", "expected"),
    [
        ("CC", "C-C", True),
        ("C=C", "C=C", True),
        ("C#N", "C#N", True),
        ("c1ccccc1", "c:c", True),
        ("CC=O", "C~O", True),
        ("CC=O", "C#,=O", True),
        ("CCO", "C=O", False),
        ("CC=O", "C#O", False),
        ("c1ccccc1", "c:c:c", True),
    ],
)
def test_unambiguous_bond_semantics(smiles, smarts, expected):
    assert bool(matching_atom_sets(smiles, smarts)) is expected


def test_logic_precedence_is_semantically_distinguishable():
    assert matching_atom_sets("C", "[C,N;D1]") == set()
    assert matching_atom_sets("CN", "[C,N;D1]") == {
        frozenset((0,)),
        frozenset((1,)),
    }
    assert matching_atom_sets("CO", "[C;N,D1]") == {frozenset((0,))}


def test_double_negation_and_commutative_operands_preserve_semantics():
    targets = ("C", "N", "O", "CN", "CCO")
    for target in targets:
        assert matching_atom_sets(target, "[!!C]") == matching_atom_sets(target, "[C]")
        assert matching_atom_sets(target, "[C,N]") == matching_atom_sets(
            target, "[N,C]"
        )
        assert matching_atom_sets(target, "[C;D1]") == matching_atom_sets(
            target, "[D1;C]"
        )


@pytest.mark.parametrize(
    ("smiles", "smarts", "expected"),
    [
        ("[Mn]", "[M]", True),
        ("C", "[M]", False),
        ("[Eu]", "[Ln]", True),
        ("[Am]", "[An]", True),
        ("[Na]", "[NP3]", True),
        ("[Mn]", "[NP3-5]", True),
        ("[Na]", "[NG1]", True),
        ("[Mn]", "[NG3-8]", True),
        ("[Eu]", "[An]", False),
    ],
)
def test_hotpot_extension_semantics(smiles, smarts, expected):
    assert bool(matching_atom_sets(smiles, smarts)) is expected
