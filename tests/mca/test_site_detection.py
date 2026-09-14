from pathlib import Path
from time import perf_counter

import pytest

import hotpot as hp
from mca.site_detection import find_nucleophilic_sites


def _sites(smiles):
    mol = hp.read_mol(smiles, fmt="smi")
    return tuple((site.atom_index, site.site_type) for site in find_nucleophilic_sites(mol))


@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        ("COC", ((1, "Ether"),)),
        ("CC(=O)C", ((2, "Ketone"), (1, "double_bond"))),
        ("CC(=O)N", ((2, "Amide"), (1, "double_bond"), (3, "atom_with_lone_pair"))),
        ("C=C[O-]", ((0, "Enolate"), (2, "anion_with_charge_minus1"), (1, "double_bond"))),
        ("CC=O", ((2, "Aldehyde"), (1, "double_bond"))),
        ("CC=N", ((2, "Imine"), (1, "double_bond"))),
        ("[NH-]C", ((0, "Nitranion"),)),
        ("[CH2-]C", ((0, "Carbanion"),)),
        ("C=[N+]([O-])[O-]", ((0, "Nitronate"), (2, "anion_with_charge_minus1"), (1, "atom_with_lone_pair"))),
        ("CC(=O)OC", ((2, "Ester"), (1, "double_bond"), (3, "atom_with_lone_pair"))),
        ("CC(=O)O", ((2, "Carboxylic acid"), (1, "double_bond"), (3, "atom_with_lone_pair"))),
        ("CCN", ((2, "Amine"),)),
        ("C=C=[N-]", ((0, "Cyanoalkyl/nitrile anion"), (2, "anion_with_charge_minus1"), (1, "double_bond"))),
        ("CC#N", ((2, "Nitrile"), (1, "triple_bond"))),
        ("[C-]#[N+]C", ((0, "Isonitrile"), (1, "atom_with_lone_pair"))),
        ("Oc1ccccc1", ((0, "Phenol"), (1, "double_bond"), (2, "double_bond"), (3, "double_bond"), (4, "double_bond"))),
        ("CO[Si](C)(C)C", ((1, "Silyl_ether"),)),
        ("n1ccccc1", ((0, "Pyridine_like_nitrogen"), (1, "double_bond"), (2, "double_bond"), (3, "double_bond"))),
        ("[O-]", ((0, "anion_with_charge_minus1"),)),
        ("C=C", ((0, "double_bond"),)),
        ("C=[N+](C)C", ((0, "double_bond_neighbouratom_with_charge_plus1"), (1, "atom_with_lone_pair"))),
        ("C#C", ((0, "triple_bond"),)),
        ("C#[S+]", ((0, "triple_bond_neighbouratom_with_charge_plus1"), (1, "atom_with_lone_pair"))),
        ("CS", ((1, "atom_with_lone_pair"),)),
    ),
)
def test_all_ordered_rules_against_static_golden(smiles, expected):
    assert _sites(smiles) == expected


def test_first_matching_rule_has_priority_over_generic_rules():
    sites = dict(_sites("C=[N+]([O-])[O-]"))

    assert sites[0] == "Nitronate"
    assert sites[2] == "anion_with_charge_minus1"


@pytest.mark.parametrize(
    ("smiles", "expected"),
    (
        ("c1ccccc1", ((0, "double_bond"),)),
        ("n1ccccc1", ((0, "Pyridine_like_nitrogen"), (1, "double_bond"), (2, "double_bond"), (3, "double_bond"))),
        ("NCCN", ((0, "Amine"),)),
        ("NCCCN(C)", ((0, "Amine"), (4, "Amine"))),
    ),
)
def test_exact_graph_automorphism_deduplication(smiles, expected):
    assert _sites(smiles) == expected


def test_charged_and_no_site_cases():
    assert _sites("[O-]") == ((0, "anion_with_charge_minus1"),)
    assert _sites("C") == ()


def test_disconnected_symmetric_components_do_not_enumerate_all_automorphisms():
    smiles = ".".join(["CCN"] * 8)
    start = perf_counter()

    assert _sites(smiles) == ((2, "Amine"),)
    assert perf_counter() - start < 2.0


def test_site_detection_has_no_rdkit_dependency():
    source = (
        Path(__file__).resolve().parents[2]
        / "hotpot"
        / "cheminfo"
        / "AImodels"
        / "mca"
        / "site_detection.py"
    ).read_text()

    assert "rdkit" not in source.lower()
