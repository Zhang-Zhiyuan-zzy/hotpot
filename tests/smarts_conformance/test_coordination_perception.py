"""L3 snapshots for Open Babel-backed coordination target preparation."""

import pytest

import hotpot as hp
from hotpot.cheminfo.core import BondKind

from .coordination_cases import (
    COORDINATION_FIXTURES,
    PERCEPTION_CASES,
    REPOSITORY_INPUTS,
    matching_atom_sets,
    matching_indices,
)


pytestmark = pytest.mark.smarts_core


@pytest.mark.parametrize("case", PERCEPTION_CASES, ids=lambda case: case.name)
def test_coordination_fixture_perception_profile(case):
    molecule = hp.read_mol(COORDINATION_FIXTURES / case.filename)
    donor = molecule.atoms[0]
    metal = molecule.atoms[4]
    coordinate_bond = molecule.bond(donor.idx, metal.idx)

    assert (len(molecule.atoms), len(molecule.bonds)) == (5, 4)
    assert (donor.symbol, metal.symbol) == ("N", "Cu")
    assert coordinate_bond.is_metal_ligand_bond
    assert coordinate_bond.bond_order == case.bond_order
    assert coordinate_bond.bond_kind is case.bond_kind
    assert coordinate_bond.bond_source == "openbabel"
    assert coordinate_bond.bond_source_metadata["raw_bond_order"] == case.bond_order
    assert len(donor.neighbours) == case.donor_degree
    assert donor.implicit_hydrogens == case.donor_implicit_hydrogens
    assert len(donor.neighbours) + donor.implicit_hydrogens == case.donor_connectivity
    assert donor.sum_bond_orders + donor.implicit_hydrogens == case.donor_valence
    assert matching_indices(
        molecule,
        (f"[N;D{case.donor_degree};X{case.donor_connectivity};v{case.donor_valence}]"),
    ) == {donor.idx}
    assert matching_atom_sets(molecule, "[N]~[M]") == {
        frozenset((donor.idx, metal.idx))
    }


@pytest.mark.parametrize(
    ("zero_filename", "source_bond_type"),
    (
        ("cu_trimethylamine_zero.mol2", "du"),
        ("cu_trimethylamine_un.mol2", "un"),
        ("cu_trimethylamine_nc.mol2", "nc"),
    ),
)
def test_mol2_non_numeric_bond_types_change_v_but_not_d_or_x(
    zero_filename, source_bond_type
):
    zero_path = COORDINATION_FIXTURES / zero_filename
    single = hp.read_mol(COORDINATION_FIXTURES / "cu_trimethylamine_single.mol2")
    zero = hp.read_mol(zero_path)
    single_donor = single.atoms[0]
    zero_donor = zero.atoms[0]
    zero_bond = zero.bond(0, 4)

    assert zero_path.read_text().splitlines()[-1].split()[-1] == source_bond_type
    assert (
        (
            len(single_donor.neighbours),
            single_donor.implicit_hydrogens,
        )
        == (
            len(zero_donor.neighbours),
            zero_donor.implicit_hydrogens,
        )
        == (4, 0)
    )
    assert (single_donor.sum_bond_orders, zero_donor.sum_bond_orders) == (4, 3)
    assert zero_bond.bond_kind is BondKind.UNKNOWN
    assert zero_bond.bond_source_metadata["raw_bond_order"] == 0
    assert zero_bond.bond_source_metadata["raw_flags"] == 0
    assert matching_atom_sets(single, "[N]-[M]") == {frozenset((0, 4))}
    assert matching_atom_sets(zero, "[N]-[M]") == set()
    assert matching_atom_sets(zero, "[N]~[M]") == {frozenset((0, 4))}


def test_same_single_bond_topology_has_format_specific_hydrogen_perception():
    mol2 = hp.read_mol(COORDINATION_FIXTURES / "cu_trimethylamine_single.mol2")
    sdf = hp.read_mol(COORDINATION_FIXTURES / "cu_trimethylamine_single.sdf")

    assert [(bond.a1idx, bond.a2idx, bond.bond_order) for bond in mol2.bonds] == [
        (bond.a1idx, bond.a2idx, bond.bond_order) for bond in sdf.bonds
    ]
    assert mol2.atoms[0].implicit_hydrogens == 0
    assert sdf.atoms[0].implicit_hydrogens == 1
    assert matching_indices(mol2, "[N;X4;v4]") == {0}
    assert matching_indices(sdf, "[N;X5;v5]") == {0}


def test_real_afoqui_contains_a_six_connected_metal():
    molecule = hp.read_mol(REPOSITORY_INPUTS / "AFOQUI_clean.cif")
    six_connected_metals = [
        atom for atom in molecule.atoms if atom.is_metal and len(atom.neighbours) == 6
    ]

    assert [atom.idx for atom in six_connected_metals] == [9]
    metal = six_connected_metals[0]
    assert metal.implicit_hydrogens == 0
    assert metal.sum_bond_orders == 6
    assert all(bond.is_metal_ligand_bond for bond in metal.bonds)
    assert matching_indices(molecule, "[M;D6;X6;v6]") == {metal.idx}


def test_real_afekax_coordinated_nitrogens_retain_x3_profile():
    molecule = hp.read_mol(REPOSITORY_INPUTS / "AFEKAX_clean.cif")
    metal = next(atom for atom in molecule.atoms if atom.is_metal)
    donors = {
        bond.another_end(metal) for bond in metal.bonds if bond.is_metal_ligand_bond
    }

    assert (metal.symbol, metal.idx) == ("Cu", 64)
    assert len(donors) == 4
    assert {
        (
            donor.symbol,
            len(donor.neighbours),
            donor.implicit_hydrogens,
            donor.sum_bond_orders,
        )
        for donor in donors
    } == {("N", 3, 0, 3)}
    assert {donor.idx for donor in donors} <= matching_indices(molecule, "[N;D3;X3;v3]")
