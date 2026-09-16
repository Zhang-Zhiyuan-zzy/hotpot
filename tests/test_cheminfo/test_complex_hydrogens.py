from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff
from hotpot.cheminfo.core import BondKind


def _coordinated(smiles, donor_index):
    molecule = read_mol(smiles)
    metal = molecule.create_atom(symbol="Zn")
    molecule.add_bond(metal, molecule.atoms[donor_index])
    molecule.refresh_atom_id()
    return molecule


def _matching_atoms(molecule, predicate):
    return [atom for atom in molecule.atoms if predicate(atom)]


def _coordinate_matching_atoms(smiles, predicate):
    molecule = read_mol(smiles)
    donors = _matching_atoms(molecule, predicate)
    metal = molecule.create_atom(symbol="Zn")
    for donor in donors:
        molecule.add_bond(metal, donor)
    molecule.refresh_atom_id()
    return molecule, tuple(donor.idx for donor in donors)


def _heavy_atom_state(molecule):
    heavy_atoms = [atom for atom in molecule.atoms if not atom.is_hydrogen]
    return (
        Counter(atom.atomic_number for atom in heavy_atoms),
        tuple(
            (atom.id, atom.atomic_number, atom.formal_charge)
            for atom in heavy_atoms
        ),
        tuple(sorted(
            (
                min(bond.atom1.id, bond.atom2.id),
                max(bond.atom1.id, bond.atom2.id),
                bond.bond_order,
            )
            for bond in molecule.bonds
            if not (bond.atom1.is_hydrogen or bond.atom2.is_hydrogen)
        )),
    )


def _topology_state(molecule):
    return (
        tuple(
            (atom.id, atom.atomic_number, atom.formal_charge)
            for atom in molecule.atoms
        ),
        tuple(sorted(
            (
                min(bond.a1idx, bond.a2idx),
                max(bond.a1idx, bond.a2idx),
                bond.bond_order,
            )
            for bond in molecule.bonds
        )),
        molecule.coordinates.copy(),
    )


@pytest.mark.parametrize(
    ("smiles", "donor_index", "expected_donor_hydrogens", "expected_charge"),
    (
        ("O", 0, 2, 0),
        ("CO", 1, 1, 0),
        ("[O-]", 0, 0, -1),
        ("N", 0, 3, 0),
    ),
)
def test_hydrogenated_working_copy_uses_ligand_covalent_valence(
    smiles,
    donor_index,
    expected_donor_hydrogens,
    expected_charge,
):
    molecule = _coordinated(smiles, donor_index)
    original = _topology_state(molecule)

    working = ff._hydrogenated_working_copy(molecule, add_hydrogens=True)

    donor = working.atoms[donor_index]
    assert donor.explicit_hydrogens == expected_donor_hydrogens
    assert donor.formal_charge == expected_charge
    assert len(working.c_bonds) == 1
    current = _topology_state(molecule)
    assert current[:2] == original[:2]
    np.testing.assert_array_equal(current[2], original[2])


def test_hydrogenated_working_copy_can_explicitly_skip_hydrogens():
    molecule = _coordinated("O", 0)

    working = ff._hydrogenated_working_copy(molecule, add_hydrogens=False)

    assert working.hydrogens == []
    assert len(working.c_bonds) == 1


@pytest.mark.parametrize(
    (
        "smiles",
        "donor_predicate",
        "expected_donor_hydrogens",
        "expected_total_hydrogens",
    ),
    (
        pytest.param(
            "Oc1ccccc1",
            lambda atom: atom.symbol == "O"
            and any(neighbour.is_aromatic for neighbour in atom.heavy_neighbours),
            1,
            6,
            id="phenol-hydroxyl-oxygen",
        ),
        pytest.param(
            "CC(=O)O",
            lambda atom: atom.symbol == "O" and atom.sum_covalent_orders == 1,
            1,
            4,
            id="carboxylic-acid-hydroxyl-oxygen",
        ),
        pytest.param(
            "[nH]1cccc1",
            lambda atom: atom.symbol == "N" and atom.is_aromatic,
            1,
            5,
            id="aromatic-nh-nitrogen",
        ),
        pytest.param(
            "CC(=O)C",
            lambda atom: atom.symbol == "O" and atom.sum_covalent_orders == 2,
            0,
            6,
            id="carbonyl-oxygen-negative-control",
        ),
    ),
)
def test_hydrogenated_working_copy_preserves_ligand_chemical_identity(
    smiles,
    donor_predicate,
    expected_donor_hydrogens,
    expected_total_hydrogens,
):
    molecule, donor_indices = _coordinate_matching_atoms(smiles, donor_predicate)
    assert len(donor_indices) == 1
    original_heavy_state = _heavy_atom_state(molecule)

    working = ff._hydrogenated_working_copy(
        molecule,
        add_hydrogens=True,
        seed=11,
    )

    donor = working.atoms[donor_indices[0]]
    assert donor.explicit_hydrogens == expected_donor_hydrogens
    assert donor.formal_charge == 0
    assert len(working.hydrogens) == expected_total_hydrogens
    assert len(working.c_bonds) == 1
    assert _heavy_atom_state(working) == original_heavy_state
    assert _heavy_atom_state(molecule) == original_heavy_state


def test_hydrogenated_working_copy_preserves_explicit_hydroxyl_hydrogen():
    molecule, donor_indices = _coordinate_matching_atoms(
        "[H]Oc1ccccc1",
        lambda atom: atom.symbol == "O",
    )
    assert len(donor_indices) == 1
    donor_index = donor_indices[0]
    donor = molecule.atoms[donor_index]
    explicit_hydrogen_id = donor.hydrogens[0].id
    original_heavy_state = _heavy_atom_state(molecule)

    working = ff._hydrogenated_working_copy(
        molecule,
        add_hydrogens=True,
        seed=13,
    )

    working_donor = working.atoms[donor_index]
    assert working_donor.explicit_hydrogens == 1
    assert explicit_hydrogen_id in {
        hydrogen.id for hydrogen in working_donor.hydrogens
    }
    assert len(working.hydrogens) == 6
    assert _heavy_atom_state(working) == original_heavy_state
    assert molecule.atoms[donor_index].explicit_hydrogens == 1


def test_hydrogenated_working_copy_preserves_bidentate_ligand_identity():
    molecule, donor_indices = _coordinate_matching_atoms(
        "OCCO",
        lambda atom: atom.symbol == "O",
    )
    assert len(donor_indices) == 2
    original_heavy_state = _heavy_atom_state(molecule)

    working = ff._hydrogenated_working_copy(
        molecule,
        add_hydrogens=True,
        seed=17,
    )

    assert [working.atoms[index].explicit_hydrogens for index in donor_indices] == [1, 1]
    assert [working.atoms[index].formal_charge for index in donor_indices] == [0, 0]
    assert len(working.hydrogens) == 6
    assert len(working.c_bonds) == 2
    assert _heavy_atom_state(working) == original_heavy_state
    assert _heavy_atom_state(molecule) == original_heavy_state


def test_optimizer_failure_does_not_modify_the_caller(monkeypatch):
    molecule = _coordinated("O", 0)
    original = _topology_state(molecule)

    def fail_after_mutating_working(working, **options):
        working.coordinates = working.coordinates + 5.0
        raise ff.ForceFieldSetupError("deliberate failure")

    monkeypatch.setattr(ff, "_run_optimizer_on_working", fail_after_mutating_working)

    with pytest.raises(ff.ForceFieldSetupError, match="deliberate failure"):
        ff.optimize(molecule, add_hydrogens=True)

    current = _topology_state(molecule)
    assert current[:2] == original[:2]
    np.testing.assert_array_equal(current[2], original[2])


def test_optimizer_failure_preserves_explicit_hydrogens_and_bidentate_topology(
    monkeypatch,
):
    molecule, donor_indices = _coordinate_matching_atoms(
        "[H]OCCO[H]",
        lambda atom: atom.symbol == "O",
    )
    assert len(donor_indices) == 2
    original = _topology_state(molecule)

    def fail_after_replacing_working_topology(working, **options):
        for bond in working.bonds:
            working.remove_bonds([bond])
        raise ff.ForceFieldSetupError("deliberate proxy failure")

    monkeypatch.setattr(
        ff,
        "_run_optimizer_on_working",
        fail_after_replacing_working_topology,
    )

    with pytest.raises(ff.ForceFieldSetupError, match="deliberate proxy failure"):
        ff.optimize(molecule, add_hydrogens=True)

    current = _topology_state(molecule)
    assert current[:2] == original[:2]
    np.testing.assert_array_equal(current[2], original[2])


def test_unsupported_dative_conversion_fails_without_mutating_caller():
    molecule = read_mol("N.[Zn]", "smi")
    molecule.add_bond(
        molecule.atoms[1],
        molecule.atoms[0],
        bond_kind=BondKind.DATIVE,
    )
    molecule.refresh_atom_id()
    original = _topology_state(molecule)

    with pytest.raises(ValueError, match="cannot represent dative bonds"):
        ff.optimize(
            molecule,
            add_hydrogens=False,
            epochs=1,
            steps_per_epoch=1,
        )

    current = _topology_state(molecule)
    assert current[:2] == original[:2]
    np.testing.assert_array_equal(current[2], original[2])


def test_successful_optimizer_commits_added_hydrogens_once(monkeypatch):
    molecule = read_mol("O")
    sentinel = SimpleNamespace(best_energy=-1.0)

    def accept_working(working, **options):
        return sentinel

    monkeypatch.setattr(ff, "_run_optimizer_on_working", accept_working)

    result = ff.optimize(molecule)

    assert result is sentinel
    assert len(molecule.hydrogens) == 2
    assert molecule.atoms[0].explicit_hydrogens == 2


def test_successful_commit_preserves_existing_object_identity_and_atom_ids(monkeypatch):
    molecule = read_mol("CO")
    molecule.atoms[0].id = 101
    molecule.atoms[1].id = 305
    original_atoms = tuple(molecule.atoms)
    original_bond = molecule.bonds[0]
    original_conformers = molecule.conformers
    original_atom_pairs = molecule.atom_pairs
    molecule._mca_sites = {original_atoms[1]: 314.0}
    original_atoms[1]._mca = 314.0
    sentinel = SimpleNamespace(best_energy=-2.0)

    def accept_working(working, **options):
        working.coordinates = working.coordinates + 1.0
        working.conformer_clear()
        working.conformer_add(working.coordinates, -2.0)
        return sentinel

    monkeypatch.setattr(ff, "_run_optimizer_on_working", accept_working)

    result = ff.optimize(molecule)

    assert result is sentinel
    assert tuple(molecule.atoms[:2]) == original_atoms
    assert molecule.bonds[0] is original_bond
    assert molecule.conformers is original_conformers
    assert molecule.atom_pairs is original_atom_pairs
    assert [atom.id for atom in molecule.atoms[:2]] == [101, 305]
    assert [atom.idx for atom in original_atoms] == [0, 1]
    assert all(atom.mol is molecule for atom in original_atoms)
    assert molecule._mca_sites == {original_atoms[1]: 314.0}
    assert original_atoms[1]._mca == 314.0
    added_ids = [atom.id for atom in molecule.atoms[2:]]
    assert len(added_ids) == len(set(added_ids))
    assert not {101, 305}.intersection(added_ids)
