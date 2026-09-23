from collections import Counter

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.forcefields import utils as ff
from hotpot.cheminfo.forcefields import working_copy
from hotpot.cheminfo.forcefields import workflows
from hotpot.cheminfo.core import BondKind


def _coordinated(smiles, donor_index):
    molecule = read_mol(smiles)
    metal = molecule.create_atom(symbol="Zn")
    molecule.add_bond(metal, molecule.atoms[donor_index])
    molecule.refresh_atom_id()
    return molecule


def _optimization_report(best_energy):
    return ff.ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=1,
        steps_submitted=1,
        initialization_steps=1,
        steps_completed=None,
        final_energy=best_energy,
        best_energy=best_energy,
        energy_unit="kJ/mol",
        rms_gradient=0.0,
        max_gradient=0.0,
        exploded=False,
    )


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

    working = working_copy._hydrogenated_working_copy(
        molecule,
        add_hydrogens=True,
    )

    donor = working.atoms[donor_index]
    assert donor.explicit_hydrogens == expected_donor_hydrogens
    assert donor.formal_charge == expected_charge
    assert len(working.c_bonds) == 1
    current = _topology_state(molecule)
    assert current[:2] == original[:2]
    np.testing.assert_array_equal(current[2], original[2])


@pytest.mark.parametrize(
    (
        "ligand_smiles",
        "complex_smiles",
        "donor_symbol",
        "expected_hydrogens",
    ),
    (
        ("CS", "[Zn]SC", "S", 1),
        ("CP", "[Zn]PC", "P", 2),
        ("C[Se]", "[Zn][Se]C", "Se", 1),
        ("C[As]", "[Zn][As]C", "As", 2),
    ),
)
def test_neutral_donor_hydrogenation_is_independent_of_input_path(
    ligand_smiles,
    complex_smiles,
    donor_symbol,
    expected_hydrogens,
):
    assembled = read_mol(ligand_smiles, "smi")
    assembled_donor = next(
        atom for atom in assembled.atoms if atom.symbol == donor_symbol
    )
    metal = assembled.create_atom(symbol="Zn")
    assembled.add_bond(metal, assembled_donor)
    assembled.refresh_atom_id()
    parsed = read_mol(complex_smiles, "smi")

    assembled_working = working_copy._hydrogenated_working_copy(
        assembled,
        add_hydrogens=True,
        seed=5,
    )
    parsed_working = working_copy._hydrogenated_working_copy(
        parsed,
        add_hydrogens=True,
        seed=5,
    )

    assembled_hydrogens = next(
        atom.explicit_hydrogens
        for atom in assembled_working.atoms
        if atom.symbol == donor_symbol
    )
    parsed_hydrogens = next(
        atom.explicit_hydrogens
        for atom in parsed_working.atoms
        if atom.symbol == donor_symbol
    )
    assert assembled_hydrogens == parsed_hydrogens == expected_hydrogens


@pytest.mark.parametrize(
    ("smiles", "donor_symbols", "expected_hydrogens"),
    (
        ("[Zn]C", ("C",), (3,)),
        ("[Zn]C#N", ("C",), (0,)),
        ("[Pt](Cl)(Cl)(Cl)(Cl)", ("Cl", "Cl", "Cl", "Cl"), (0, 0, 0, 0)),
    ),
)
def test_hydrogenated_working_copy_does_not_reprotonate_covalent_metal_bonds(
    smiles,
    donor_symbols,
    expected_hydrogens,
):
    molecule = read_mol(smiles, "smi")
    original_heavy_state = _heavy_atom_state(molecule)

    working = working_copy._hydrogenated_working_copy(
        molecule,
        add_hydrogens=True,
        seed=7,
    )

    donors = [
        atom
        for atom in working.atoms
        if atom.symbol in donor_symbols and not atom.is_metal
    ]
    assert tuple(atom.explicit_hydrogens for atom in donors) == expected_hydrogens
    assert _heavy_atom_state(working) == original_heavy_state
    assert _heavy_atom_state(molecule) == original_heavy_state


def test_hydrogenated_working_copy_can_explicitly_skip_hydrogens():
    molecule = _coordinated("O", 0)

    working = working_copy._hydrogenated_working_copy(molecule, add_hydrogens=False)

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

    working = working_copy._hydrogenated_working_copy(
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

    working = working_copy._hydrogenated_working_copy(
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

    working = working_copy._hydrogenated_working_copy(
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

    monkeypatch.setattr(workflows, "_optimize_working_mol", fail_after_mutating_working)

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
        workflows,
        "_optimize_working_mol",
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
    sentinel = _optimization_report(-1.0)

    def accept_working(working, **options):
        return sentinel

    monkeypatch.setattr(workflows, "_optimize_working_mol", accept_working)

    result = ff.optimize(molecule)

    assert result.best_energy == sentinel.best_energy
    assert result.trajectory is not None
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
    sentinel = _optimization_report(-2.0)

    def accept_working(working, **options):
        working.coordinates = working.coordinates + 1.0
        working.conformer_clear()
        working.conformer_add(working.coordinates, -2.0)
        return sentinel

    monkeypatch.setattr(workflows, "_optimize_working_mol", accept_working)

    result = ff.optimize(molecule)

    assert result.best_energy == sentinel.best_energy
    assert result.trajectory is not None
    assert all(
        current is original
        for current, original in zip(molecule.atoms[:2], original_atoms)
    )
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


def test_commit_rejects_a_working_copy_that_removed_an_original_atom(monkeypatch):
    molecule = read_mol("CC")
    working = molecule.copy()
    working._atoms.pop()
    original_atoms = tuple(molecule.atoms)
    original_coordinates = molecule.coordinates.copy()
    create_calls = 0

    def create_atom(attrs):
        nonlocal create_calls
        create_calls += 1

    monkeypatch.setattr(molecule, "_create_atom_from_array", create_atom)

    with pytest.raises(ValueError, match="removed an original atom"):
        working_copy._commit_working_copy(molecule, working)

    assert create_calls == 0
    assert tuple(molecule.atoms) == original_atoms
    np.testing.assert_array_equal(molecule.coordinates, original_coordinates)


def test_commit_rejects_changed_original_bond_topology_before_mutation():
    molecule = read_mol("CC")
    working = molecule.copy()
    working.bonds[0].bond_order = 2.0
    original_atom_attrs = tuple(atom.attrs for atom in molecule.atoms)
    original_bond = molecule.bonds[0]

    with pytest.raises(ValueError, match="changed the original bond topology"):
        working_copy._commit_working_copy(molecule, working)

    assert molecule.bonds[0] is original_bond
    assert all(
        atom.attrs is attrs
        for atom, attrs in zip(molecule.atoms, original_atom_attrs)
    )


def test_failed_commit_restores_every_caller_owned_container(monkeypatch):
    molecule = read_mol("CO")
    molecule.conformer_add(molecule.coordinates.copy(), -1.0)
    molecule.atom_pairs.update_pairs()
    _ = molecule.rings
    _ = molecule.cycle_basis_rings
    _ = molecule.ligand_rings
    _ = molecule.ligand_cycle_basis_rings
    working = working_copy._hydrogenated_working_copy(
        molecule,
        add_hydrogens=True,
        seed=13,
    )
    working.coordinates = working.coordinates + 1.0
    working.conformer_clear()
    working.conformer_add(working.coordinates, -2.0)

    original_atoms = tuple(molecule.atoms)
    original_bonds = tuple(molecule.bonds)
    original_atom_attrs = tuple(atom.attrs for atom in original_atoms)
    original_neighbours = tuple(atom._neighbours for atom in original_atoms)
    original_atom_bonds = tuple(atom._bonds for atom in original_atoms)
    original_graph = molecule._graph
    original_pairs = molecule.atom_pairs
    original_pair_items = tuple(original_pairs.items())
    original_conformers = molecule.conformers
    original_conformer_state = dict(original_conformers.__dict__)
    original_conformer_index = molecule._conformers_index
    original_caches = (
        molecule._row2idx,
        molecule._angles,
        molecule._torsions,
        molecule._rings,
        molecule._cycle_basis_rings,
        molecule._ring_indices_cache,
        molecule._ligand_rings,
        molecule._ligand_cycle_basis_rings,
        molecule._ligand_rings_signature,
        molecule._obmol,
    )

    def fail_after_partial_update(atom_pairs):
        dict.clear(atom_pairs)
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(type(molecule.atom_pairs), "update_pairs", fail_after_partial_update)

    with pytest.raises(RuntimeError, match="injected commit failure"):
        working_copy._commit_working_copy(molecule, working)

    assert tuple(molecule.atoms) == original_atoms
    assert tuple(molecule.bonds) == original_bonds
    assert all(
        atom.attrs is attrs
        for atom, attrs in zip(original_atoms, original_atom_attrs)
    )
    assert all(
        atom._neighbours is neighbours
        for atom, neighbours in zip(original_atoms, original_neighbours)
    )
    assert all(
        atom._bonds is bonds
        for atom, bonds in zip(original_atoms, original_atom_bonds)
    )
    assert molecule._graph is original_graph
    assert molecule.atom_pairs is original_pairs
    assert tuple(molecule.atom_pairs.items()) == original_pair_items
    assert molecule.conformers is original_conformers
    assert all(
        molecule.conformers.__dict__[name] is value
        for name, value in original_conformer_state.items()
    )
    assert molecule._conformers_index == original_conformer_index
    assert (
        molecule._row2idx,
        molecule._angles,
        molecule._torsions,
        molecule._rings,
        molecule._cycle_basis_rings,
        molecule._ring_indices_cache,
        molecule._ligand_rings,
        molecule._ligand_cycle_basis_rings,
        molecule._ligand_rings_signature,
        molecule._obmol,
    ) == original_caches
