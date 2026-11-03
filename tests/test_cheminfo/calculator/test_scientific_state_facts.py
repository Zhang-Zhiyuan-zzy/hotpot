"""Contracts and identity behavior for immutable molecular facts."""

import json
from dataclasses import FrozenInstanceError, asdict

import hotpot
import numpy as np
import pytest

from hotpot.cheminfo.core import BondKind, Molecule
from hotpot.cheminfo.structure.snapshots import MolecularFactsIdentity
from tests.test_cheminfo._stage5_fixtures import supplied_aromatic_flags
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts


def _raw_factual_state(mol):
    """Capture exact carrier bytes without getters that prepare chemistry."""
    table = mol._atom_table
    size = len(table)
    return (
        tuple(block[:size].tobytes() for block in table._blocks),
        table._presence[:size].tobytes(),
        table._sources[:size].tobytes(),
        table._group_status[:size].tobytes(),
        tuple(frozenset(seed) for seed in table._dirty_seeds),
        tuple((bond._attrs.tobytes(), bond._bond_kind, bond._is_aromatic) for bond in mol._bonds),
    )


def _hydrogen_extraction_fixture(case):
    if case == "parsed_ethanol":
        return hotpot.read_mol("CCO", "smi")
    mol = Molecule()
    if case == "explicit_zero":
        mol.create_atom(atomic_number=6, implicit_hydrogens=0)
    elif case == "absent_carbon_h":
        mol.create_atom(atomic_number=6)
    elif case == "absent_element_h":
        mol.create_atom()
    elif case == "element_change":
        atom = mol.create_atom(atomic_number=6)
        mol.calc_implicit_hydrogens()
        atom.atomic_number = 8
    elif case == "topology_change":
        mol.create_atom(atomic_number=6)
        mol.create_atom(atomic_number=6)
        mol.calc_implicit_hydrogens()
        mol.add_bond(0, 1)
    return mol


@pytest.mark.parametrize(
    ("case", "implicit_counts", "nuclear_charge_sum"),
    (
        ("explicit_zero", (None,), 6),
        ("absent_carbon_h", (4,), 10),
        ("absent_element_h", (None,), 0),
        ("element_change", (2,), 10),
        ("topology_change", (3, 3), 18),
        ("parsed_ethanol", (3, 2, 1), 26),
    ),
)
def test_extraction_prepares_only_its_working_copy(case, implicit_counts, nuclear_charge_sum):
    mol = _hydrogen_extraction_fixture(case)
    before = _raw_factual_state(mol)

    facts = extract_molecular_facts(mol)

    assert _raw_factual_state(mol) == before
    assert facts.hydrogen_inventory.implicit_counts == implicit_counts
    assert facts.hydrogen_inventory.explicit_counts == (0,) * len(implicit_counts)
    assert facts.nuclear_charge_sum == nuclear_charge_sum


def test_conflicting_hydrogen_preparation_preserves_the_original_carrier():
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol = supplied_aromatic_flags(mol, (True,))
    mol.atoms[0].atomic_number = 7
    before = _raw_factual_state(mol)

    with pytest.raises(RuntimeError) as failure:
        extract_molecular_facts(mol)

    assert str(failure.value) == (
        "Implicit-H representation conflicts with current facts; call calc_implicit_hydrogens() explicitly"
    )
    assert _raw_factual_state(mol) == before


def test_extract_molecular_facts_is_immutable_and_non_mutating() -> None:
    mol = hotpot.read_mol("CCO", "smi")
    before = _raw_factual_state(mol)

    facts = extract_molecular_facts(mol)

    assert _raw_factual_state(mol) == before
    assert facts.atomic_numbers == (6, 6, 8)
    assert facts.nuclear_charge_sum == 26
    assert isinstance(facts.identity, MolecularFactsIdentity)
    with pytest.raises(FrozenInstanceError):
        facts.nuclear_charge_sum = 0


def test_molecular_facts_have_deterministic_json_serialization() -> None:
    mol = hotpot.read_mol("[Eu].N", "smi")

    first = json.dumps(asdict(extract_molecular_facts(mol)), sort_keys=True)
    second = json.dumps(asdict(extract_molecular_facts(mol)), sort_keys=True)

    assert first == second


def test_coordinates_change_only_the_geometry_digest() -> None:
    mol = hotpot.read_mol("CCO", "smi")
    mol.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.1, 0.0))
    )
    first = extract_molecular_facts(mol)

    mol.coordinates = mol.coordinates + (3.0, -2.0, 1.0)
    second = extract_molecular_facts(mol)

    assert first.identity == second.identity
    assert first.geometry_digest != second.geometry_digest


def test_hydrogen_or_topology_change_changes_chemical_identity() -> None:
    mol = hotpot.read_mol("CCO", "smi")
    original = extract_molecular_facts(mol)

    mol.atoms[0].implicit_hydrogens -= 1
    changed_hydrogen = extract_molecular_facts(mol)
    assert changed_hydrogen.identity != original.identity

    mol.atoms[0].implicit_hydrogens += 1
    mol.add_bond(0, 2)
    changed_topology = extract_molecular_facts(mol)
    assert changed_topology.identity != original.identity


def test_unknown_metal_bond_is_not_silently_promoted_to_coordination() -> None:
    mol = hotpot.read_mol("[Eu]N", "smi")
    facts = extract_molecular_facts(mol)

    assert facts.semantic_bonds[0].kind == "unknown"
    assert len(facts.source_components) == 1
    assert len(facts.covalent_components) == 1


def test_declared_coordination_edge_splits_only_the_covalent_view() -> None:
    mol = hotpot.read_mol("[Eu].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)

    assert len(facts.source_components) == 1
    assert tuple(component.atom_indices for component in facts.covalent_components) == (
        (0,),
        (1,),
    )
    assert facts.semantic_bonds[0].kind == "coordination"


def test_dot_fragments_remain_distinct_source_components() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Eu].N", "smi"))

    assert tuple(component.atom_indices for component in facts.source_components) == (
        (0,),
        (1,),
    )


def test_verified_atom_map_makes_identity_independent_of_storage_order() -> None:
    first = Molecule()
    first.create_atom(symbol="C", id=10, implicit_hydrogens=3)
    first.create_atom(symbol="O", id=20, implicit_hydrogens=1)
    first.add_bond(0, 1)

    reordered = Molecule()
    reordered.create_atom(symbol="O", id=20, implicit_hydrogens=1)
    reordered.create_atom(symbol="C", id=10, implicit_hydrogens=3)
    reordered.add_bond(0, 1)

    assert extract_molecular_facts(first).identity == extract_molecular_facts(
        reordered
    ).identity
