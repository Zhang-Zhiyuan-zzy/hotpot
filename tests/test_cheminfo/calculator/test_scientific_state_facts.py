"""Contracts and identity behavior for immutable molecular facts."""

import json
from dataclasses import FrozenInstanceError, asdict

import hotpot
import numpy as np
import pytest

from hotpot.cheminfo.core import BondKind, Molecule
from hotpot.cheminfo.calculator.contracts import MolecularFactsIdentity
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts


def test_extract_molecular_facts_is_immutable_and_non_mutating() -> None:
    mol = hotpot.read_mol("CCO", "smi")
    before = (
        tuple(atom.attrs.tolist() for atom in mol.atoms),
        tuple((bond.a1idx, bond.a2idx, bond.bond_order) for bond in mol.bonds),
    )

    facts = extract_molecular_facts(mol)

    after = (
        tuple(atom.attrs.tolist() for atom in mol.atoms),
        tuple((bond.a1idx, bond.a2idx, bond.bond_order) for bond in mol.bonds),
    )
    assert before == after
    assert facts.atomic_numbers == (6, 6, 8)
    assert facts.nuclear_charge_sum == 20
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
