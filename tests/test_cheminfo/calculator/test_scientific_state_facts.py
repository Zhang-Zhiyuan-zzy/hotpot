"""Contracts and identity behavior for immutable molecular facts."""

import json
from dataclasses import FrozenInstanceError, asdict

import hotpot
import numpy as np
import pytest

from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
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
        ("explicit_zero", (0,), 6),
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
        "HYDROGEN.REPRESENTATION_CONFLICT atom_index=0: Implicit-H representation conflicts with current facts; call calc_implicit_hydrogens() explicitly"
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


@pytest.mark.parametrize("source", ("imported", "inferred"))
def test_current_imported_and_inferred_zero_hydrogens_remain_known(source):
    if source == "imported":
        mol = hotpot.read_mol("[Cl-]", "smi")
        source_codes = (ATOM_SOURCE_CODES["openbabel"], ATOM_SOURCE_CODES["rdkit"])
    else:
        mol = Molecule()
        mol.create_atom(atomic_number=17, formal_charge=-1)
        mol.calc_implicit_hydrogens()
        source_codes = (ATOM_SOURCE_CODES["implicit_hydrogens"],)
    table = mol._atom_table
    field = FIELD_INDEX["implicit_hydrogens"]
    assert table._columns[field][0] == 0
    assert int(table._presence[0]) & FIELD_PRESENCE_BITS[field]
    assert table._group_status[0, GROUP_INDEX["IMPLICIT_H"]] == _FactStatus.CURRENT
    assert int(table._sources[0, field]) in source_codes
    before = _raw_factual_state(mol)

    facts = extract_molecular_facts(mol)

    assert facts.hydrogen_inventory.implicit_counts == (0,)
    assert facts.nuclear_charge_sum == 17
    assert _raw_factual_state(mol) == before


def _unknown_neighbour_molecule(*, known_zero):
    mol = Molecule()
    supplied_hydrogens = {"implicit_hydrogens": 0} if known_zero else {}
    mol.create_atom(atomic_number=6, id=0, **supplied_hydrogens)
    mol.create_atom(id=1)
    mol.add_bond(0, 1)
    return mol


def test_known_zero_and_unresolved_hydrogens_have_distinct_fact_identities():
    known = _unknown_neighbour_molecule(known_zero=True)
    unknown = _unknown_neighbour_molecule(known_zero=False)
    before_known = _raw_factual_state(known)
    before_unknown = _raw_factual_state(unknown)

    known_facts = extract_molecular_facts(known)
    unknown_facts = extract_molecular_facts(unknown)

    assert known_facts.atomic_numbers == unknown_facts.atomic_numbers == (6, 0)
    assert known_facts.atom_map == unknown_facts.atom_map == (0, 1)
    assert known_facts.semantic_bonds == unknown_facts.semantic_bonds
    assert known_facts.hydrogen_inventory.implicit_counts == (0, None)
    assert unknown_facts.hydrogen_inventory.implicit_counts == (None, None)
    assert known_facts.nuclear_charge_sum == unknown_facts.nuclear_charge_sum == 6
    assert known_facts.identity != unknown_facts.identity
    assert _raw_factual_state(known) == before_known
    assert _raw_factual_state(unknown) == before_unknown


def test_unresolved_raw_zero_is_not_promoted_to_a_known_zero():
    mol = Molecule()
    mol.create_atom()
    mol.calc_implicit_hydrogens()
    table = mol._atom_table
    field = FIELD_INDEX["implicit_hydrogens"]
    assert table._columns[field][0] == 0
    assert table._group_status[0, GROUP_INDEX["IMPLICIT_H"]] == _FactStatus.UNRESOLVED
    assert not int(table._presence[0]) & FIELD_PRESENCE_BITS[field]
    before = _raw_factual_state(mol)

    facts = extract_molecular_facts(mol)

    assert facts.hydrogen_inventory.implicit_counts == (None,)
    assert facts.nuclear_charge_sum == 0
    assert _raw_factual_state(mol) == before


def test_pending_inferred_zero_requires_preparation_on_the_working_copy():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=17, formal_charge=-1)
    mol.calc_implicit_hydrogens()
    atom.atomic_number = 0
    table = mol._atom_table
    field = FIELD_INDEX["implicit_hydrogens"]
    assert table._columns[field][0] == 0
    assert int(table._presence[0]) & FIELD_PRESENCE_BITS[field]
    assert atom._uid in table._dirty_seeds[GROUP_INDEX["IMPLICIT_H"]]
    before = _raw_factual_state(mol)

    facts = extract_molecular_facts(mol)

    assert facts.hydrogen_inventory.implicit_counts == (None,)
    assert facts.nuclear_charge_sum == 0
    assert _raw_factual_state(mol) == before


def test_chloride_zero_h_preserves_charge_electrons_lewis_and_singlet_projection():
    from hotpot.cheminfo.operations.calculation.scientific_state.contracts import ProviderSet, ProviderToggle
    from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import BackendProjectionStatus, resolve_and_project_state
    from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
    from tests.test_cheminfo.calculator.test_backend_state_projection import _AllElectronBackend

    mol = hotpot.read_mol("[Cl-]", "smi")
    before = _raw_factual_state(mol)
    facts = extract_molecular_facts(mol)

    workflow = resolve_and_project_state(
        facts,
        _AllElectronBackend(),
        StateRequest(net_charge=-1, two_s=0),
        provider_set=ProviderSet((ProviderToggle("PROVIDER-001", True),)),
    )

    bundle = workflow.scientific_state
    assert facts.hydrogen_inventory.implicit_counts == (0,)
    assert facts.nuclear_charge_sum == 17
    assert tuple(candidate.charge for result in bundle.net_charge_results for candidate in result.candidates) == (-1,)
    assert tuple(
        (candidate.atom_formal_charges, candidate.nonbonding_electrons, candidate.radical_electrons)
        for result in bundle.lewis_results for candidate in result.candidates
    ) == (((-1,), (8,), (0,)),)
    assert tuple(
        (candidate.electron_count, candidate.two_s, candidate.multiplicity)
        for result in bundle.electronic_state_results for candidate in result.candidates
    ) == ((18, 0, 1),)
    result = workflow.backend_projection
    assert result.status is BackendProjectionStatus.PROJECTED
    assert (result.projection.charge, result.projection.effective_electron_count, result.projection.uhf, result.projection.multiplicity) == (-1, 18, 0, 1)
    assert _raw_factual_state(mol) == before


def test_evidence_from_unresolved_h_is_rejected_for_the_known_zero_boundary():
    from hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge import resolve_net_charge
    from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import ScientificAdmissibility
    from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

    unknown = _unknown_neighbour_molecule(known_zero=False)
    known = _unknown_neighbour_molecule(known_zero=True)
    before_unknown = _raw_factual_state(unknown)
    before_known = _raw_factual_state(known)
    unknown_facts = extract_molecular_facts(unknown)
    known_facts = extract_molecular_facts(known)
    provided = resolve_net_charge(unknown_facts, StateRequest(net_charge=0))[0]
    assert provided.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert tuple(candidate.charge for candidate in provided.candidates) == (0,)

    rejected = resolve_net_charge(known_facts, provided=(provided,))[1]

    assert rejected.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert tuple(finding.code for finding in rejected.findings) == ("HARD-001.FACTS_IDENTITY_MISMATCH",)
    assert _raw_factual_state(unknown) == before_unknown
    assert _raw_factual_state(known) == before_known
