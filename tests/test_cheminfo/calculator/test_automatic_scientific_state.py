"""Preregistered automatic coordination-state holdouts and source boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import hotpot
import pytest

from hotpot.cheminfo.operations.calculation.scientific_state import profiles
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
from hotpot.cheminfo.operations.calculation.scientific_state.facts import (
    extract_molecular_facts, _state_request_from_molecule,
    _source_radical_electrons, _molecule_state_source_identity,
)
from hotpot.cheminfo.operations.calculation.scientific_state.manifest import _coordination_screening_manifest
from hotpot.cheminfo.operations.calculation.scientific_state.profile_decision import _candidate_index, _molecular_two_s
from hotpot.cheminfo.operations.calculation.scientific_state.state_resolver import resolve_scientific_state
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateBudget, CoverageStatus, ScientificOperationRequirements, ScientificQuantity,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import (
    CandidateSearchStatus, MaterializationStatus,
)
from hotpot.cheminfo.operations.calculation.scientific_state.composition._decision_snapshot import (
    _decision_snapshot_from_json, _decision_snapshot_to_json,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts import (
    SpinContributionCandidate, SpinContributionScope,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import CoordinationCountingCandidate
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
from hotpot.cheminfo.structure._commit import _capture_source, _commit_molecular_declarations
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure.molecule import Molecule


_PROTOCOL = Path(__file__).resolve().parents[3] / "plan/20_fact_rule_architecture/stages/stage08/scientific-state/automatic-workflow/activation_protocol.json"
_HOLDOUTS = tuple(json.loads(_PROTOCOL.read_text())["independent_holdout"])


@pytest.fixture
def candidate_profile(monkeypatch):
    """Isolate preregistered candidate validation from production activation."""
    _, payload = profiles._profile_payload()
    payload = {**payload, "activation_status": "active"}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    monkeypatch.setattr(profiles, "_profile_payload", lambda: (encoded, payload))
    profiles._coordination_screening_profile.cache_clear()
    yield
    profiles._coordination_screening_profile.cache_clear()



@pytest.fixture
def planned_profile(monkeypatch):
    """Keep the inactive-artifact boundary independent of production activation."""
    _, payload = profiles._profile_payload()
    payload = {**payload, "activation_status": "planned"}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    monkeypatch.setattr(profiles, "_profile_payload", lambda: (encoded, payload))
    profiles._coordination_screening_profile.cache_clear()
    yield
    profiles._coordination_screening_profile.cache_clear()


def _molecule(ligand="O", metal=63):
    if ligand == "[CH2]CO":
        from rdkit import Chem
        from hotpot.cheminfo.operations.conversion import from_rdmol
        mol = from_rdmol(Chem.MolFromSmiles(ligand), Molecule())
    else:
        mol = hotpot.read_mol(ligand, "smi")
    donor = next((atom for atom in mol.atoms if atom.atomic_number in (7, 8, 15, 16)), mol.atoms[0])
    center = mol.create_atom(atomic_number=metal)
    mol.add_bond(center, donor, bond_kind=BondKind.DATIVE)
    _commit_molecular_declarations(mol, _capture_source(mol),
                                   charge=(False, None, None), spin_mult=(False, None, None))
    return mol


def _resolve(mol, request=StateRequest(), *, requirements=None, budget=None):
    facts = extract_molecular_facts(mol)
    request = _state_request_from_molecule(mol, facts, request)
    manifest = _coordination_screening_manifest(
        source_atom_formal_charges=request.atom_formal_charges,
        source_component_charges=request.component_charges,
        source_radical_electrons=_source_radical_electrons(mol),
        source_evidence_identity=_molecule_state_source_identity(mol, facts),
    )
    return resolve_scientific_state(
        facts, request, policy=manifest.policy, requirements=requirements,
        provider_set=manifest.provider_set, rule_set=manifest.rule_set,
        candidate_budget=manifest.candidate_budget if budget is None else budget,
    )


def _selected(bundle):
    selected = bundle.execution_decision.selected_state_identity
    return next((path for path in bundle.candidate_space.expanded_paths if path.identity == selected), None)


def _two_s(bundle, path):
    return _molecular_two_s(path, _candidate_index(bundle.candidate_space.factors))


@pytest.mark.parametrize("case", _HOLDOUTS, ids=lambda case: case["id"])
def test_preregistered_holdout_charge_and_physical_spin(candidate_profile, case):
    number = {"Eu": 63, "La": 57, "Gd": 64, "Zn": 30, "Cu": 29}[case["metal"]]
    result = _resolve(_molecule(case["ligand"], number))
    selected = _selected(result)

    assert selected is not None, result.execution_decision
    assert selected.net_charge.value == case["expected_charge"]
    assert _two_s(result, selected) == case["physical_two_s"]
    electron_count = result.facts.nuclear_charge_sum - selected.net_charge.value
    assert 0 <= _two_s(result, selected) <= electron_count
    assert (electron_count - _two_s(result, selected)) % 2 == 0
    assert all(result.admissibility.value != "inadmissible" for result in result.lewis_results)


def test_planned_profile_remains_gated_until_owner_activation(planned_profile):
    result = _resolve(_molecule())

    assert _selected(result) is None
    assert tuple(finding.code for finding in result.execution_decision.static_profile_eligibility.findings) == (
        "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",
    )
    assert result.execution_decision.action.value == "no_automatic_state"


@pytest.mark.parametrize("declaration", ("invocation", "molecule"))
def test_explicit_euii_charge_wins_over_common_euiii_default(candidate_profile, declaration):
    mol = _molecule()
    request = StateRequest(net_charge=2) if declaration == "invocation" else StateRequest()
    if declaration == "molecule":
        mol.charge = 2
    result = _resolve(mol, request)
    selected = _selected(result)

    assert selected.net_charge.value == 2
    assert _two_s(result, selected) == 7


def test_explicit_legal_molecular_spin_wins_over_hund_and_spin_defaults(candidate_profile):
    result = _resolve(_molecule(), StateRequest(net_charge=3, two_s=2, multiplicity=3))
    selected = _selected(result)

    assert selected.net_charge.value == 3
    assert _two_s(result, selected) == 2


def test_explicit_zero_charge_is_present_and_selects_legal_generic_zero_state(candidate_profile):
    result = _resolve(_molecule(metal=30), StateRequest(net_charge=0))
    selected = _selected(result)

    assert selected.net_charge.value == 0
    assert _two_s(result, selected) == 0


def test_invalid_explicit_spin_parity_is_a_hard_failure(candidate_profile):
    result = _resolve(_molecule(), StateRequest(net_charge=3, two_s=1, multiplicity=2))

    assert _selected(result) is None
    assert result.execution_decision.action.value == "inadmissible"
    assert any("PARITY" in finding.code for result in result.electronic_state_results for finding in result.findings)


def test_complete_supplied_euiii_and_conflicting_molecular_charge_remain_conflicting(candidate_profile):
    mol = hotpot.read_mol("[Eu+3].O", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    result = _resolve(mol, StateRequest(net_charge=2))

    assert _selected(result) is None
    assert result.execution_decision.action.value == "conflict"
    assert any(finding.code == "HARD-002.ATOM_CHARGE_SUM_MISMATCH"
               for result in result.net_charge_results for finding in result.findings)


def test_charged_ligand_has_one_source_count_without_neutral_l_alternative(candidate_profile):
    result = _resolve(_molecule("C[N+](C)(C)CCO"))
    ligand = result.facts.covalent_components[0]
    candidates = tuple(value.candidate for factor in result.candidate_space.factors for value in factor.values
                       if isinstance(value.candidate, CoordinationCountingCandidate)
                       and any(fragment.component_identity == ligand.identity for fragment in value.candidate.fragments))

    assert len(candidates) == 1
    assert candidates[0].fragments[0].ionic_bookkeeping_charge == 1
    assert candidates[0].provenance.authority.value == "verified_scientific_source"
    assert not next(trace.outcome for trace in result.rule_trace if trace.outcome.rule_id == "OPTIONAL-003").applicability.matches


def test_partial_supplied_charge_uses_named_completion_without_erasing_charge(candidate_profile):
    mol = Molecule()
    mol.create_atom(atomic_number=7, formal_charge=1, implicit_hydrogens=3)
    mol.create_atom(atomic_number=6, implicit_hydrogens=2)
    mol.create_atom(atomic_number=8, implicit_hydrogens=1)
    mol.add_bond(0, 1)
    mol.add_bond(1, 2)
    center = mol.create_atom(atomic_number=63)
    mol.add_bond(center, mol.atoms[2], bond_kind=BondKind.DATIVE)
    result = _resolve(mol)
    selected = _selected(result)

    assert selected.net_charge.value == 4
    assert any("representation_default:unspecified-site-formal-charge-zero-v1" in candidate.provenance.assumption_ids
               for factor in result.candidate_space.factors for value in factor.values
               for candidate in (value.candidate,))


def test_absent_numeric_zero_is_not_promoted_to_supplied_charge():
    mol = Molecule()
    mol.create_atom(atomic_number=8)
    facts = extract_molecular_facts(mol)
    request = _state_request_from_molecule(mol, facts, StateRequest())

    assert request.atom_formal_charges is None
    assert request.component_charges is None
    assert request.charge_candidates == ()
    mol.atoms[0].formal_charge = 0
    request = _state_request_from_molecule(mol, extract_molecular_facts(mol), StateRequest())
    assert request.atom_formal_charges.charges == (0,)
    assert request.atom_formal_charges.coverage is CoverageStatus.COMPLETE
    assert request.charge_candidates[0].charge == 0


def test_source_helpers_preserve_carrier_and_explicit_invocation_fields():
    mol = _molecule("[CH2]CO")
    mol.charge = 2
    mol.spin_mult = 8
    table = mol._atom_table
    before = (table._presence.copy(), table._sources.copy(), table._group_status.copy(),
              tuple(tuple(row) for row in table._stereo_facts[:len(table)]))
    facts = extract_molecular_facts(mol)
    request = _state_request_from_molecule(mol, facts, StateRequest(net_charge=3, multiplicity=7))

    assert request.net_charge == 3
    assert request.multiplicity == 7
    assert request.charge_candidates == request.electronic_candidates == ()
    assert _source_radical_electrons(mol)[0] == 1
    assert (table._presence == before[0]).all()
    assert (table._sources == before[1]).all()
    assert (table._group_status == before[2]).all()
    assert tuple(tuple(row) for row in table._stereo_facts[:len(table)]) == before[3]


def test_radical_source_spin_is_retained_and_coupling_ambiguity_is_visible(candidate_profile):
    result = _resolve(_molecule("[CH2]CO"))
    contributions = tuple(value.candidate for factor in result.candidate_space.factors for value in factor.values
                          if isinstance(value.candidate, SpinContributionCandidate)
                          and value.candidate.scope is SpinContributionScope.COMPONENT)

    assert {candidate.two_s for candidate in contributions} == {1}
    assert {path.net_charge.value for path in result.candidate_space.expanded_paths}.issuperset({2, 3})
    assert {5, 7}.issubset({_two_s(result, path) for path in result.candidate_space.expanded_paths
                           if path.net_charge.value == 3})
    assert any(reason.code == "DECISION.LOWEST_COUPLED_SPIN_DEFAULT" for reason in result.execution_decision.rationale)


def test_source_binding_identity_changes_for_declarations_without_structural_change():
    mol = _molecule()
    facts = extract_molecular_facts(mol)
    first = _molecule_state_source_identity(mol, facts)
    mol.spin_mult = 7
    assert extract_molecular_facts(mol).identity == facts.identity
    assert _molecule_state_source_identity(mol, facts) != first


def test_unknown_hydrogen_cannot_supply_required_component_spin(candidate_profile):
    mol = _molecule()
    facts = extract_molecular_facts(mol)
    facts = replace(facts, hydrogen_inventory=replace(
        facts.hydrogen_inventory, implicit_counts=(None, 0),
    ))
    request = _state_request_from_molecule(mol, facts, StateRequest())
    manifest = _coordination_screening_manifest(source_atom_formal_charges=request.atom_formal_charges)
    result = resolve_scientific_state(
        facts, request, policy=manifest.policy, provider_set=manifest.provider_set,
        rule_set=manifest.rule_set, candidate_budget=manifest.candidate_budget,
        requirements=ScientificOperationRequirements("required-spin", frozenset({ScientificQuantity.ELECTRONIC_STATE}), frozenset()),
    )

    assert _selected(result) is None


def test_supplied_spectator_ions_and_ligands_are_counted_once(candidate_profile):
    mol = hotpot.read_mol("[Na+].CC[O-].O.[Cl-]", "smi")
    center = mol.create_atom(atomic_number=63)
    for donor in tuple(atom for atom in mol.atoms if atom.atomic_number == 8):
        mol.add_bond(center, donor, bond_kind=BondKind.DATIVE)
    _commit_molecular_declarations(mol, _capture_source(mol),
                                   charge=(False, None, None), spin_mult=(False, None, None))
    result = _resolve(mol)
    selected = _selected(result)
    counting = tuple(value.candidate
                     for factor in result.candidate_space.factors for value in factor.values
                     if isinstance(value.candidate, CoordinationCountingCandidate))
    component_counts = tuple(fragment.component_identity
                             for candidate in counting for fragment in candidate.fragments)

    assert selected.net_charge.value == 2
    assert _two_s(result, selected) == 6
    assert len(component_counts) == len(set(component_counts)) == 4
    assert sorted(fragment.ionic_bookkeeping_charge
                  for candidate in counting for fragment in candidate.fragments) == [-1, -1, 0, 1]
    assert not next(trace.outcome for trace in result.rule_trace
                    if trace.outcome.rule_id == "OPTIONAL-002").applicability.matches


@pytest.mark.parametrize("center_count", (2, 3))
def test_multicenter_spin_domain_obeys_frozen_bounds_and_abstains(candidate_profile, center_count):
    mol = hotpot.read_mol(".".join("O" for _ in range(center_count)), "smi")
    for donor in tuple(mol.atoms):
        center = mol.create_atom(atomic_number=63)
        mol.add_bond(center, donor, bond_kind=BondKind.DATIVE)
    _commit_molecular_declarations(mol, _capture_source(mol),
                                   charge=(False, None, None), spin_mult=(False, None, None))
    result = _resolve(mol)
    space = result.candidate_space
    molecular_factor = next(factor for factor in space.factors
                            if factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
                            and factor.factor_key.subject_identity.startswith("molecule:"))

    assert all(len(factor.values) <= 64 for factor in space.factors)
    assert len(space.expanded_paths) <= 4096
    assert len(space.frontier) <= 1024
    assert molecular_factor.materialization_status is MaterializationStatus.RESOURCE_TRUNCATED
    assert molecular_factor.unexpanded_domain_summary is not None
    assert space.search_status is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
    assert _selected(result) is None
    assert result.execution_decision.action.value == "no_automatic_state"


def test_generic_rule_materialization_limit_is_not_silently_complete(candidate_profile):
    result = _resolve(_molecule(metal=29), budget=CandidateBudget(1, 8, 8))
    outcome = next(trace.outcome for trace in result.rule_trace
                   if trace.outcome.rule_id == "OPTIONAL-016")
    oxidation_factor = next(factor for factor in outcome.candidate_proposal.factors
                            if factor.factor_key.quantity is ScientificQuantity.OXIDATION_STATE)

    assert len(oxidation_factor.materialized_candidates) == 1
    assert oxidation_factor.materialization_status is MaterializationStatus.RESOURCE_TRUNCATED
    assert oxidation_factor.unexpanded_domain_summary.remaining_count_upper_bound > 0
    assert all(len(factor.values) <= 1 for factor in result.candidate_space.factors)
    assert len(result.candidate_space.expanded_paths) <= 8
    assert len(result.candidate_space.frontier) <= 8
    assert _selected(result) is None
    assert not result.execution_decision.static_profile_eligibility.in_domain


@pytest.mark.parametrize("ligand,metal", (("C[N+](C)(C)CCO", 63), ("[CH2]CO", 63), ("O", 30)))
def test_automatic_rule_decision_snapshot_round_trip_is_exact(candidate_profile, ligand, metal):
    result = _resolve(_molecule(ligand, metal))
    snapshot = _decision_snapshot_to_json(
        result.candidate_space, result.scientific_resolution,
        result.execution_decision, result.rule_trace,
    )
    restored = _decision_snapshot_from_json(snapshot)

    assert restored == (result.candidate_space, result.scientific_resolution,
                        result.execution_decision, result.rule_trace)
    assert _decision_snapshot_to_json(*restored) == snapshot
    assert {trace.outcome.rule_id for trace in restored[3]}.issuperset({"OPTIONAL-015", "OPTIONAL-016"})
    assert restored[2].selected_state_identity is not None


def test_absent_backing_values_do_not_change_source_or_evidence_identity():
    mol = Molecule()
    mol.create_atom(atomic_number=8, formal_charge=0, implicit_hydrogens=2)
    mol.create_atom(atomic_number=63, implicit_hydrogens=0)
    mol.add_bond(1, 0, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    identity = _molecule_state_source_identity(mol, facts)
    request = _state_request_from_molecule(mol, facts, StateRequest())
    mol._charge_value, mol._charge_source = 0, "openbabel"
    mol._spin_value, mol._spin_source = 1, "openbabel"
    field = FIELD_INDEX["formal_charge"]
    mol._atom_table._columns[field][1] = 4
    mol._atom_table._sources[1, field] = ATOM_SOURCE_CODES["explicit"]

    assert _molecule_state_source_identity(mol, facts) == identity
    assert _state_request_from_molecule(mol, facts, StateRequest()) == request
    assert request.atom_formal_charges.charges == (0, None)
