"""Shared scientific-state resolution for xTB CLI and pipeline consumers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Union

from hotpot.cheminfo.structure.snapshots import MolecularFactsIdentity
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import (
    BackendProjectionResult,
    BackendProjectionStatus,
    BackendStateProjection,
    BackendStateRequest,
    project_state,
    resolve_and_project_state,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import ScientificStateIdentity
from hotpot.cheminfo.operations.calculation.scientific_state.state_resolver import (
    ScientificStateBundle,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
from hotpot.cheminfo.operations.calculation.scientific_state.facts import (
    extract_molecular_facts, _state_request_from_molecule,
    _source_radical_electrons, _molecule_state_source_identity,
)
from hotpot.cheminfo.operations.calculation.scientific_state.manifest import (
    _coordination_screening_manifest,
)
from hotpot.cheminfo.rules.chemistry.elements import _is_metal
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import ScientificFinding
from hotpot.cheminfo.structure.molecule import Molecule

from .backend import probe_xtb_backend
from .contracts import XTBInputError, XTBMethod
from .state_projection import XTBBackendContext


_ExecutablePath = Union[str, Path]


@dataclass(frozen=True)
class ResolvedXTBState:
    """One scientific bundle and its executable xTB projection."""

    scientific_state: ScientificStateBundle
    projection: BackendStateProjection


class XTBStateResolutionError(XTBInputError):
    """Internal typed boundary between science and backend projection."""

    sample_failure_code: Optional[str] = "XTB-STATE-UNAVAILABLE"

    def __init__(
        self,
        scientific_state: ScientificStateBundle,
        projection_result: BackendProjectionResult,
    ) -> None:
        self.scientific_state = scientific_state
        self.projection_result = projection_result
        details = "; ".join(
            f"{finding.code}: {finding.message}"
            for finding in projection_result.findings
        )
        super().__init__(
            "xTB state projection ended as "
            f"{projection_result.status.value}"
            + ("" if not details else f": {details}")
        )


def resolve_xtb_state(
    mol: Molecule,
    method: XTBMethod,
    *,
    executable: Optional[_ExecutablePath],
    environment: Mapping[str, str],
    charge: Optional[int],
    uhf: Optional[int],
    policy: Optional[StatePolicy],
    carried_state: Optional[ScientificStateBundle] = None,
    expected_scientific_identity: Optional[ScientificStateIdentity] = None,
    expected_facts_identity: Optional[MolecularFactsIdentity] = None,
) -> ResolvedXTBState:
    """Resolve once and project into one probed xTB method context."""
    backend_info = probe_xtb_backend(executable, environment=environment)
    backend = XTBBackendContext(backend_info, method)
    facts = extract_molecular_facts(mol)
    backend_request = BackendStateRequest(uhf=uhf)
    request = _state_request_from_molecule(mol, facts, StateRequest(net_charge=charge))
    source_identity = _molecule_state_source_identity(mol, facts)
    effective_policy = policy
    if effective_policy is None:
        effective_policy = (
            StatePolicy.COORDINATION_SCREENING_V1
            if any(bond.kind == "coordination"
                   and any(_is_metal(facts.atomic_numbers[index]) for index in bond.atom_indices)
                   for bond in facts.semantic_bonds)
            else StatePolicy.STRICT
        )
    configuration = {}
    profile_binding = effective_policy.value
    if effective_policy is StatePolicy.COORDINATION_SCREENING_V1:
        manifest = _coordination_screening_manifest(
            source_atom_formal_charges=request.atom_formal_charges,
            source_component_charges=request.component_charges,
            source_radical_electrons=_source_radical_electrons(mol),
            source_evidence_identity=source_identity,
        )
        profile_binding = manifest.profile_content_digest
        configuration = {
            "provider_set": manifest.provider_set, "rule_set": manifest.rule_set,
            "candidate_budget": manifest.candidate_budget,
        }
    if carried_state is not None and carried_state.facts.identity != facts.identity:
        raise XTBInputError(
            "Carried scientific state does not match the current molecule"
        )
    source_binding_matches = carried_state is not None and any(
        finding.code == "SOURCE.MOLECULE_STATE_BINDING"
        and finding.related_identities == (source_identity, profile_binding)
        for result in carried_state.net_charge_results for finding in result.findings
    )
    if (carried_state is not None and charge is None and source_binding_matches
            and carried_state.execution_decision.static_profile_eligibility.profile_id == effective_policy.value):
        scientific_state = carried_state
        projection_result = project_state(
            scientific_state,
            backend,
            request=backend_request,
        )
    else:
        result = resolve_and_project_state(
            facts,
            backend,
            request,
            policy=effective_policy,
            backend_request=backend_request,
            **configuration,
        )
        scientific_state = result.scientific_state
        first_charge, *other_charges = scientific_state.net_charge_results
        source_binding = ScientificFinding(
            "SOURCE.MOLECULE_STATE_BINDING",
            "Resolution binds current supplied charge, spin, formal-charge presence and radical evidence.",
            (source_identity, profile_binding),
        )
        scientific_state = replace(
            scientific_state,
            net_charge_results=(replace(first_charge, findings=first_charge.findings + (source_binding,)),
                                *other_charges),
        )
        projection_result = result.backend_projection

    if (
        projection_result.status is not BackendProjectionStatus.PROJECTED
        or projection_result.projection is None
    ):
        raise XTBStateResolutionError(scientific_state, projection_result)

    projection = projection_result.projection
    if (
        expected_facts_identity is not None
        and expected_facts_identity != facts.identity
    ):
        raise XTBInputError(
            "Input stream state provenance does not match the current molecule"
        )
    if (
        expected_scientific_identity is not None
        and expected_scientific_identity != projection.scientific_state_identity
    ):
        raise XTBInputError(
            "Resolved scientific state does not match input stream provenance"
        )
    return ResolvedXTBState(scientific_state, projection)
