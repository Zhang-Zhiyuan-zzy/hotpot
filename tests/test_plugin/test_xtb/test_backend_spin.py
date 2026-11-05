"""Physical molecular spin and ordinary xTB active-electron spin are distinct."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional, Protocol

import pytest

import hotpot
from hotpot.cheminfo.adapters.calculation.contracts import BackendIdentity
from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import (
    BackendDecisionSource,
    BackendExecutionEvidence,
    BackendProjectionStatus,
    BackendStateRequest,
    BackendStateValues,
    ExecutionPolicySet,
    _projection_validation_findings,
    project_state,
)
from hotpot.cheminfo.operations.calculation.scientific_state.facts import (
    extract_molecular_facts,
)
from hotpot.cheminfo.operations.calculation.scientific_state.state_resolver import (
    ScientificStateBundle,
    resolve_scientific_state,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    ScientificOperationRequirements,
    ScientificQuantity,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
from hotpot.plugins.xtb import state_projection
from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.contracts import XTBMethod
from hotpot.plugins.xtb.state_projection import XTBBackendContext


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


@pytest.fixture
def context(fake_xtb_factory: FakeXTBFactory) -> XTBBackendContext:
    return XTBBackendContext(
        probe_xtb_backend(fake_xtb_factory(), environment=dict(os.environ)),
        XTBMethod.GFN2_XTB,
    )


def _state(smiles: str, charge: int, two_s: Optional[int]) -> ScientificStateBundle:
    return resolve_scientific_state(
        extract_molecular_facts(hotpot.read_mol(smiles, "smi")),
        StateRequest(net_charge=charge, two_s=two_s),
        requirements=ScientificOperationRequirements(
            "backend-spin-test",
            frozenset({ScientificQuantity.NET_CHARGE}),
            frozenset({ScientificQuantity.ELECTRONIC_STATE}),
        ),
    )


def _history(
    state: ScientificStateBundle,
    context: XTBBackendContext,
    uhf: int,
    *,
    backend_identity: Optional[BackendIdentity] = None,
) -> BackendExecutionEvidence:
    identity = state.execution_decision.selected_state_identity
    assert identity is not None
    values = BackendStateValues(charge=0, uhf=uhf, multiplicity=None)
    return BackendExecutionEvidence(
        scientific_state_identity=identity,
        backend_identity=context.identity if backend_identity is None else backend_identity,
        requested_state=values,
        executed_state=values,
        artifact_digests=(f"sha256:active-uhf-{uhf}",),
    )


# Independently transcribed boundary values from official xTB 6.7.1
# src/pqn.f90::ncore and src/prog/main.F90's Ce--Lu override. GFN0 has
# maxElem=86 and shares this CLI initialization (src/xtb/calculator.f90).
@pytest.mark.parametrize("method", (XTBMethod.GFN0_XTB, XTBMethod.GFN1_XTB, XTBMethod.GFN2_XTB))
@pytest.mark.parametrize(
    ("atomic_number", "active_electrons"),
    (
        (1, 1), (2, 2), (3, 1), (10, 8), (11, 1), (18, 8),
        (19, 1), (29, 11), (30, 2), (36, 8), (37, 1), (47, 11),
        (48, 2), (54, 8), (55, 1), (57, 3), (58, 3), (63, 3),
        (64, 3), (71, 3), (72, 4), (79, 11), (80, 2), (86, 8),
    ),
)
def test_ordinary_gfn_active_electron_contribution(
    context: XTBBackendContext,
    method: XTBMethod,
    atomic_number: int,
    active_electrons: int,
) -> None:
    backend = replace(context, method=method)

    assert backend.effective_nuclear_charge(atomic_number) == active_electrons


@pytest.mark.parametrize("method", (XTBMethod.GFN0_XTB, XTBMethod.GFN1_XTB, XTBMethod.GFN2_XTB))
@pytest.mark.parametrize(
    ("symbol", "two_s"),
    (
        ("Ce", 1), ("Pr", 2), ("Nd", 3), ("Pm", 4), ("Sm", 5),
        ("Eu", 6), ("Gd", 7), ("Tb", 6), ("Dy", 5), ("Ho", 4),
        ("Er", 3), ("Tm", 2), ("Yb", 1), ("Lu", 0),
    ),
)
def test_trivalent_lanthanide_physical_spin_stays_out_of_active_projection(
    context: XTBBackendContext,
    method: XTBMethod,
    symbol: str,
    two_s: int,
) -> None:
    backend = replace(context, method=method)
    state = _state(f"[{symbol}+3]", 3, two_s)
    original_results = state.electronic_state_results
    original_decision = state.execution_decision

    result = project_state(state, backend)

    assert result.status is BackendProjectionStatus.PROJECTED
    projection = result.projection
    assert projection is not None
    assert projection.effective_electron_count == 0
    assert projection.uhf == 0
    assert projection.multiplicity is None
    assert projection.execution_decision.source is BackendDecisionSource.EXECUTION_DEFAULT
    assert state.electronic_state_results is original_results
    assert state.execution_decision is original_decision
    assert {
        (candidate.two_s, candidate.multiplicity)
        for evidence in state.electronic_state_results
        for candidate in evidence.candidates
    } == {(two_s, two_s + 1)}


def test_euiii_closed_ligands_keep_physical_m7_with_active_uhf0(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6)

    result = project_state(state, context)

    projection = result.projection
    assert projection is not None
    assert state.facts.nuclear_charge_sum == 114
    assert projection.effective_electron_count == 24
    assert (projection.uhf, projection.multiplicity) == (0, None)
    assert projection.execution_decision.rationale[0].code == "DEFAULT-004.LOWEST_LEGAL_UHF"
    assert {
        (candidate.two_s, candidate.multiplicity)
        for evidence in state.electronic_state_results
        for candidate in evidence.candidates
    } == {(6, 7)}
    assert _projection_validation_findings(state.facts, context, projection) == ()


@pytest.mark.parametrize("two_s", (1, 13))
def test_multiple_f_in_core_centers_do_not_subtract_physical_spins(
    context: XTBBackendContext,
    two_s: int,
) -> None:
    state = _state("[Eu+3].[Gd+3]", 6, two_s)

    projection = project_state(state, context).projection

    assert projection is not None
    assert projection.effective_electron_count == 0
    assert (projection.uhf, projection.multiplicity) == (0, None)


def test_f_in_core_with_an_odd_active_system_selects_operational_uhf1(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[N]", 3, 7)

    projection = project_state(state, context).projection

    assert projection is not None
    assert projection.effective_electron_count == 5
    assert (projection.uhf, projection.multiplicity) == (1, None)
    assert projection.execution_decision.source is BackendDecisionSource.EXECUTION_DEFAULT


def test_explicit_active_uhf_precedes_physical_spin_and_history(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6)

    projection = project_state(
        state,
        context,
        request=BackendStateRequest(uhf=2),
        execution_history=(_history(state, context, 4),),
    ).projection

    assert projection is not None
    assert (projection.uhf, projection.multiplicity) == (2, None)
    assert projection.execution_decision.source is BackendDecisionSource.EXPLICIT


@pytest.mark.parametrize(
    ("uhf", "code"),
    ((1, "HARD-013.UHF_PARITY"), (26, "HARD-013.UHF_OUT_OF_RANGE")),
)
def test_explicit_uhf_obeys_active_electron_hard_legality(
    context: XTBBackendContext,
    uhf: int,
    code: str,
) -> None:
    result = project_state(
        _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6),
        context,
        request=BackendStateRequest(uhf=uhf),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.projection is None
    assert result.findings[0].code == code


def test_f_in_core_uses_hard_valid_exact_context_history_before_default(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6)

    projection = project_state(
        state,
        context,
        execution_history=(_history(state, context, 2),),
    ).projection

    assert projection is not None
    assert (projection.uhf, projection.multiplicity) == (2, None)
    assert projection.execution_decision.source is BackendDecisionSource.EXACT_CONTEXT_HISTORY


def test_f_in_core_rejects_conflicting_exact_context_history(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6)

    result = project_state(
        state,
        context,
        execution_history=(_history(state, context, 2), _history(state, context, 4)),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.findings[0].code == "EXECUTION-001.CONFLICTING_HISTORY"


def test_history_with_wrong_context_or_illegal_active_spin_is_not_reused(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6)
    other = replace(context, method=XTBMethod.GFN1_XTB)

    projection = project_state(
        state,
        context,
        execution_history=(
            _history(state, context, 1),
            _history(state, context, 2, backend_identity=other.identity),
        ),
    ).projection

    assert projection is not None
    assert projection.uhf == 0
    assert projection.execution_decision.source is BackendDecisionSource.EXECUTION_DEFAULT


def test_f_in_core_cannot_infer_active_spin_when_execution_default_is_disabled(
    context: XTBBackendContext,
) -> None:
    result = project_state(
        _state("[Eu+3]", 3, 6),
        context,
        execution_policies=ExecutionPolicySet(("EXECUTION-001",)),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.findings[0].code == "DEFAULT-004.DISABLED"


def test_non_lanthanide_retains_highest_ms_and_uses_active_electron_count(
    context: XTBBackendContext,
) -> None:
    projection = project_state(_state("C", 0, 2), context).projection

    assert projection is not None
    assert projection.effective_electron_count == 8
    assert (projection.uhf, projection.multiplicity) == (2, 3)
    assert projection.execution_decision.source is BackendDecisionSource.SCIENTIFIC_STATE


def test_non_lanthanide_unresolved_spin_has_no_fabricated_multiplicity(
    context: XTBBackendContext,
) -> None:
    projection = project_state(_state("[N]", 0, None), context).projection

    assert projection is not None
    assert projection.effective_electron_count == 5
    assert (projection.uhf, projection.multiplicity) == (1, None)


def test_projection_validation_rejects_old_all_electron_count(
    context: XTBBackendContext,
) -> None:
    state = _state("[Eu+3].[Cl-].[Cl-].[Cl-]", 0, 6)
    projection = project_state(state, context).projection
    assert projection is not None
    old_count = replace(projection, effective_electron_count=114)

    findings = _projection_validation_findings(state.facts, context, old_count)

    assert [finding.code for finding in findings] == ["HARD-013.ELECTRON_COUNT_MISMATCH"]


def test_backend_identity_binds_active_spin_convention(
    context: XTBBackendContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = context.identity

    monkeypatch.setattr(
        state_projection,
        "_GFN_SPIN_PROJECTION_DEFINITION",
        "different-active-spin-convention",
    )

    assert context.identity != original


def test_backend_identity_binds_electron_definition_and_version(
    context: XTBBackendContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = context.identity
    other_version = replace(context.backend_info, version="6.7.0")
    assert replace(context, backend_info=other_version).identity != original

    monkeypatch.setattr(
        state_projection,
        "_GFN_VALENCE_ELECTRON_DEFINITION",
        "different-electron-convention",
    )

    assert context.identity != original


def test_scientific_spin_is_checked_against_active_electron_range(
    context: XTBBackendContext,
) -> None:
    # Methane has ten physical electrons, but ordinary GFN has eight active.
    result = project_state(_state("C", 0, 10), context)

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.findings[0].code == "HARD-013.UHF_OUT_OF_RANGE"


def test_gfnff_remains_charge_only_for_a_physical_lanthanide_state(
    context: XTBBackendContext,
) -> None:
    backend = replace(context, method=XTBMethod.GFNFF)
    state = _state("[Eu+3]", 3, 6)

    projection = project_state(state, backend).projection

    assert projection is not None
    assert (projection.charge, projection.uhf, projection.multiplicity) == (3, None, None)
    assert projection.projection_convention == "charge_only"


def test_workflow_validation_consumes_the_corrected_electron_definition(
    context: XTBBackendContext,
) -> None:
    from hotpot.plugins.xtb.contracts import XTBInputError
    from hotpot.plugins.xtb.workflow import _validate_projection

    mol = hotpot.read_mol("[Eu+3].[Cl-].[Cl-].[Cl-]", "smi")
    state = resolve_scientific_state(
        extract_molecular_facts(mol),
        StateRequest(net_charge=0, two_s=6),
        requirements=context.scientific_operation_requirements,
    )
    projection = project_state(state, context).projection
    assert projection is not None

    _validate_projection(mol, projection, context)
    with pytest.raises(XTBInputError, match="HARD-013.ELECTRON_COUNT_MISMATCH"):
        _validate_projection(
            mol,
            replace(projection, effective_electron_count=114),
            context,
        )


@pytest.mark.parametrize("symbol", ("Fr", "Am"))
def test_unsupported_element_validation_returns_named_findings(
    context: XTBBackendContext,
    symbol: str,
) -> None:
    from hotpot.plugins.xtb.contracts import XTBInputError
    from hotpot.plugins.xtb.workflow import _validate_projection

    projection = project_state(_state("[Eu+3]", 3, 6), context).projection
    assert projection is not None
    molecule = hotpot.read_mol(f"[{symbol}]", "smi")
    facts = extract_molecular_facts(molecule)

    findings = _projection_validation_findings(facts, context, projection)

    assert "HARD-008.UNSUPPORTED_ELEMENT" in {finding.code for finding in findings}
    assert "HARD-013.ELECTRON_COUNT_MISMATCH" not in {finding.code for finding in findings}
    with pytest.raises(XTBInputError, match="HARD-008.UNSUPPORTED_ELEMENT"):
        _validate_projection(molecule, projection, context)
