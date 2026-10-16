"""End-to-end backend state projection tests."""

from typing import Optional

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    BackendIdentity,
    ScientificOperationRequirements,
    ScientificQuantity,
)
from hotpot.cheminfo.calculator.execution_selection import (
    BackendDecisionSource,
    BackendExecutionEvidence,
    BackendProjectionStatus,
    BackendStateRequest,
    BackendStateValues,
    ExecutionPolicySet,
    project_state,
    resolve_and_project_state,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
    resolve_scientific_state,
)


class _AllElectronBackend:
    def __init__(
        self,
        convention: str = "pure_spin_highest_ms",
        *,
        method_supported: bool = True,
        max_atomic_number: int = 118,
    ) -> None:
        self._convention = convention
        self._method_supported = method_supported
        self._max_atomic_number = max_atomic_number

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(f"test:{self._convention}")

    @property
    def scientific_operation_requirements(
        self,
    ) -> ScientificOperationRequirements:
        return ScientificOperationRequirements(
            "test-backend",
            frozenset({ScientificQuantity.NET_CHARGE}),
            frozenset({ScientificQuantity.ELECTRONIC_STATE}),
        )

    @property
    def projection_convention(self) -> str:
        return self._convention

    @property
    def consumes_uhf(self) -> bool:
        return self._convention != "charge_only"

    @property
    def method_supported(self) -> bool:
        return self._method_supported

    def supports_atomic_number(self, atomic_number: int) -> bool:
        return 0 < atomic_number <= self._max_atomic_number

    @staticmethod
    def effective_nuclear_charge(atomic_number: int) -> int:
        return atomic_number


class _EffectiveElectronBackend(_AllElectronBackend):
    def __init__(self, effective_nuclear_charge: int) -> None:
        super().__init__()
        self._effective_nuclear_charge = effective_nuclear_charge

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(f"test:ecp:{self._effective_nuclear_charge}")

    def effective_nuclear_charge(self, atomic_number: int) -> int:
        if atomic_number == 54:
            return self._effective_nuclear_charge
        return atomic_number


def _state(
    smiles: str = "C",
    *,
    charge: int = 0,
    two_s: Optional[int] = None,
):
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    required = {ScientificQuantity.NET_CHARGE}
    if two_s is not None:
        required.add(ScientificQuantity.ELECTRONIC_STATE)
    return resolve_scientific_state(
        facts,
        StateRequest(net_charge=charge, two_s=two_s),
        requirements=ScientificOperationRequirements(
            "projection-test",
            frozenset(required),
            frozenset(),
        ),
    )


def test_projection_requires_a_selected_scientific_state() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))
    analysis = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
    )

    result = project_state(analysis, _AllElectronBackend())

    assert result.status is BackendProjectionStatus.NOT_PROJECTED
    assert result.projection is None
    assert result.findings[0].code == "HARD-008.SCIENTIFIC_STATE_NOT_SELECTED"


def test_highest_ms_projection_maps_selected_pure_spin_to_uhf() -> None:
    result = project_state(_state(two_s=2), _AllElectronBackend())

    assert result.status is BackendProjectionStatus.PROJECTED
    assert result.projection is not None
    assert result.projection.uhf == 2
    assert result.projection.multiplicity == 3
    assert (
        result.projection.execution_decision.source
        is BackendDecisionSource.SCIENTIFIC_STATE
    )


def test_explicit_uhf_must_match_selected_pure_spin() -> None:
    result = project_state(
        _state(two_s=0),
        _AllElectronBackend(),
        request=BackendStateRequest(uhf=2),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.projection is None
    assert result.findings[0].code.endswith("SCIENTIFIC_SPIN_CONFLICT")


def test_default_004_supplies_only_an_execution_uhf() -> None:
    state = _state("[N]", two_s=None)
    result = project_state(state, _AllElectronBackend())

    assert result.status is BackendProjectionStatus.PROJECTED
    assert result.projection is not None
    assert result.projection.effective_electron_count == 7
    assert result.projection.uhf == 1
    assert result.projection.multiplicity is None
    assert state.execution_decision == _state("[N]", two_s=None).execution_decision


def test_disabling_default_004_leaves_unresolved_uhf_unprojected() -> None:
    result = project_state(
        _state("[N]", two_s=None),
        _AllElectronBackend(),
        execution_policies=ExecutionPolicySet(("EXECUTION-001",)),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.findings[0].code == "DEFAULT-004.DISABLED"


def test_exact_context_history_precedes_default_004() -> None:
    state = _state("[N]", two_s=None)
    state_identity = state.execution_decision.selected_state_identity
    assert state_identity is not None
    backend = _AllElectronBackend()
    values = BackendStateValues(charge=0, uhf=3, multiplicity=4)
    evidence = BackendExecutionEvidence(
        scientific_state_identity=state_identity,
        backend_identity=backend.identity,
        requested_state=values,
        executed_state=values,
        artifact_digests=("sha256:test",),
    )

    result = project_state(
        state,
        backend,
        execution_history=(evidence,),
    )

    assert result.projection is not None
    assert result.projection.uhf == 3
    assert (
        result.projection.execution_decision.source
        is BackendDecisionSource.EXACT_CONTEXT_HISTORY
    )


def test_charge_only_projection_rejects_explicit_uhf() -> None:
    backend = _AllElectronBackend("charge_only")
    result = project_state(
        _state(two_s=None),
        backend,
        request=BackendStateRequest(uhf=0),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.findings[0].code == "HARD-008.UHF_NOT_CONSUMED"


def test_backend_support_is_reported_without_fabricating_a_projection() -> None:
    method_result = project_state(
        _state(two_s=0),
        _AllElectronBackend(method_supported=False),
    )
    element_result = project_state(
        _state("[Am]", two_s=1),
        _AllElectronBackend(max_atomic_number=86),
    )

    assert method_result.status is BackendProjectionStatus.BACKEND_UNSUPPORTED
    assert method_result.projection is None
    assert method_result.findings[0].code == "HARD-008.UNSUPPORTED_METHOD"
    assert element_result.status is BackendProjectionStatus.BACKEND_UNSUPPORTED
    assert element_result.projection is None
    assert element_result.findings[0].related_identities == ("95",)


def test_combined_workflow_resolves_backend_requirements_before_projection() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))

    result = resolve_and_project_state(
        facts,
        _AllElectronBackend(),
        StateRequest(net_charge=0),
    )

    assert result.scientific_state.execution_decision.selected_state_identity
    assert (
        result.backend_projection.status
        is BackendProjectionStatus.PROJECTED
    )
    assert result.backend_projection.projection is not None
    assert result.backend_projection.projection.uhf == 0


def test_ecp_projection_uses_backend_effective_electrons() -> None:
    result = project_state(
        _state("[Xe]", two_s=0),
        _EffectiveElectronBackend(effective_nuclear_charge=8),
    )

    assert result.status is BackendProjectionStatus.PROJECTED
    assert result.projection is not None
    assert result.projection.effective_electron_count == 8
    assert result.projection.uhf == 0


def test_ecp_projection_rejects_molecularly_legal_out_of_range_uhf() -> None:
    result = project_state(
        _state("[Xe]", two_s=10),
        _EffectiveElectronBackend(effective_nuclear_charge=8),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.projection is None
    assert result.findings[0].code == "HARD-013.UHF_OUT_OF_RANGE"


def test_ecp_projection_rejects_molecularly_legal_effective_parity() -> None:
    result = project_state(
        _state("[Xe]", two_s=2),
        _EffectiveElectronBackend(effective_nuclear_charge=7),
    )

    assert result.status is BackendProjectionStatus.PROJECTION_CONFLICT
    assert result.projection is None
    assert result.findings[0].code == "HARD-013.UHF_PARITY"
