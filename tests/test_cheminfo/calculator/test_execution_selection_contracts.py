"""Backend execution-selection contract tests."""

from dataclasses import FrozenInstanceError

import pytest

from hotpot.cheminfo.calculator.contracts import BackendIdentity
from hotpot.cheminfo.calculator.execution_selection import (
    BackendExecutability,
    BackendProjectionResult,
    BackendProjectionStatus,
    BackendStateProjection,
    BackendStateRequest,
    ExecutionPolicySet,
    _effective_electron_count,
    _lowest_legal_uhf,
    _uhf_legality_finding,
    _validated_execution_policy_ids,
)


def test_projection_result_requires_exactly_one_success_projection() -> None:
    backend_identity = BackendIdentity("xtb-context")

    with pytest.raises(ValueError, match="Only PROJECTED"):
        BackendProjectionResult(
            status=BackendProjectionStatus.PROJECTED,
            backend_identity=backend_identity,
            request=BackendStateRequest(),
            projection=None,
            findings=(),
        )

    with pytest.raises(ValueError, match="Only PROJECTED"):
        BackendProjectionResult(
            status=BackendProjectionStatus.PROJECTION_CONFLICT,
            backend_identity=backend_identity,
            request=BackendStateRequest(),
            projection=_projection(backend_identity),
            findings=(),
        )


def test_projection_result_rejects_mismatched_backend_identity() -> None:
    with pytest.raises(ValueError, match="same backend context"):
        BackendProjectionResult(
            status=BackendProjectionStatus.PROJECTED,
            backend_identity=BackendIdentity("requested-context"),
            request=BackendStateRequest(),
            projection=_projection(BackendIdentity("other-context")),
            findings=(),
        )


def test_execution_selection_contracts_are_immutable() -> None:
    request = BackendStateRequest(uhf=2)

    with pytest.raises(FrozenInstanceError):
        request.uhf = 0


def test_effective_electron_count_includes_stored_implicit_hydrogen() -> None:
    from hotpot.cheminfo.calculator.contracts import (
        ComponentIdentity,
        HydrogenInventory,
        MolecularFacts,
        MolecularFactsIdentity,
        ScientificOperationRequirements,
    )

    class AllElectronBackend:
        identity = BackendIdentity("all-electron")
        scientific_operation_requirements = ScientificOperationRequirements(
            "test",
            frozenset(),
            frozenset(),
        )
        projection_convention = "pure_spin_highest_ms"

        @staticmethod
        def effective_nuclear_charge(atomic_number: int) -> int:
            return atomic_number

    facts = MolecularFacts(
        identity=MolecularFactsIdentity("methane"),
        geometry_digest=None,
        atomic_numbers=(6,),
        mass_numbers=(None,),
        atom_map=(0,),
        source_components=(ComponentIdentity("source", (0,)),),
        covalent_components=(ComponentIdentity("covalent", (0,)),),
        hydrogen_inventory=HydrogenInventory((0,), (4,)),
        semantic_bonds=(),
        coordinates=None,
        nuclear_charge_sum=10,
    )

    assert _effective_electron_count(facts, AllElectronBackend(), 0) == 10


@pytest.mark.parametrize(
    ("electron_count", "uhf", "finding_code"),
    (
        (-1, 0, "HARD-013.NEGATIVE_EFFECTIVE_ELECTRON_COUNT"),
        (4, 5, "HARD-013.UHF_OUT_OF_RANGE"),
        (4, -1, "HARD-013.UHF_OUT_OF_RANGE"),
        (4, 1, "HARD-013.UHF_PARITY"),
    ),
)
def test_hard_013_rejects_illegal_backend_states(
    electron_count: int,
    uhf: int,
    finding_code: str,
) -> None:
    finding = _uhf_legality_finding(electron_count, uhf)

    assert finding is not None
    assert finding.code == finding_code


@pytest.mark.parametrize(("electron_count", "expected"), ((4, 0), (5, 1)))
def test_default_004_returns_lowest_parity_uhf(
    electron_count: int,
    expected: int,
) -> None:
    assert _lowest_legal_uhf(electron_count) == expected
    assert _uhf_legality_finding(electron_count, expected) is None


def test_execution_policy_set_rejects_unknown_or_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        _validated_execution_policy_ids(
            ExecutionPolicySet(("DEFAULT-004", "DEFAULT-004"))
        )
    with pytest.raises(ValueError, match="Unknown"):
        _validated_execution_policy_ids(ExecutionPolicySet(("UNKNOWN",)))


def _projection(backend_identity: BackendIdentity) -> BackendStateProjection:
    from hotpot.cheminfo.calculator.contracts import MolecularFactsIdentity
    from hotpot.cheminfo.calculator.execution_selection import (
        BackendDecisionSource,
        BackendExecutionDecision,
        BackendStateValues,
    )
    from hotpot.cheminfo.calculator.state_composition import (
        ScientificStateIdentity,
    )

    state_values = BackendStateValues(charge=0, uhf=0, multiplicity=1)
    return BackendStateProjection(
        scientific_state_identity=ScientificStateIdentity("state"),
        facts_identity=MolecularFactsIdentity("facts"),
        backend_identity=backend_identity,
        effective_electron_count=10,
        charge=0,
        uhf=0,
        multiplicity=1,
        projection_convention="pure_spin_highest_ms",
        execution_decision=BackendExecutionDecision(
            backend_identity=backend_identity,
            requested_state=state_values,
            selected_state=state_values,
            source=BackendDecisionSource.EXPLICIT,
            rationale=(),
        ),
        executability=BackendExecutability.SUPPORTED,
    )
