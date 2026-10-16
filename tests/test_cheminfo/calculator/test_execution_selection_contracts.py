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
