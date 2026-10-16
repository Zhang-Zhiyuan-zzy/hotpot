"""Shared scientific-state resolution for xTB CLI and pipeline consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

from hotpot.cheminfo.calculator.contracts import (
    MolecularFactsIdentity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.execution_selection import (
    BackendProjectionStatus,
    BackendStateProjection,
    BackendStateRequest,
    project_state,
    resolve_and_project_state,
)
from hotpot.cheminfo.calculator.state_composition import ScientificStateIdentity
from hotpot.cheminfo.calculator.state_resolver import (
    ScientificStateBundle,
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.core import Molecule

from .backend import probe_xtb_backend
from .contracts import XTBInputError, XTBMethod
from .state_projection import XTBBackendContext


_ExecutablePath = Union[str, Path]


@dataclass(frozen=True)
class ResolvedXTBState:
    """One scientific bundle and its executable xTB projection."""

    scientific_state: ScientificStateBundle
    projection: BackendStateProjection


def resolve_xtb_state(
    mol: Molecule,
    method: XTBMethod,
    *,
    executable: Optional[_ExecutablePath],
    environment: Mapping[str, str],
    charge: Optional[int],
    uhf: Optional[int],
    policy: StatePolicy,
    carried_state: Optional[ScientificStateBundle] = None,
    expected_scientific_identity: Optional[ScientificStateIdentity] = None,
    expected_facts_identity: Optional[MolecularFactsIdentity] = None,
) -> ResolvedXTBState:
    """Resolve once and project into one probed xTB method context."""
    backend_info = probe_xtb_backend(executable, environment=environment)
    backend = XTBBackendContext(backend_info, method)
    facts = extract_molecular_facts(mol)
    backend_request = BackendStateRequest(uhf=uhf)

    if carried_state is not None and charge is None:
        if carried_state.facts.identity != facts.identity:
            raise XTBInputError(
                "Carried scientific state does not match the current molecule"
            )
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
            StateRequest(net_charge=charge),
            policy=policy,
            backend_request=backend_request,
        )
        scientific_state = result.scientific_state
        projection_result = result.backend_projection

    if (
        projection_result.status is not BackendProjectionStatus.PROJECTED
        or projection_result.projection is None
    ):
        details = "; ".join(
            f"{finding.code}: {finding.message}"
            for finding in projection_result.findings
        )
        raise XTBInputError(
            f"xTB state projection ended as {projection_result.status.value}"
            + ("" if not details else f": {details}")
        )

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
