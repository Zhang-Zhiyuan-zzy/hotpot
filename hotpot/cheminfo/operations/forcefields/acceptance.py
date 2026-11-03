"""Chemical and force-field policy for accepting molecular structures."""

from __future__ import annotations

from dataclasses import replace
from typing import (
    Callable,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import numpy as np

from hotpot.cheminfo.rules.mathematics import geometry as geo
from hotpot.cheminfo.adapters.structure import geometry as geometry_adapter
from .contracts import (
    AcceptanceCheck,
    AcceptanceLevel,
    ForceFieldAcceptanceEvidence,
    ForceFieldDiagnosticValue,
    ForceFieldStage,
    ForceFieldValidationReport,
    StructureAcceptanceThresholds,
)
from hotpot.cheminfo.adapters.native.forcefields.reports import (
    NativeBondRingFinding,
    NativeRingCheckpointReport,
    NativeRingGraphScope,
)
from hotpot.cheminfo.rules.physics.forcefields.settings import _BOND_RING_MAX_SIZE
from hotpot.cheminfo.structure.forcefields import TopologyReference
from hotpot.cheminfo.adapters.structure.forcefields import (
    capture_topology,
    capture_atom_pair_acceptance_values,
    capture_bond_acceptance_values,
    capture_coordination_acceptance_values,
    capture_acceptance_bond_positions,
)
from hotpot.cheminfo.rules.physics.forcefields import acceptance as acceptance_rules
from hotpot.cheminfo.rules.physics.forcefields.contracts import (
    _BondRingAcceptanceFacts,
    _BondRingAcceptanceFinding,
)


__all__ = (
    "evaluate_structure_acceptance",
    "is_structure_accepted",
)


class _CoordinationMetrics(TypedDict):
    metal_index: int
    coordination_number: int
    donor_indices: Tuple[int, ...]
    distances: Tuple[float, ...]
    angles: Tuple[float, ...]


_NATIVE_RING_SCOPES: dict[NativeRingGraphScope, geometry_adapter.RingScope] = {
    NativeRingGraphScope.LIGAND_SKELETON: "ligand_skeleton",
    NativeRingGraphScope.FULL_GRAPH: "full_graph",
}


_UNRETURNABLE_FRAME_FAILURES = frozenset({
    "coordinate_shape",
    "finite_coordinates",
})


def _has_unreturnable_frame_failure(
    report: ForceFieldValidationReport,
) -> bool:
    """Return whether a frame cannot safely cross a workflow boundary."""
    return any(
        check.name in _UNRETURNABLE_FRAME_FAILURES
        or check.name == "topology"
        or check.name.startswith("topology_")
        for check in report.failures
    )


def _format_geometry_checks(
    prefix: str,
    checks: Tuple[AcceptanceCheck, ...],
) -> str:
    """Render failed geometry checks without discarding measured evidence."""
    details = "; ".join(
        f"{check.name}(measured={check.measured!r}, "
        f"threshold={check.threshold!r}, "
        f"atom_indices={check.atom_indices!r}, "
        f"bond_indices={check.bond_indices!r})"
        for check in checks
    )
    return f"{prefix}: {details}"


def _geometry_bond_ring_acceptance_finding(
    finding: "geometry_adapter.BondRingFinding[Ring, Bond]",
) -> _BondRingAcceptanceFinding:
    state = finding.relation.state
    return _BondRingAcceptanceFinding(
        ring_key=tuple(finding.target.ring.key),
        bond_key=tuple(finding.target.bond.key),
        state=state,
        indeterminacy_causes=(
            tuple(finding.relation.indeterminacy_causes)
            if state is geo.PiercingState.UNDETERMINED
            else ()
        ),
    )


def _geometry_bond_ring_acceptance_facts(
    report: "geometry_adapter.BondRingScreeningReport[Ring, Bond]",
) -> _BondRingAcceptanceFacts:
    return _BondRingAcceptanceFacts(
        actionable_findings=tuple(
            _geometry_bond_ring_acceptance_finding(finding)
            for finding in (*report.piercings, *report.undetermined)
        ),
        ring_scope=report.ring_scope,
        max_ring_size=report.max_ring_size,
        selected_ring_count=report.selected_ring_count,
        excluded_ring_count=report.excluded_ring_count,
        piercing_pair_count=report.piercing_pair_count,
        undetermined_pair_count=report.undetermined_pair_count,
        scan_complete=report.scan_complete,
    )


def _native_bond_ring_acceptance_finding(
    finding: NativeBondRingFinding,
) -> _BondRingAcceptanceFinding:
    state = finding.state
    return _BondRingAcceptanceFinding(
        ring_key=tuple(finding.ring_atom_indices),
        bond_key=tuple(finding.bond_key),
        state=state,
        indeterminacy_causes=(
            tuple(finding.indeterminacy_causes)
            if state is geo.PiercingState.UNDETERMINED
            else ()
        ),
    )


def _native_bond_ring_acceptance_facts(
    report: NativeRingCheckpointReport,
) -> _BondRingAcceptanceFacts:
    return _BondRingAcceptanceFacts(
        actionable_findings=tuple(
            _native_bond_ring_acceptance_finding(finding)
            for finding in report.actionable_findings
        ),
        ring_scope=_NATIVE_RING_SCOPES[report.scope],
        max_ring_size=report.maximum_actionable_ring_size,
        selected_ring_count=report.selected_ring_count,
        excluded_ring_count=report.excluded_ring_count,
        piercing_pair_count=report.piercing_pair_count,
        undetermined_pair_count=report.undetermined_pair_count,
        scan_complete=report.scan_complete,
    )


def _topology_checks(
    mol: "Molecule",
    reference: TopologyReference,
) -> Tuple[AcceptanceCheck, ...]:
    checks = acceptance_rules._topology_atom_count_checks(
        len(mol.atoms), len(reference.atoms),
    )
    if checks:
        return checks
    return acceptance_rules._topology_checks(capture_topology(mol), reference)


def _coordinate_acceptance_section(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    bool,
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return coordinate checks, finiteness, and base structure metrics."""
    checks, finite_ok = acceptance_rules._coordinate_acceptance_checks(
        coordinates, len(atoms),
    )
    if checks[1].atom_indices:
        checks = (checks[0], replace(
            checks[1],
            atom_indices=tuple(
                int(atoms[row].idx) for row in checks[1].atom_indices
            ),
        ))
    metrics: dict[str, ForceFieldDiagnosticValue] = {
        "atom_count": len(atoms),
        "bond_count": len(mol.bonds),
    }
    return tuple(checks), finite_ok, metrics


def _atom_pair_distance_acceptance_section(
    mol: "Molecule",
    level: AcceptanceLevel,
    limits: StructureAcceptanceThresholds,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return atom-pair distance checks and distance metrics."""
    distances = geometry_adapter.measure_atom_pair_distances(mol, "all")
    metrics: dict[str, ForceFieldDiagnosticValue] = {}
    if distances:
        metrics["minimum_pair_distance"] = min(
            float(distance.measurement.distance) for distance in distances
        )

    if level == "off":
        return (), metrics

    pairs = capture_atom_pair_acceptance_values(
        distances,
        capture_radii=level in ("standard", "strict"),
        overlap_tolerance=limits.overlap_tolerance,
    )
    return acceptance_rules._atom_pair_acceptance_checks(pairs, level, limits), metrics


def _bond_geometry_acceptance_section(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
    level: AcceptanceLevel,
    limits: StructureAcceptanceThresholds,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return explicit-bond geometry checks and bond-length metrics."""
    bonds = capture_bond_acceptance_values(
        mol, atoms, capture_radii=level in ("standard", "strict"),
    )
    checks, maximum_bond_length = acceptance_rules._bond_geometry_acceptance_checks(
        coordinates, bonds, level, limits,
    )
    return checks, {"maximum_bond_length": maximum_bond_length}


def _bond_ring_coordination_acceptance_section(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
    bond_ring_facts: _BondRingAcceptanceFacts,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return bond-ring checks and coordination-environment metrics."""
    metrics: dict[str, ForceFieldDiagnosticValue] = {
        "bond_ring_piercing_count": bond_ring_facts.piercing_pair_count,
        "bond_ring_undetermined_count": bond_ring_facts.undetermined_pair_count,
        "bond_ring_scan_complete": bond_ring_facts.scan_complete,
        "bond_ring_selected_ring_count": bond_ring_facts.selected_ring_count,
        "bond_ring_excluded_ring_count": bond_ring_facts.excluded_ring_count,
        "bond_ring_max_ring_size": bond_ring_facts.max_ring_size,
        "bond_ring_scope": bond_ring_facts.ring_scope,
        "coordination_environments": tuple(
            _CoordinationMetrics(
                metal_index=metal,
                coordination_number=count,
                donor_indices=donors,
                distances=distances,
                angles=angles,
            )
            for metal, count, donors, distances, angles in acceptance_rules._coordination_metrics(
                coordinates, *capture_coordination_acceptance_values(mol, atoms),
            )
        ),
    }
    return acceptance_rules._bond_ring_acceptance_checks(
        bond_ring_facts, capture_acceptance_bond_positions(mol),
    ), metrics


def _evaluate_structure_acceptance(
    mol: "Molecule",
    *,
    bond_ring_facts_provider: Callable[[], _BondRingAcceptanceFacts],
    consume_bond_ring_report: bool,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> ForceFieldValidationReport:
    """Apply chemistry and force-field acceptance policy to geometry facts."""
    if level not in ("off", "basic", "standard", "strict"):
        raise ValueError(f"Unknown structure acceptance level: {level!r}")
    if forcefield_stage not in ("candidate", "final"):
        raise ValueError(f"Unknown force-field stage: {forcefield_stage!r}")

    limits = acceptance_rules._resolve_acceptance_thresholds(thresholds)
    atoms = tuple(mol.atoms)
    coordinates = np.asarray(mol.coordinates, dtype=float)
    coordinate_checks, finite_ok, metrics = _coordinate_acceptance_section(
        mol,
        atoms,
        coordinates,
    )
    checks = list(coordinate_checks)

    if topology_reference is not None:
        checks.extend(_topology_checks(mol, topology_reference))
    checks.extend(acceptance_rules._forcefield_acceptance_checks(
        forcefield_report,
        level,
        limits,
        forcefield_stage,
    ))

    if not finite_ok:
        passed = acceptance_rules._acceptance_checks_pass(checks)
        return ForceFieldValidationReport(level, passed, tuple(checks), metrics)

    atom_pair_checks, atom_pair_metrics = (
        _atom_pair_distance_acceptance_section(mol, level, limits)
    )
    checks.extend(atom_pair_checks)
    metrics.update(atom_pair_metrics)

    if level != "off":
        bond_checks, bond_metrics = _bond_geometry_acceptance_section(
            mol,
            atoms,
            coordinates,
            level,
            limits,
        )
        checks.extend(bond_checks)
        metrics.update(bond_metrics)

    if consume_bond_ring_report:
        bond_ring_facts = bond_ring_facts_provider()
        bond_ring_checks, bond_ring_metrics = (
            _bond_ring_coordination_acceptance_section(
                mol,
                atoms,
                coordinates,
                bond_ring_facts,
            )
        )
        checks.extend(bond_ring_checks)
        metrics.update(bond_ring_metrics)

    passed = acceptance_rules._acceptance_checks_pass(checks)
    return ForceFieldValidationReport(level, passed, tuple(checks), metrics)


def evaluate_structure_acceptance_at_checkpoint(
    mol: "Molecule",
    *,
    bond_ring_report: "geometry_adapter.BondRingScreeningReport[Ring, Bond]",
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> ForceFieldValidationReport:
    """Evaluate a stage checkpoint using its existing bond-ring evidence."""
    return _evaluate_structure_acceptance(
        mol,
        bond_ring_facts_provider=lambda: (
            _geometry_bond_ring_acceptance_facts(bond_ring_report)
        ),
        consume_bond_ring_report=True,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    )


def evaluate_structure_acceptance_at_native_checkpoint(
    mol: "Molecule",
    *,
    bond_ring_report: NativeRingCheckpointReport,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> ForceFieldValidationReport:
    """Evaluate a native checkpoint without repeating bond-ring geometry."""
    return _evaluate_structure_acceptance(
        mol,
        bond_ring_facts_provider=lambda: (
            _native_bond_ring_acceptance_facts(bond_ring_report)
        ),
        consume_bond_ring_report=True,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    )


def evaluate_structure_acceptance(
    mol: "Molecule",
    *,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> ForceFieldValidationReport:
    """Apply chemistry and force-field acceptance policy to geometry facts."""
    def scan_bond_ring_relations() -> _BondRingAcceptanceFacts:
        return _geometry_bond_ring_acceptance_facts(
            geometry_adapter.screen_bond_ring_relations(
                mol,
                ring_scope="ligand_skeleton",
                max_ring_size=_BOND_RING_MAX_SIZE,
            )
        )

    return _evaluate_structure_acceptance(
        mol,
        bond_ring_facts_provider=scan_bond_ring_relations,
        consume_bond_ring_report=level in ("standard", "strict"),
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    )


def is_structure_accepted(
    mol: "Molecule",
    *,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> bool:
    """Return the result of :func:`evaluate_structure_acceptance`."""
    return evaluate_structure_acceptance(
        mol,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    ).passed
