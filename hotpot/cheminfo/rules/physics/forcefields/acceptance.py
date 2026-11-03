"""Pure value comparisons for force-field structure acceptance."""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from ...mathematics import geometry as geo
from ....structure.forcefields import TopologyReference
from .contracts import (
    AcceptanceCheck, AcceptanceLevel, ForceFieldAcceptanceEvidence,
    ForceFieldStage, StructureAcceptanceThresholds,
    _AtomPairAcceptanceIssue, _BondRingAcceptanceFacts,
)


def _acceptance_checks_pass(checks: Sequence[AcceptanceCheck]) -> bool:
    """Return whether no failed error-level acceptance check is present."""
    return not any(
        not check.passed and check.severity == "error"
        for check in checks
    )


def _resolve_acceptance_thresholds(
    thresholds: Optional[StructureAcceptanceThresholds],
) -> StructureAcceptanceThresholds:
    return thresholds if thresholds is not None else StructureAcceptanceThresholds()


def _forcefield_acceptance_checks(
    report: Optional[ForceFieldAcceptanceEvidence],
    level: AcceptanceLevel,
    thresholds: StructureAcceptanceThresholds,
    stage: ForceFieldStage,
) -> Tuple[AcceptanceCheck, ...]:
    if report is None:
        if level == "strict":
            return (AcceptanceCheck(
                name="forcefield_report",
                passed=False,
                measured=None,
                threshold="complete force-field report",
                message="Strict structure validation requires force-field diagnostics",
            ),)
        return ()

    checks = []
    setup_succeeded = report.get("setup_succeeded")
    checks.append(AcceptanceCheck(
        name="forcefield_setup",
        passed=setup_succeeded is not None and bool(setup_succeeded),
        measured=setup_succeeded,
        threshold=True,
        message="Force-field setup must succeed",
    ))

    required_finite_fields = ["final_energy"]
    if stage == "final":
        required_finite_fields.extend(("rms_gradient", "max_gradient"))
    for field_name in required_finite_fields:
        value = report.get(field_name)
        finite = value is not None and bool(np.isfinite(value))
        checks.append(AcceptanceCheck(
            name=f"finite_{field_name}",
            passed=finite,
            measured=None if value is None else float(value),
            threshold="finite",
            message=f"{field_name.replace('_', ' ')} must be finite",
        ))

    if level in ("basic", "standard", "strict"):
        exploded = report.get("exploded")
        checks.append(AcceptanceCheck(
            name="backend_explosion",
            passed=exploded is not None and not bool(exploded),
            measured=exploded,
            threshold=False,
            message="The force-field backend must report a non-exploded structure",
        ))

    if stage == "final" and level in ("standard", "strict"):
        converged = report.get("converged")
        if converged is not None or level == "strict":
            checks.append(AcceptanceCheck(
                name="forcefield_convergence",
                passed=converged is not None and bool(converged),
                severity="error" if level == "strict" else "warning",
                measured=converged,
                threshold=True,
                message="The force-field backend did not report convergence",
            ))

    if stage == "final" and level == "strict":
        gradient_limits = (
            ("rms_gradient", thresholds.strict_rms_gradient),
            ("max_gradient", thresholds.strict_max_gradient),
        )
        for field_name, limit in gradient_limits:
            value = report.get(field_name)
            if value is not None and np.isfinite(value):
                checks.append(AcceptanceCheck(
                    name=field_name,
                    passed=float(value) <= limit,
                    measured=float(value),
                    threshold=limit,
                    message=(
                        f"{field_name.replace('_', ' ')} exceeds the strict limit"
                    ),
                ))

        segment_epochs_completed = report.get("segment_epochs_completed")
        converged = bool(report.get("converged"))
        no_history_required = (
            segment_epochs_completed is not None
            and int(segment_epochs_completed) == 1
            and converged
        )
        stability_checks = (
            ("energy_changes", thresholds.strict_energy_change),
            ("max_displacements", thresholds.strict_max_displacement),
        )
        stability_observations = []
        for field_name, limit in stability_checks:
            history = report.get(field_name)
            values = () if history is None else tuple(history)
            recent = values[-thresholds.strict_stability_window:]
            value = max(recent) if recent else None
            stability_observations.append(len(recent))
            checks.append(AcceptanceCheck(
                name=field_name.removesuffix("s"),
                passed=no_history_required or (
                    value is not None
                    and np.isfinite(value)
                    and float(value) <= limit
                ),
                measured=None if value is None else float(value),
                threshold=limit,
                message=(
                    f"{field_name.replace('_', ' ')} do not satisfy the strict limit"
                ),
            ))

        observations = min(stability_observations)
        if segment_epochs_completed is None:
            epochs_completed = report.get("epochs_completed")
            required_observations = (
                min(thresholds.strict_stability_window, int(epochs_completed))
                if epochs_completed is not None
                else thresholds.strict_stability_window
            )
        else:
            required_observations = min(
                thresholds.strict_stability_window,
                max(int(segment_epochs_completed) - 1, 0),
            )
        checks.append(AcceptanceCheck(
            name="stability_observations",
            passed=(
                no_history_required
                or required_observations > 0
                and observations >= required_observations
            ),
            measured=observations,
            threshold=required_observations,
            message="Strict validation requires a stable multi-epoch history",
        ))
    return tuple(checks)


def _topology_atom_count_checks(actual_count: int, original_count: int) -> Tuple[AcceptanceCheck, ...]:
    if actual_count < original_count:
        return (AcceptanceCheck(
            name="topology_atom_count",
            passed=False,
            measured=actual_count,
            threshold=f">={original_count}",
            message="Original atoms were removed",
        ),)
    return ()


def _topology_checks(
    candidate: TopologyReference,
    reference: TopologyReference,
) -> Tuple[AcceptanceCheck, ...]:
    atoms = candidate.atoms
    checks = []
    original_count = len(reference.atoms)

    for signature, atom in zip(reference.atoms, atoms[:original_count]):
        measured = (atom.atom_id, atom.atomic_number, atom.formal_charge)
        expected = (
            signature.atom_id,
            signature.atomic_number,
            signature.formal_charge,
        )
        if measured != expected:
            checks.append(AcceptanceCheck(
                name="topology_atom_identity",
                passed=False,
                measured=measured,
                threshold=expected,
                atom_indices=(signature.index,),
                message="An original atom identity or formal charge changed",
            ))

    added_indices = set(range(original_count, len(atoms)))
    if added_indices and not reference.allow_added_hydrogens:
        checks.append(AcceptanceCheck(
            name="topology_added_atoms",
            passed=False,
            measured=len(added_indices),
            threshold=0,
            atom_indices=tuple(sorted(added_indices)),
            message="Additional atoms are not allowed by this topology reference",
        ))
    elif added_indices:
        non_hydrogens = tuple(
            index
            for index in added_indices
            if int(atoms[index].atomic_number) != 1
        )
        if non_hydrogens:
            checks.append(AcceptanceCheck(
                name="topology_added_atoms",
                passed=False,
                measured=tuple(
                    int(atoms[index].atomic_number)
                    for index in non_hydrogens
                ),
                threshold="hydrogen only",
                atom_indices=non_hydrogens,
                message="Only hydrogen atoms may be added during preparation",
            ))

    candidate_bonds = {
        signature.atom_indices: signature for signature in candidate.bonds
    }
    reference_bonds = {
        signature.atom_indices: signature for signature in reference.bonds
    }

    for endpoints, expected in reference_bonds.items():
        measured = candidate_bonds.get(endpoints)
        if measured != expected:
            checks.append(AcceptanceCheck(
                name="topology_original_bond",
                passed=False,
                measured=measured,
                threshold=expected,
                atom_indices=endpoints,
                message="An original bond was removed or changed",
            ))

    added_bonds = set(candidate_bonds).difference(reference_bonds)
    invalid_added_bonds = tuple(sorted(
        endpoints
        for endpoints in added_bonds
        if not reference.allow_added_hydrogens
        or sum(endpoint in added_indices for endpoint in endpoints) != 1
    ))
    for endpoints in invalid_added_bonds:
        checks.append(AcceptanceCheck(
            name="topology_added_bond",
            passed=False,
            measured=endpoints,
            threshold="one added H endpoint",
            atom_indices=endpoints,
            message="Only new X-H bonds may be added during preparation",
        ))

    if reference.allow_added_hydrogens:
        degree = {index: 0 for index in added_indices}
        for endpoints in added_bonds:
            for endpoint in endpoints:
                if endpoint in degree:
                    degree[endpoint] += 1
        invalid_hydrogens = tuple(
            index for index, count in sorted(degree.items()) if count != 1
        )
        if invalid_hydrogens:
            checks.append(AcceptanceCheck(
                name="topology_added_hydrogen_degree",
                passed=False,
                measured=tuple(
                    degree[index] for index in invalid_hydrogens
                ),
                threshold=1,
                atom_indices=invalid_hydrogens,
                message="Each added hydrogen must have exactly one new bond",
            ))

    if not checks:
        checks.append(AcceptanceCheck(
            name="topology",
            passed=True,
            measured=(len(atoms), len(candidate_bonds)),
            threshold=(original_count, len(reference_bonds)),
            message="Original topology is preserved",
        ))
    return tuple(checks)


def _overlap_issues(
    distances: Sequence[Tuple[Tuple[int, int], bool, float, Optional[float]]],
    tolerance: float,
) -> Tuple[_AtomPairAcceptanceIssue, ...]:
    return tuple(
        _AtomPairAcceptanceIssue(
            kind="overlap",
            atom_indices=distance[0],
            distance=float(distance[2]),
            threshold=float(tolerance),
        )
        for distance in distances
        if distance[2] <= tolerance
    )


def _too_close_issues(
    distances: Sequence[Tuple[Tuple[int, int], bool, float, Optional[float]]],
    *,
    minimum_distance: float,
    covalent_radius_scale: Optional[float],
    pair_scope: Literal["all", "bonded", "nonbonded"],
    include_overlaps: bool,
    overlap_tolerance: float,
) -> Tuple[_AtomPairAcceptanceIssue, ...]:
    issues = []
    for distance in distances:
        if pair_scope == "bonded" and not distance[1]:
            continue
        if pair_scope == "nonbonded" and distance[1]:
            continue
        measured = float(distance[2])
        if not include_overlaps and measured <= overlap_tolerance:
            continue
        threshold = minimum_distance
        if covalent_radius_scale is not None:
            threshold = max(
                threshold,
                covalent_radius_scale * distance[3],
            )
        if measured < threshold:
            issues.append(_AtomPairAcceptanceIssue(
                kind="too_close",
                atom_indices=distance[0],
                distance=measured,
                threshold=float(threshold),
            ))
    return tuple(issues)


def _atom_pair_acceptance_checks(
    distances: Sequence[Tuple[Tuple[int, int], bool, float, Optional[float]]],
    level: AcceptanceLevel,
    limits: StructureAcceptanceThresholds,
) -> Tuple[AcceptanceCheck, ...]:
    """Return immutable atom-pair checks from detached measurements."""
    if level == "off":
        return ()

    checks = []
    overlaps = _overlap_issues(distances, limits.overlap_tolerance)
    if overlaps:
        checks.extend(AcceptanceCheck(
            name="atom_overlap",
            passed=False,
            measured=issue.distance,
            threshold=issue.threshold,
            atom_indices=issue.atom_indices,
            message="Two atoms occupy indistinguishable coordinates",
        ) for issue in overlaps)
    else:
        checks.append(AcceptanceCheck(
            name="atom_overlap",
            passed=True,
            measured=0,
            threshold=limits.overlap_tolerance,
        ))

    close_pairs_by_atoms = {
        issue.atom_indices: issue
        for issue in _too_close_issues(
            distances,
            minimum_distance=limits.basic_minimum_distance,
            covalent_radius_scale=None,
            pair_scope="all",
            include_overlaps=False,
            overlap_tolerance=limits.overlap_tolerance,
        )
    }
    if level in ("standard", "strict"):
        close_pairs_by_atoms.update(
            (issue.atom_indices, issue)
            for issue in _too_close_issues(
                distances,
                minimum_distance=limits.standard_minimum_distance,
                covalent_radius_scale=limits.standard_covalent_radius_scale,
                pair_scope="nonbonded",
                include_overlaps=False,
                overlap_tolerance=limits.overlap_tolerance,
            )
        )
    close_pairs = tuple(
        close_pairs_by_atoms[key] for key in sorted(close_pairs_by_atoms)
    )
    if close_pairs:
        checks.extend(AcceptanceCheck(
            name="atom_too_close",
            passed=False,
            measured=issue.distance,
            threshold=issue.threshold,
            atom_indices=issue.atom_indices,
            message="An atom pair is closer than the allowed separation",
        ) for issue in close_pairs)
    else:
        checks.append(AcceptanceCheck(
            name="atom_too_close",
            passed=True,
            measured=0,
            threshold=(
                limits.basic_minimum_distance
                if level == "basic"
                else (
                    limits.basic_minimum_distance,
                    limits.standard_minimum_distance,
                    limits.standard_covalent_radius_scale,
                )
            ),
        ))
    return tuple(checks)


def _coordinate_acceptance_checks(
    coordinates: np.ndarray,
    atom_count: int,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    bool,
]:
    """Return coordinate checks with row positions and a finite flag."""
    expected_shape = (atom_count, 3)
    shape_ok = coordinates.shape == expected_shape
    checks = [AcceptanceCheck(
        name="coordinate_shape",
        passed=shape_ok,
        measured=tuple(coordinates.shape),
        threshold=expected_shape,
        message="Coordinates must contain one Cartesian row per atom",
    )]

    finite_ok = shape_ok and bool(np.all(np.isfinite(coordinates)))
    nonfinite_indices = ()
    if shape_ok and not finite_ok:
        nonfinite_indices = tuple(
            int(index)
            for index in np.flatnonzero(
                ~np.all(np.isfinite(coordinates), axis=1)
            )
        )
    checks.append(AcceptanceCheck(
        name="finite_coordinates",
        passed=finite_ok,
        measured=finite_ok,
        threshold=True,
        atom_indices=nonfinite_indices,
        message="All Cartesian coordinates must be finite",
    ))
    return tuple(checks), finite_ok


def _bond_geometry_acceptance_checks(
    coordinates: np.ndarray,
    bonds: Sequence[Tuple[int, Tuple[int, int], Tuple[int, int], Optional[float], Optional[bool]]],
    level: AcceptanceLevel,
    limits: StructureAcceptanceThresholds,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    float,
]:
    """Return explicit-bond geometry checks and the maximum length."""
    checks = []
    maximum_bond_length = 0.0
    short_bond_count = 0
    for bond_index, (first, second), atom_indices, radius_sum, metal_ligand in bonds:
        distance = float(np.linalg.norm(
            coordinates[first] - coordinates[second]
        ))
        maximum_bond_length = max(maximum_bond_length, distance)
        valid_length = 0.0 < distance <= limits.maximum_bond_distance
        if not valid_length:
            checks.append(AcceptanceCheck(
                name="bond_distance",
                passed=False,
                measured=distance,
                threshold=(0.0, limits.maximum_bond_distance),
                atom_indices=atom_indices,
                bond_indices=(bond_index,),
                message="An explicit bond has an invalid or exploded length",
            ))

        if level in ("standard", "strict"):
            if radius_sum > 0.0:
                ratio = distance / radius_sum
                ratio_limits = (
                    limits.metal_ligand_bond_ratio
                    if metal_ligand
                    else limits.covalent_bond_ratio
                )
                if ratio < ratio_limits[0]:
                    short_bond_count += 1
                    checks.append(AcceptanceCheck(
                        name="short_bond",
                        passed=False,
                        measured=distance,
                        threshold=ratio_limits[0] * radius_sum,
                        atom_indices=atom_indices,
                        bond_indices=(bond_index,),
                        message=(
                            "An explicit bond is shorter than its "
                            "radius-scaled limit"
                        ),
                    ))
                if not ratio_limits[0] <= ratio <= ratio_limits[1]:
                    checks.append(AcceptanceCheck(
                        name="bond_length_ratio",
                        passed=False,
                        measured=ratio,
                        threshold=ratio_limits,
                        atom_indices=atom_indices,
                        bond_indices=(bond_index,),
                        message=(
                            "Bond length is inconsistent with covalent radii"
                        ),
                    ))
    if not any(check.name == "bond_distance" for check in checks):
        checks.append(AcceptanceCheck(
            name="bond_distance",
            passed=True,
            measured=maximum_bond_length,
            threshold=(0.0, limits.maximum_bond_distance),
        ))
    if level in ("standard", "strict") and not any(
        check.name == "bond_length_ratio" for check in checks
    ):
        checks.append(AcceptanceCheck(
            name="bond_length_ratio",
            passed=True,
            measured=None,
            threshold=(
                limits.covalent_bond_ratio,
                limits.metal_ligand_bond_ratio,
            ),
        ))
    if level in ("standard", "strict") and short_bond_count == 0:
        checks.append(AcceptanceCheck(
            name="short_bond",
            passed=True,
            measured=0,
            threshold=(
                limits.covalent_bond_ratio[0],
                limits.metal_ligand_bond_ratio[0],
            ),
            message="No explicit bond is below its radius-scaled limit",
        ))
    return tuple(checks), maximum_bond_length


def _coordination_metrics(
    coordinates: np.ndarray,
    metal_rows: Tuple[int, ...],
    metal_donor_rows: Tuple[Tuple[int, int], ...],
    row_labels: Tuple[Tuple[int, int], ...],
) -> Tuple[Tuple[int, int, Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]], ...]:
    labels = dict(row_labels)
    donors = {index: [] for index in metal_rows}
    for metal, donor in metal_donor_rows:
        donors[metal].append(donor)

    environments = []
    for metal, donor_indices in sorted(donors.items()):
        donor_indices = sorted(donor_indices)
        vectors = [
            coordinates[index] - coordinates[metal]
            for index in donor_indices
        ]
        distances = [float(np.linalg.norm(vector)) for vector in vectors]
        angles = []
        for first, second in combinations(vectors, 2):
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            if denominator > 0.0:
                cosine = np.clip(
                    np.dot(first, second) / denominator,
                    -1.0,
                    1.0,
                )
                angles.append(float(np.degrees(np.arccos(cosine))))
        environments.append((
            labels[metal],
            len(donor_indices),
            tuple(labels[index] for index in donor_indices),
            tuple(distances),
            tuple(angles),
        ))
    return tuple(environments)


def _bond_ring_acceptance_checks(
    facts: _BondRingAcceptanceFacts,
    bond_position_values: Tuple[Tuple[Tuple[int, int], int], ...],
) -> Tuple[AcceptanceCheck, ...]:
    bond_positions = dict(bond_position_values)
    checks = []
    for finding in facts.piercings:
        bond_key = finding.bond_key
        checks.append(AcceptanceCheck(
            name="bond_ring_piercing",
            passed=False,
            measured=finding.ring_key,
            threshold=geo.PiercingState.DOES_NOT_PIERCE.value,
            atom_indices=bond_key,
            bond_indices=(bond_positions[bond_key],),
            message="A finite bond segment pierces a selected ring surface",
        ))
    for finding in facts.undetermined:
        bond_key = finding.bond_key
        checks.append(AcceptanceCheck(
            name="bond_ring_piercing",
            passed=False,
            severity="warning",
            measured=tuple(
                sorted(cause.value for cause in finding.indeterminacy_causes)
            ),
            threshold=geo.PiercingState.DOES_NOT_PIERCE.value,
            atom_indices=bond_key,
            bond_indices=(bond_positions[bond_key],),
            message="The bond-ring spatial relation is mathematically undetermined",
        ))
    if facts.excluded_ring_count:
        checks.append(AcceptanceCheck(
            name="bond_ring_scope_coverage",
            passed=False,
            severity="warning",
            measured=facts.excluded_ring_count,
            threshold=0,
            message=(
                "Some rings exceed the configured maximum size and were not "
                "evaluated for bond-ring piercing"
            ),
        ))
    if not checks:
        checks.append(AcceptanceCheck(
            name="bond_ring_piercing",
            passed=True,
            measured=geo.PiercingState.DOES_NOT_PIERCE.value,
            threshold=geo.PiercingState.DOES_NOT_PIERCE.value,
        ))
    return tuple(checks)
