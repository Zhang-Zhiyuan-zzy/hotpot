"""Per-owner stereo capture, deferred preparation and ordered public operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..adapters.structure.stereochemistry import (
    _capture_stereo_snapshot, _check_stereo_capture, _pack_gate6_target,
    _stereo_prepared_fields,
)
from ..structure._commit import _TRANSACTIONS
from ..structure.stereo import (
    StereoAlignmentReport, StereoAssignmentReport, StereoValidationReport,
)
from ..rules.chemistry.stereochemistry.algebra import _declares_gate7_coordination
from ..rules.chemistry.stereochemistry.graph import (
    _gate6_preflight, _project_gate6_graph, _gate6_stereogenicity_candidates,
    _gate6_stereogenicity_issues,
)
from ..rules.chemistry.stereochemistry.validation import (
    _validate_gate2_stereochemistry, _coordinate_validation_report,
)
from ..rules.chemistry.stereochemistry.assignment import _stereo_assignment_report
from ..rules.chemistry.stereochemistry.alignment import (
    _alignment_mapping_issues, _align_stereo_payloads,
)
from .chemistry import _require_implicit_hydrogens

__all__ = ()


@contextmanager
def _stereo_owner_scope(mol, *, nonblocking=False):
    """Acquire only this stereo scope's recursion on one molecular owner."""

    if not mol._mutation_lock.acquire(blocking=not nonblocking):
        raise RuntimeError(
            "Stereochemistry alignment cannot acquire a distinct owner while a factual transaction is active"
        )
    try:
        yield
    finally:
        mol._mutation_lock.release()


def _capture_stereo_owner(mol, *, nonblocking=False):
    with _stereo_owner_scope(mol, nonblocking=nonblocking):
        return _capture_stereo_snapshot(mol)


def _gate6_projection(mol, *, capture=None, nonblocking=False, prepared=None):
    """Use the production capture/check/H path for one represented projection."""

    if capture is None:
        capture = _capture_stereo_owner(mol, nonblocking=nonblocking)
    snapshot = capture[0]
    preflight, issues = _gate6_preflight(snapshot)
    if issues:
        return None, issues
    with _stereo_owner_scope(mol, nonblocking=nonblocking):
        _check_stereo_capture(mol, capture)
        hydrogens = _require_implicit_hydrogens(mol)
        scalar_fields = _stereo_prepared_fields(
            mol, fields=("atomic_number", "formal_charge", "is_aromatic"),
        )
    if prepared is not None:
        prepared["atomic_numbers"] = tuple(row[0] for row in scalar_fields)
    return _project_gate6_graph(snapshot, preflight, hydrogens, scalar_fields)


def _gate7_projection_result(mol, *, capture=None, nonblocking=False, prepared=None):
    """A coordination head requests projection before Gate-2 validation."""

    if capture is None:
        capture = _capture_stereo_owner(mol, nonblocking=nonblocking)
    has_coordination = any(
        _declares_gate7_coordination(record.descriptor)
        for record in capture[0].payload.records
    )
    return (
        _gate6_projection(mol, capture=capture, nonblocking=nonblocking, prepared=prepared)
        if has_coordination else None
    )


def _validate_graph_stereochemistry(
    mol, *, capture=None, projection_result=None, nonblocking=False, prepared=None,
):
    """Keep conditional preparation/native work outside pure report rules."""

    if capture is None:
        capture = _capture_stereo_owner(mol, nonblocking=nonblocking)
    if projection_result is None:
        projection_result = _gate7_projection_result(
            mol, capture=capture, nonblocking=nonblocking, prepared=prepared,
        )
    snapshot = capture[0]
    report = _validate_gate2_stereochemistry(snapshot, projection_result=projection_result)
    candidates, valid_indices, unsupported_indices = _gate6_stereogenicity_candidates(
        snapshot.payload, report,
    )
    if not candidates:
        return report
    new_projection = projection_result is None
    if new_projection:
        projection_result = _gate6_projection(
            mol, capture=capture, nonblocking=nonblocking, prepared=prepared,
        )
    graph, projection_issues = projection_result
    if graph is None:
        issues = projection_issues if new_projection else ()
    else:
        from ..search.native import _NativeTarget

        native_target = _NativeTarget(_pack_gate6_target(graph))
        issues = _gate6_stereogenicity_issues(
            snapshot.payload, graph, candidates, valid_indices, unsupported_indices,
            native_target.iter_mappings,
        )
    if not issues:
        return report
    return StereoValidationReport(is_valid=False, issues=(*report.issues, *issues))


def validate_stereochemistry(
    mol,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoValidationReport:
    """Validate graph stereogenicity and optional coordinate semantics."""

    capture = _capture_stereo_owner(mol)
    report = _validate_graph_stereochemistry(mol, capture=capture)
    return _coordinate_validation_report(capture[0], coordinates, report)


def assign_stereochemistry(
    mol,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoAssignmentReport:
    """Assign coordinate descriptors and direct-priority names without mutation."""

    capture = _capture_stereo_owner(mol)
    prepared = {}
    report = _validate_graph_stereochemistry(mol, capture=capture, prepared=prepared)
    snapshot = capture[0]
    return _stereo_assignment_report(
        snapshot.payload, len(snapshot.atom_keys), prepared.get("atomic_numbers", ()),
        report, coordinates,
    )


def align_stereochemistry(
    reference,
    target,
    atom_mapping: Mapping[int, int],
) -> StereoAlignmentReport:
    """Compare supported descriptors under an explicit atom mapping."""

    same_owner = reference is target
    nonblocking = not same_owner and bool(_TRANSACTIONS.get())
    reference_capture = _capture_stereo_owner(reference, nonblocking=nonblocking)
    target_capture = (
        reference_capture if same_owner
        else _capture_stereo_owner(target, nonblocking=nonblocking)
    )
    reference_snapshot, target_snapshot = reference_capture[0], target_capture[0]
    entries = tuple(sorted(
        atom_mapping.items(), key=lambda item: (repr(item[0]), repr(item[1])),
    ))
    mapping_issues, valid_mapping = _alignment_mapping_issues(
        len(reference_snapshot.atom_keys), len(target_snapshot.atom_keys),
        reference_snapshot.payload, target_snapshot.payload, entries,
    )
    reference_projection = _gate7_projection_result(
        reference, capture=reference_capture, nonblocking=nonblocking,
    )
    target_projection = (
        reference_projection if same_owner
        else _gate7_projection_result(target, capture=target_capture, nonblocking=nonblocking)
    )
    reference_validation = _validate_graph_stereochemistry(
        reference, capture=reference_capture, projection_result=reference_projection,
        nonblocking=nonblocking,
    )
    target_validation = (
        reference_validation if same_owner
        else _validate_graph_stereochemistry(
            target, capture=target_capture, projection_result=target_projection,
            nonblocking=nonblocking,
        )
    )
    return _align_stereo_payloads(
        reference_snapshot, target_snapshot, valid_mapping, mapping_issues,
        reference_validation, target_validation, reference_projection, target_projection,
    )
