"""Hotpot stereochemistry transport contracts and public operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from .structure.stereo import (
    AtomTransportRecord,
    BondTransportRecord,
    HydrogenTransportState,
    StereoAlignmentReport,
    StereoAnchor,
    StereoAnchorKind,
    StereoAssignmentReport,
    StereoCarrier,
    StereoCarrierKind,
    StereoDerivedDescriptor,
    StereoGroupKind,
    StereoGroupRef,
    StereoIssueCode,
    StereoPayload,
    StereoRecord,
    StereoType,
    StereoValidationIssue,
    StereoValidationReport,
)
from .structure._stereo import _get_stereo_payload
from .operations import chemistry as _chemistry
from .operations import stereochemistry as _stereochemistry

if TYPE_CHECKING:
    from .core import Molecule


__all__ = (
    "AtomTransportRecord",
    "BondTransportRecord",
    "HydrogenTransportState",
    "StereoAlignmentReport",
    "StereoAnchor",
    "StereoAnchorKind",
    "StereoAssignmentReport",
    "StereoCarrier",
    "StereoCarrierKind",
    "StereoDerivedDescriptor",
    "StereoGroupKind",
    "StereoGroupRef",
    "StereoIssueCode",
    "StereoPayload",
    "StereoRecord",
    "StereoType",
    "StereoValidationIssue",
    "StereoValidationReport",
    "align_stereochemistry",
    "assign_stereochemistry",
    "get_stereo_payload",
    "set_stereo_payload",
    "validate_stereochemistry",
)


def get_stereo_payload(mol: Molecule) -> StereoPayload:
    """Return the exact Level-1 payload stored by a molecule."""

    return _get_stereo_payload(mol)


def set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None:
    """Store Level-1 facts without chemical interpretation."""

    _chemistry._set_stereo_payload(mol, payload)


def validate_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoValidationReport:
    """Validate graph stereogenicity and optional coordinate semantics."""

    return _stereochemistry.validate_stereochemistry(mol, coordinates=coordinates)


def assign_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoAssignmentReport:
    """Assign coordinate descriptors and direct-priority names without mutation."""

    return _stereochemistry.assign_stereochemistry(mol, coordinates=coordinates)


def align_stereochemistry(
    reference: Molecule,
    target: Molecule,
    atom_mapping: Mapping[int, int],
) -> StereoAlignmentReport:
    """Compare supported descriptors under an explicit atom mapping."""

    return _stereochemistry.align_stereochemistry(reference, target, atom_mapping)
