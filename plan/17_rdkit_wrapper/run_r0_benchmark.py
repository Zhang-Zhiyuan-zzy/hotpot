#!/usr/bin/env python3
"""Run the detached Plan 17 R0 native-ETKDG benchmark.

This route parses one ligand into a Hotpot molecule, calls the public native
``rdWrappers`` sampler, and observes a copied coordinate batch only after
generation has finished.  It deliberately contains no geometry gate,
perturbation, force field, minimization, or conformer-selection step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


CANONICAL_INPUT_SHA256 = (
    "e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7"
)
CANONICAL_INPUT_COUNT = 187
DEFAULT_CONFORMER_COUNTS = (1, 8, 32)
BASE_SEED = 20260921
UNIQUENESS_RMSD_ANGSTROM = 1.0e-3


def _declares_level2(descriptor: str) -> bool:
    """Return whether a descriptor declares the Level-2 namespace."""

    return descriptor == "hotpot.stereo/2" or descriptor.startswith(
        "hotpot.stereo/2/"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_corpus(
    path: Path,
    *,
    expected_count: int = CANONICAL_INPUT_COUNT,
    expected_sha256: str = CANONICAL_INPUT_SHA256,
) -> tuple[dict[str, object], ...]:
    """Load the byte-frozen corpus while retaining physical line identities."""

    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "R0 input SHA-256 differs from the canonical 187-ligand corpus: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )

    cases = tuple(
        {
            "case_index": line_number,
            "smiles": stripped.split(maxsplit=1)[0],
        }
        for line_number, line in enumerate(
            source.decode("utf-8").splitlines(), start=1
        )
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    if len(cases) != expected_count:
        raise ValueError(
            f"expected {expected_count} input ligands, found {len(cases)}"
        )
    return cases


def _parse_smiles(smiles: str):
    """Parse through Hotpot's real public graph-construction path."""

    import hotpot

    return hotpot.read_mol(smiles, fmt="smi")


def _digest_update(digest, value: object) -> None:
    if value is None:
        digest.update(b"none;")
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array:")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b":")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(b":")
        digest.update(array.tobytes(order="C"))
        digest.update(b";")
    elif isinstance(value, np.generic):
        _digest_update(digest, value.item())
    elif isinstance(value, Enum):
        _digest_update(digest, value.value)
    elif is_dataclass(value) and not isinstance(value, type):
        digest.update(type(value).__qualname__.encode("utf-8"))
        digest.update(b"{")
        for field in fields(value):
            digest.update(field.name.encode("utf-8"))
            digest.update(b"=")
            _digest_update(digest, getattr(value, field.name))
        digest.update(b"};")
    elif isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=str):
            _digest_update(digest, str(key))
            _digest_update(digest, value[key])
        digest.update(b"};")
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence[")
        for item in value:
            _digest_update(digest, item)
        digest.update(b"];")
    else:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b":")
        digest.update(str(value).encode("utf-8"))
        digest.update(b";")


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        _digest_update(digest, value)
    return digest.hexdigest()


def _coordinate_hash(coordinates: np.ndarray) -> str:
    canonical = np.ascontiguousarray(coordinates, dtype="<f8")
    return _digest(canonical)


def _snapshot_molecule(molecule) -> dict[str, object]:
    """Snapshot the exact fields consumed by native ETKDG plus live geometry."""

    from hotpot.cheminfo._native.buffers import pack_molecule
    from hotpot.cheminfo.rdWrappers.native import _RDKIT_ETKDG_FIELDS
    from hotpot.cheminfo.stereo import get_stereo_payload

    packed = pack_molecule(molecule, _RDKIT_ETKDG_FIELDS)
    payload = get_stereo_payload(molecule)
    coordinates = np.ascontiguousarray(molecule.coordinates, dtype=np.float64)
    return {
        "atom_count": packed.atom_count,
        "bond_count": packed.bond_count,
        "atom_order_sha256": _digest(packed.atomic_numbers),
        "formal_charge_sha256": _digest(packed.formal_charges),
        "aromaticity_sha256": _digest(
            packed.atom_aromatic, packed.bond_aromatic
        ),
        "topology_sha256": _digest(
            packed.bond_indices, packed.bond_kinds
        ),
        "level1_sha256": _digest(
            packed.stereo_records,
            packed.atom_transport,
            packed.bond_transport,
        ),
        "stereo_record_count": len(payload.records),
        "atom_transport_count": len(payload.atom_facts),
        "bond_transport_count": len(payload.bond_facts),
        "stereo_payload_nonempty": bool(
            payload.records or payload.atom_facts or payload.bond_facts
        ),
        "level2_declared_record_count": sum(
            _declares_level2(record.descriptor) for record in payload.records
        ),
        "coordinates_sha256": _coordinate_hash(coordinates),
        "conformer_count": len(molecule.conformers),
    }


def _issue_record(issue) -> dict[str, object]:
    """Serialize one public stereo issue without leaking Enum objects."""

    return {
        "code": issue.code.value,
        "message": issue.message,
        "record_index": issue.record_index,
        "atom_indices": list(issue.atom_indices),
    }


def _derived_record(descriptor) -> dict[str, object]:
    """Serialize one public derived descriptor for benchmark evidence."""

    return {
        "record_index": descriptor.record_index,
        "stereo_type": descriptor.stereo_type.value,
        "label": descriptor.label,
        "system": descriptor.system,
        "is_absolute": descriptor.is_absolute,
    }


def _observe_level2_stereochemistry(
    molecule,
    coordinates: np.ndarray,
) -> dict[str, object]:
    """Assess declared Level-2 facts without changing the R0 native route."""

    from hotpot.cheminfo.stereo import (
        align_stereochemistry,
        assign_stereochemistry,
        get_stereo_payload,
        set_stereo_payload,
        validate_stereochemistry,
    )

    payload = get_stereo_payload(molecule)
    declared_record_count = sum(
        _declares_level2(record.descriptor) for record in payload.records
    )
    source_payload_sha256 = _digest(payload)
    if declared_record_count == 0:
        return {
            "applicability": "NOT_APPLICABLE",
            "reason": "input carries no hotpot.stereo/2 record",
            "declared_record_count": 0,
            "conformer_count": int(coordinates.shape[0]),
            "source_payload_sha256": source_payload_sha256,
            "input_payload_preserved": (
                _digest(get_stereo_payload(molecule)) == source_payload_sha256
            ),
            "conformers": [],
        }

    identity_mapping = dict(enumerate(range(len(molecule.atoms))))
    conformer_reports = []
    for conformer_index, conformer in enumerate(coordinates):
        validation = validate_stereochemistry(
            molecule,
            coordinates=conformer,
        )
        assignment = assign_stereochemistry(
            molecule,
            coordinates=conformer,
        )
        assigned = molecule.copy()
        set_stereo_payload(assigned, assignment.payload)
        alignment = align_stereochemistry(
            molecule,
            assigned,
            identity_mapping,
        )
        conformer_reports.append(
            {
                "conformer_index": conformer_index,
                "is_valid": validation.is_valid,
                "validation_issues": [
                    _issue_record(issue) for issue in validation.issues
                ],
                "assignment_unresolved": [
                    _issue_record(issue) for issue in assignment.unresolved
                ],
                "assigned_payload_sha256": _digest(assignment.payload),
                "derived": [
                    _derived_record(descriptor)
                    for descriptor in assignment.derived
                ],
                "is_equivalent": alignment.is_equivalent,
                "alignment_issues": [
                    _issue_record(issue) for issue in alignment.issues
                ],
                "aligned_payload_sha256": _digest(alignment.aligned_payload),
            }
        )

    return {
        "applicability": "APPLICABLE",
        "reason": None,
        "declared_record_count": declared_record_count,
        "conformer_count": len(conformer_reports),
        "source_payload_sha256": source_payload_sha256,
        "input_payload_preserved": (
            _digest(get_stereo_payload(molecule)) == source_payload_sha256
        ),
        "conformers": conformer_reports,
    }


def _fixed_row_kabsch_rmsd(
    first: np.ndarray, second: np.ndarray
) -> float:
    """Return RMSD after rigid alignment without atom permutation/symmetry."""

    if first.shape[0] == 0:
        return 0.0
    first_centered = first - np.mean(first, axis=0)
    second_centered = second - np.mean(second, axis=0)
    covariance = first_centered.T @ second_centered
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    difference = first_centered @ rotation - second_centered
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def _detached_coordinate_observation(
    coordinates: np.ndarray,
    *,
    atom_count: int,
    uniqueness_threshold: float = UNIQUENESS_RMSD_ANGSTROM,
) -> dict[str, object]:
    """Measure a detached batch without rejecting, repairing, or selecting it."""

    detached = np.array(coordinates, dtype=np.float64, order="C", copy=True)
    expected_shape = detached.ndim == 3 and detached.shape[1:] == (
        atom_count,
        3,
    )
    finite = bool(np.isfinite(detached).all())

    pairwise = []
    representatives: list[int] = []
    if expected_shape and finite:
        distance_by_pair = {}
        for first_index in range(detached.shape[0]):
            for second_index in range(first_index):
                distance = _fixed_row_kabsch_rmsd(
                    detached[first_index], detached[second_index]
                )
                distance_by_pair[first_index, second_index] = distance
                pairwise.append(distance)
        for index in range(detached.shape[0]):
            distances = [
                distance_by_pair[max(index, representative), min(index, representative)]
                for representative in representatives
            ]
            if not distances or all(
                distance > uniqueness_threshold for distance in distances
            ):
                representatives.append(index)

    return {
        "coordinate_sha256": _coordinate_hash(detached),
        "conformer_coordinate_sha256": [
            _coordinate_hash(conformer) for conformer in detached
        ]
        if expected_shape
        else [],
        "finite": finite,
        "atom_row_shape_exact": expected_shape,
        "unique_conformer_count": (
            len(representatives) if expected_shape and finite else None
        ),
        "uniqueness_method": (
            "fixed_atom_row_kabsch_rmsd_not_chemically_symmetry_aware"
        ),
        "uniqueness_threshold_angstrom": uniqueness_threshold,
        "pairwise_rmsd_min_angstrom": min(pairwise) if pairwise else None,
        "pairwise_rmsd_median_angstrom": (
            float(np.median(pairwise)) if pairwise else None
        ),
        "pairwise_rmsd_max_angstrom": max(pairwise) if pairwise else None,
    }


def _generate_conformers(molecule, *, conformer_count: int, seed: int):
    from hotpot.cheminfo import rdWrappers

    return rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(
            conformer_count=conformer_count,
            seed=seed,
            prune_rms_threshold=None,
            use_random_coordinates=False,
        ),
    )


def _same_batch(first, second) -> bool:
    first_coordinates = np.asarray(first.coordinates)
    second_coordinates = np.asarray(second.coordinates)
    return bool(
        first.status == second.status
        and first.diagnostic == second.diagnostic
        and first.records == second.records
        and first.engine == second.engine
        and first.engine_version == second.engine_version
        and np.array_equal(first_coordinates, second_coordinates, equal_nan=True)
    )


def _run_trial(
    case: Mapping[str, object],
    conformer_count: int,
    *,
    parse_molecule: Callable[[str], object] = _parse_smiles,
    generate: Callable[..., object] = _generate_conformers,
    snapshot_molecule: Callable[[object], dict[str, object]] = _snapshot_molecule,
    observe: Callable[..., dict[str, object]] = _detached_coordinate_observation,
    observe_level2: Callable[..., dict[str, object]] = (
        _observe_level2_stereochemistry
    ),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Run one matrix cell; generation completes before observation starts."""

    case_index = int(case["case_index"])
    smiles = str(case["smiles"])
    seed = BASE_SEED + case_index
    trial_started = clock_ns()
    record: dict[str, object] = {
        "route": "R0",
        "case_index": case_index,
        "smiles": smiles,
        "requested_conformer_count": conformer_count,
        "seed": seed,
        "outer_workers": 1,
        "etkdg_internal_threads": 1,
    }

    parse_started = clock_ns()
    try:
        molecule = parse_molecule(smiles)
    except Exception as error:  # Each failed corpus row remains evidence.
        record.update(
            {
                "status": "PARSE_FAILED",
                "diagnostic": f"{type(error).__name__}: {error}",
                "parse_seconds": (clock_ns() - parse_started) / 1.0e9,
                "input_snapshot_seconds": 0.0,
                "generate_seconds": 0.0,
                "determinism_rerun_seconds": 0.0,
                "observation_seconds": 0.0,
                "level2_observation_seconds": 0.0,
                "total_seconds": (clock_ns() - trial_started) / 1.0e9,
                "returned_conformer_count": 0,
                "finite": False,
                "atom_row_shape_exact": False,
                "input_nonmutating": None,
                "same_seed_deterministic": None,
                "level2_stereo": {
                    "applicability": "UNAVAILABLE",
                    "reason": "molecule parse failed",
                    "conformers": [],
                },
            }
        )
        return record
    record["parse_seconds"] = (clock_ns() - parse_started) / 1.0e9

    snapshot_started = clock_ns()
    try:
        before = snapshot_molecule(molecule)
    except Exception as error:
        record.update(
            {
                "status": "INPUT_SNAPSHOT_FAILED",
                "diagnostic": f"{type(error).__name__}: {error}",
                "input_snapshot_seconds": (
                    clock_ns() - snapshot_started
                ) / 1.0e9,
                "generate_seconds": 0.0,
                "determinism_rerun_seconds": 0.0,
                "observation_seconds": 0.0,
                "level2_observation_seconds": 0.0,
                "total_seconds": (clock_ns() - trial_started) / 1.0e9,
                "returned_conformer_count": 0,
                "finite": False,
                "atom_row_shape_exact": False,
                "input_nonmutating": None,
                "same_seed_deterministic": None,
                "level2_stereo": {
                    "applicability": "UNAVAILABLE",
                    "reason": "input snapshot failed",
                    "conformers": [],
                },
            }
        )
        return record
    record["input_snapshot_seconds"] = (
        clock_ns() - snapshot_started
    ) / 1.0e9
    record.update(
        {
            "input_stereo_record_count": before["stereo_record_count"],
            "input_atom_transport_count": before["atom_transport_count"],
            "input_bond_transport_count": before["bond_transport_count"],
            "input_stereo_payload_nonempty": before[
                "stereo_payload_nonempty"
            ],
            "input_level2_declared_record_count": before[
                "level2_declared_record_count"
            ],
        }
    )

    generate_started = clock_ns()
    try:
        result = generate(
            molecule,
            conformer_count=conformer_count,
            seed=seed,
        )
    except Exception as error:  # Keep the route stage-resolved.
        record.update(
            {
                "status": "GENERATION_EXCEPTION",
                "diagnostic": f"{type(error).__name__}: {error}",
                "generate_seconds": (clock_ns() - generate_started) / 1.0e9,
                "determinism_rerun_seconds": 0.0,
                "level2_observation_seconds": 0.0,
                "returned_conformer_count": 0,
                "finite": False,
                "atom_row_shape_exact": False,
                "same_seed_deterministic": None,
                "level2_stereo": {
                    "applicability": "UNAVAILABLE",
                    "reason": "native generation failed",
                    "declared_record_count": before[
                        "level2_declared_record_count"
                    ],
                    "conformers": [],
                },
            }
        )
        observation_started = clock_ns()
        after = snapshot_molecule(molecule)
        record["observation_seconds"] = (
            clock_ns() - observation_started
        ) / 1.0e9
        record.update(_input_parity(before, after))
        record["total_seconds"] = (clock_ns() - trial_started) / 1.0e9
        return record
    record["generate_seconds"] = (clock_ns() - generate_started) / 1.0e9

    rerun_started = clock_ns()
    rerun = None
    rerun_diagnostic = None
    try:
        rerun = generate(
            molecule,
            conformer_count=conformer_count,
            seed=seed,
        )
    except Exception as error:
        rerun_diagnostic = f"{type(error).__name__}: {error}"
    record["determinism_rerun_seconds"] = (
        clock_ns() - rerun_started
    ) / 1.0e9

    observation_started = clock_ns()
    after = snapshot_molecule(molecule)
    coordinates = np.asarray(result.coordinates)
    observation = observe(coordinates, atom_count=int(before["atom_count"]))
    record.update(_input_parity(before, after))
    record.update(observation)
    record.update(
        {
            "status": result.status.name,
            "status_code": int(result.status),
            "diagnostic": result.diagnostic,
            "returned_conformer_count": int(coordinates.shape[0])
            if coordinates.ndim >= 1
            else 0,
            "record_count": len(result.records),
            "record_count_matches_returned": (
                len(result.records) == coordinates.shape[0]
                if coordinates.ndim >= 1
                else False
            ),
            "returned_count_matches_requested": (
                coordinates.ndim >= 1
                and coordinates.shape[0] == conformer_count
            ),
            "record_ordinals": [item.ordinal for item in result.records],
            "engine": result.engine,
            "engine_version": result.engine_version,
            "same_seed_deterministic": (
                _same_batch(result, rerun) if rerun is not None else False
            ),
            "determinism_rerun_diagnostic": rerun_diagnostic,
            "determinism_coordinate_sha256": (
                _coordinate_hash(np.asarray(rerun.coordinates))
                if rerun is not None
                else None
            ),
            "atom_row_order_contract": (
                "native_result_rows_follow_fixed_input_rows; "
                "the coordinate-only result carries no independent atom labels"
            ),
        }
    )
    record["observation_seconds"] = (
        clock_ns() - observation_started
    ) / 1.0e9
    level2_started = clock_ns()
    level2_coordinates = np.array(
        coordinates,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    record["level2_stereo"] = observe_level2(
        molecule,
        level2_coordinates,
    )
    record["level2_observation_seconds"] = (
        clock_ns() - level2_started
    ) / 1.0e9
    record["total_seconds"] = (clock_ns() - trial_started) / 1.0e9
    return record


def _input_parity(
    before: Mapping[str, object], after: Mapping[str, object]
) -> dict[str, object]:
    names = (
        "atom_order_sha256",
        "formal_charge_sha256",
        "aromaticity_sha256",
        "topology_sha256",
        "level1_sha256",
    )
    parity = {
        name.removesuffix("_sha256") + "_preserved": before[name] == after[name]
        for name in names
    }
    parity.update(
        {
            "input_atom_count": before["atom_count"],
            "input_bond_count": before["bond_count"],
            "input_coordinates_preserved": (
                before["coordinates_sha256"] == after["coordinates_sha256"]
            ),
            "input_conformers_preserved": (
                before["conformer_count"] == after["conformer_count"]
            ),
        }
    )
    parity["input_nonmutating"] = all(
        value for key, value in parity.items() if key.endswith("_preserved")
    )
    return parity


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(values: Iterable[float]) -> dict[str, float | None]:
    measured = tuple(float(value) for value in values)
    return {
        "p50_seconds": _percentile(measured, 0.50),
        "p95_seconds": _percentile(measured, 0.95),
        "max_seconds": max(measured) if measured else None,
        "aggregate_seconds": sum(measured),
    }


def _level2_aggregate(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate semantic observations without counting N/A as a pass."""

    observations = tuple(
        record.get(
            "level2_stereo",
            {
                "applicability": "UNAVAILABLE",
                "reason": "record predates Level-2 observation",
                "conformers": [],
            },
        )
        for record in records
    )
    conformers = tuple(
        conformer
        for observation in observations
        for conformer in observation.get("conformers", ())
    )
    validation_issue_codes = Counter(
        issue["code"]
        for conformer in conformers
        for issue in conformer["validation_issues"]
    )
    assignment_issue_codes = Counter(
        issue["code"]
        for conformer in conformers
        for issue in conformer["assignment_unresolved"]
    )
    alignment_issue_codes = Counter(
        issue["code"]
        for conformer in conformers
        for issue in conformer["alignment_issues"]
    )
    applicability_counts = Counter(
        str(observation["applicability"]) for observation in observations
    )
    return {
        "trial_applicability_counts": dict(sorted(applicability_counts.items())),
        "not_applicable_returned_conformer_count": sum(
            int(observation.get("conformer_count", 0))
            for observation in observations
            if observation["applicability"] == "NOT_APPLICABLE"
        ),
        "assessed_conformer_count": len(conformers),
        "valid_conformer_count": sum(
            conformer["is_valid"] is True for conformer in conformers
        ),
        "invalid_conformer_count": sum(
            conformer["is_valid"] is False for conformer in conformers
        ),
        "equivalent_conformer_count": sum(
            conformer["is_equivalent"] is True for conformer in conformers
        ),
        "non_equivalent_conformer_count": sum(
            conformer["is_equivalent"] is False for conformer in conformers
        ),
        "validation_issue_code_counts": dict(
            sorted(validation_issue_codes.items())
        ),
        "assignment_issue_code_counts": dict(
            sorted(assignment_issue_codes.items())
        ),
        "alignment_issue_code_counts": dict(
            sorted(alignment_issue_codes.items())
        ),
        "input_payload_preservation_failure_count": sum(
            observation.get("input_payload_preserved") is False
            for observation in observations
        ),
    }


def _aggregate_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    timing_names = (
        "parse_seconds",
        "input_snapshot_seconds",
        "generate_seconds",
        "determinism_rerun_seconds",
        "observation_seconds",
        "level2_observation_seconds",
        "total_seconds",
    )
    timings = {
        name.removesuffix("_seconds"): _timing_summary(
            float(record[name]) for record in records if name in record
        )
        for name in timing_names
    }
    generate_seconds = float(timings["generate"]["aggregate_seconds"] or 0.0)
    returned = sum(
        int(record.get("returned_conformer_count", 0)) for record in records
    )
    trial_count = len(records)
    counters = {
        "full_return_count": sum(
            record.get("returned_count_matches_requested") is True
            for record in records
        ),
        "nonfinite_count": sum(
            int(record.get("returned_conformer_count", 0)) > 0
            and record.get("finite") is False
            for record in records
        ),
        "atom_row_shape_failure_count": sum(
            "coordinate_sha256" in record
            and record.get("atom_row_shape_exact") is False
            for record in records
        ),
        "atom_order_parity_failure_count": sum(
            record.get("atom_order_preserved") is False for record in records
        ),
        "topology_parity_failure_count": sum(
            record.get("topology_preserved") is False for record in records
        ),
        "level1_parity_failure_count": sum(
            record.get("level1_preserved") is False for record in records
        ),
        "input_nonmutation_failure_count": sum(
            record.get("input_nonmutating") is False for record in records
        ),
        "same_seed_mismatch_count": sum(
            record.get("same_seed_deterministic") is False
            for record in records
        ),
    }
    return {
        "status_counts": dict(
            sorted(Counter(str(record["status"]) for record in records).items())
        ),
        "timings": timings,
        "throughput": {
            "denominator": "aggregate_primary_generate_seconds",
            "ligands_per_second": (
                trial_count / generate_seconds if generate_seconds else None
            ),
            "conformers_per_second": (
                returned / generate_seconds if generate_seconds else None
            ),
            "returned_primary_conformer_count": returned,
        },
        "verification_counts": counters,
        "stereo_payload_coverage": {
            "nonempty_input_trial_count": sum(
                record.get("input_stereo_payload_nonempty") is True
                for record in records
            ),
            "level2_declared_input_trial_count": sum(
                int(record.get("input_level2_declared_record_count", 0)) > 0
                for record in records
            ),
            "input_stereo_record_count": sum(
                int(record.get("input_stereo_record_count", 0))
                for record in records
            ),
            "input_atom_transport_count": sum(
                int(record.get("input_atom_transport_count", 0))
                for record in records
            ),
            "input_bond_transport_count": sum(
                int(record.get("input_bond_transport_count", 0))
                for record in records
            ),
        },
        "level2_stereo": _level2_aggregate(records),
    }


def _summarize(
    records: Sequence[Mapping[str, object]],
    *,
    corpus: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    aggregate = _aggregate_records(records)
    trial_count = len(records)
    counts = sorted(
        {int(record["requested_conformer_count"]) for record in records}
    )
    return {
        "schema": "hotpot.plan17.r0/2",
        "route": "R0",
        "corpus": dict(corpus),
        "matrix": {
            "conformer_counts": counts,
            "trial_count": trial_count,
            "distinct_ligand_count": len(
                {int(record["case_index"]) for record in records}
            ),
            "base_seed": BASE_SEED,
        },
        "execution_policy": {
            "input_parser": "hotpot.read_mol(smiles, fmt='smi')",
            "outer_workers": 1,
            "etkdg_internal_threads": 1,
            "hotpot_geometry_gate": False,
            "hotpot_perturbation": False,
            "openbabel_force_field_or_optimization": False,
            "rdkit_force_field": False,
            "conformer_selection": False,
            "observation_is_detached_and_post_generation": True,
            "level2_observation_is_post_generation": True,
            "level2_observation_is_detached": True,
            "level2_observation_is_non_gating": True,
            "level2_not_applicable_is_not_counted_as_pass": True,
            "primary_generate_timing_includes_hotpot_native_buffer_packing": True,
        },
        "runtime": dict(runtime),
        **aggregate,
        "by_conformer_count": {
            str(count): _aggregate_records(
                tuple(
                    record
                    for record in records
                    if int(record["requested_conformer_count"]) == count
                )
            )
            for count in counts
        },
    }


def _git_value(repository: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _runtime_metadata(repository: Path) -> dict[str, object]:
    from rdkit import rdBase

    from hotpot.cheminfo import rdWrappers

    native = rdWrappers.runtime_info()
    affinity = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    status = _git_value(repository, "status", "--porcelain")
    return {
        "native": {
            "hotpot_version": native.hotpot_version,
            "rdkit_version": native.rdkit_version,
            "native": native.native,
        },
        "python_rdkit_version": rdBase.rdkitVersion,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "affinity_count": affinity,
            "model": _cpu_model(),
        },
        "git": {
            "commit": _git_value(repository, "rev-parse", "HEAD"),
            "branch": _git_value(repository, "branch", "--show-current"),
            "dirty": bool(status),
        },
    }


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return platform.processor() or None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() in {"model name", "Hardware"}:
            return value.strip() or None
    return platform.processor() or None


def _load_development_extension(path: Path):
    """Load one explicitly requested development build as ``_rd_native``."""

    importlib.import_module("hotpot.cheminfo.rdWrappers")
    module_name = "hotpot.cheminfo.rdWrappers._rd_native"
    specification = importlib.util.spec_from_file_location(
        module_name, path.resolve()
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load development extension {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in value.split(","))
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("counts must be positive integers")
    return counts


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the detached Plan 17 R0 native-ETKDG benchmark"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--counts",
        type=_parse_counts,
        default=DEFAULT_CONFORMER_COUNTS,
        help="comma-separated conformer counts (default: 1,8,32)",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        help="explicit development _rd_native build; installed module is default",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _argument_parser().parse_args(arguments)
    input_path = options.input
    cases = _load_corpus(input_path)
    if options.extension is not None:
        _load_development_extension(options.extension)

    repository = Path(__file__).resolve().parents[2]
    runtime = _runtime_metadata(repository)
    records = [
        _run_trial(case, count)
        for case in cases
        for count in options.counts
    ]
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "r0_cases.jsonl"
    summary_path = output_dir / "r0_summary.json"
    corpus = {
        "input": str(input_path),
        "byte_sha256": CANONICAL_INPUT_SHA256,
        "ligand_count": CANONICAL_INPUT_COUNT,
        "case_identity": "one_based_physical_input_line_number",
    }
    summary = _summarize(records, corpus=corpus, runtime=runtime)
    _write_jsonl(jsonl_path, records)
    _write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
