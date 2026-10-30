#!/usr/bin/env python3
"""Run detached Plan 17 C1/C2 coordination-embedding benchmarks.

C1 and C2 consume the same explicit native-graph JSON artifact. C1 invokes
stock whole-complex ETKDGv3; C2 overlays manifest-supplied coordination bounds.
Neither route calls Open Babel, UFF/MMFF optimization, ``hotpot.ff``, a Hotpot
geometry gate, or perturbation. Bound conformance is observed only after native
generation and never rejects, repairs, or selects a returned conformer.

Each canonical-manifest case names ``native_structure_path`` (or defaults to
``case-XXXX.json`` under ``--structure-root``). The structure JSON schema is::

    {"schema_version": 1, "case_index": 1,
     "atoms": [{"atomic_number": 7, "formal_charge": 0,
                "is_aromatic": false}, ...],
     "bonds": [{"begin": 0, "end": 2, "bond_order": 1.0,
                "bond_kind": "dative",
                "bond_direction": "atom1_to_atom2"}, ...],
     "constraints": {
       "policy": {"name": "hotpot-native-placement-v1",
                  "distance_unit": "angstrom"},
       "coordination_pairs": [[0, 2]],
       "metal_donor_bounds_angstrom": [[1.9, 2.1]],
       "donor_pair_indices": [],
       "donor_pair_bounds_angstrom": []}}

Formal mode accepts only the byte-frozen 182-case manifest identity. ``--smoke``
accepts a development cohort but labels every output as non-formal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


FORMAL_MANIFEST_SHA256 = (
    "4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377"
)
FORMAL_CASE_COUNT = 182
DEFAULT_CONFORMER_COUNTS = (1, 8, 32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_manifest(path: Path, *, smoke: bool) -> tuple[Mapping[str, object], ...]:
    digest = _sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(payload["cases"])
    if not smoke:
        if digest != FORMAL_MANIFEST_SHA256:
            raise ValueError(
                "formal C1/C2 manifest SHA-256 mismatch: expected "
                f"{FORMAL_MANIFEST_SHA256}, found {digest}; pass --smoke only "
                "for explicitly non-formal development evidence"
            )
        if len(cases) != FORMAL_CASE_COUNT:
            raise ValueError(
                f"formal C1/C2 cohort must contain {FORMAL_CASE_COUNT} cases"
            )
    return cases


def _safe_structure_path(root: Path, case: Mapping[str, object]) -> Path:
    relative = Path(
        str(case.get("native_structure_path", f"case-{int(case['index']):04d}.json"))
    )
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe native structure path: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"native structure escapes structure root: {relative}")
    return resolved


def _load_native_structure(path: Path, *, case_index: int) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != 1:
        raise ValueError("unsupported native coordination-structure schema")
    if int(payload["case_index"]) != case_index:
        raise ValueError("native structure case_index differs from manifest")
    return payload


def _build_molecule(structure: Mapping[str, object]):
    from dataclasses import replace

    from hotpot.cheminfo.core import BondKind, Molecule
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX

    molecule = Molecule()
    for atom in structure["atoms"]:
        supplied = {}
        if "atom_id" in atom:
            supplied["id"] = int(atom["atom_id"])
        if "implicit_hydrogens" in atom:
            supplied["implicit_hydrogens"] = int(atom["implicit_hydrogens"])
        molecule.create_atom(
            atomic_number=int(atom["atomic_number"]),
            formal_charge=int(atom.get("formal_charge", 0)),
            **supplied,
        )
    for bond in structure["bonds"]:
        molecule.add_bond(
            int(bond["begin"]),
            int(bond["end"]),
            bond_order=float(bond["bond_order"]),
            bond_kind=BondKind(str(bond["bond_kind"])),
            bond_direction=bond.get("bond_direction"),
        )
    record, metadata = molecule._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype="<f8").reshape(record.atom_count, 19).copy()
    presence = np.frombuffer(record.atom_presence, dtype="<u4").copy()
    sources = np.frombuffer(record.atom_sources, dtype="u1").reshape(record.atom_count, 19).copy()
    field = FIELD_INDEX["is_aromatic"]
    values[:, field] = [bool(atom.get("is_aromatic", False)) for atom in structure["atoms"]]
    presence |= 1 << field
    sources[:, field] = ATOM_SOURCE_CODES["explicit"]
    return Molecule._build_molecule(replace(
        record, atom_values=values.tobytes(), atom_presence=presence.tobytes(),
        atom_sources=sources.tobytes(),
        bond_aromatic=tuple(bool(bond.get("is_aromatic", bond["bond_kind"] == "aromatic"))
                            for bond in structure["bonds"]),
    ), bond_metadata=metadata)


def _constraints(structure: Mapping[str, object]):
    from hotpot.cheminfo import rdWrappers

    source = structure["constraints"]
    donor_pairs = np.asarray(
        source.get("donor_pair_indices", []), dtype=np.int32
    ).reshape((-1, 2))
    donor_bounds = np.asarray(
        source.get("donor_pair_bounds_angstrom", []), dtype=np.float64
    ).reshape((-1, 2))
    return rdWrappers.CoordinationEmbeddingConstraints(
        coordination_pairs=np.asarray(
            source["coordination_pairs"], dtype=np.int32
        ).reshape((-1, 2)),
        metal_donor_bounds=np.asarray(
            source["metal_donor_bounds_angstrom"], dtype=np.float64
        ).reshape((-1, 2)),
        donor_pair_indices=donor_pairs if len(donor_pairs) else None,
        donor_pair_bounds=donor_bounds if len(donor_bounds) else None,
    )


def _graph_digest(molecule) -> str:
    from hotpot.cheminfo._native.buffers import pack_molecule
    from hotpot.cheminfo.rdWrappers.native import _RDKIT_ETKDG_FIELDS

    packed = pack_molecule(molecule, _RDKIT_ETKDG_FIELDS)
    value = {
        "atom_count": packed.atom_count,
        "bond_count": packed.bond_count,
        "atomic_numbers": packed.atomic_numbers.tolist(),
        "formal_charges": packed.formal_charges.tolist(),
        "atom_aromatic": packed.atom_aromatic.tolist(),
        "bond_indices": packed.bond_indices.tolist(),
        "bond_kinds": packed.bond_kinds.tolist(),
        "bond_aromatic": packed.bond_aromatic.tolist(),
    }
    return _canonical_digest(value)


def _canonical_topology_digest(molecule) -> str:
    topology = {
        "atoms": [
            {
                "atomic_number": int(atom.atomic_number),
                "formal_charge": int(atom.formal_charge),
            }
            for atom in molecule.atoms
        ],
        "bonds": sorted(
            (
                min(int(bond.atom1.idx), int(bond.atom2.idx)),
                max(int(bond.atom1.idx), int(bond.atom2.idx)),
                float(bond.bond_order),
                bond.bond_kind.value,
            )
            for bond in molecule.bonds
        ),
    }
    return _canonical_digest(topology)


def _bound_observation(coordinates: np.ndarray, constraints) -> dict[str, object]:
    coordinate_rows = np.asarray(coordinates, dtype=np.float64)
    if len(coordinate_rows) == 0:
        return {
            "metal_donor_distance_count": 0,
            "metal_donor_max_violation_angstrom": None,
            "donor_pair_distance_count": 0,
            "donor_pair_max_violation_angstrom": None,
            "strict_bound_satisfied_conformer_count": 0,
            "all_conformers_strictly_bound_satisfied": False,
            "max_bound_violation_angstrom": None,
        }
    pair_sets = (
        (
            "metal_donor",
            np.asarray(constraints.coordination_pairs, dtype=np.int32),
            np.asarray(constraints.metal_donor_bounds, dtype=np.float64),
        ),
        (
            "donor_pair",
            np.empty((0, 2), dtype=np.int32)
            if constraints.donor_pair_indices is None
            else np.asarray(constraints.donor_pair_indices, dtype=np.int32),
            np.empty((0, 2), dtype=np.float64)
            if constraints.donor_pair_bounds is None
            else np.asarray(constraints.donor_pair_bounds, dtype=np.float64),
        ),
    )
    all_violations = []
    result: dict[str, object] = {}
    conformer_satisfied = np.ones(len(coordinate_rows), dtype=bool)
    for name, pairs, bounds in pair_sets:
        if len(pairs) == 0:
            result[f"{name}_distance_count"] = 0
            result[f"{name}_max_violation_angstrom"] = 0.0
            continue
        distances = np.stack(
            [
                np.linalg.norm(
                    coordinate_rows[:, first, :] - coordinate_rows[:, second, :],
                    axis=1,
                )
                for first, second in pairs
            ],
            axis=1,
        )
        violations = np.maximum(
            np.maximum(bounds[None, :, 0] - distances, 0.0),
            np.maximum(distances - bounds[None, :, 1], 0.0),
        )
        conformer_satisfied &= np.all(violations == 0.0, axis=1)
        all_violations.append(violations.reshape(-1))
        result[f"{name}_distance_count"] = int(distances.size)
        result[f"{name}_distance_min_angstrom"] = float(np.min(distances))
        result[f"{name}_distance_max_angstrom"] = float(np.max(distances))
        result[f"{name}_max_violation_angstrom"] = float(np.max(violations))
    result["strict_bound_satisfied_conformer_count"] = int(
        np.count_nonzero(conformer_satisfied)
    )
    result["all_conformers_strictly_bound_satisfied"] = bool(
        np.all(conformer_satisfied)
    )
    result["max_bound_violation_angstrom"] = (
        float(np.max(np.concatenate(all_violations))) if all_violations else 0.0
    )
    return result


def _native_generate(molecule, *, conformer_count, seed, constraints):
    from hotpot.cheminfo import rdWrappers

    return rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(
            conformer_count=conformer_count,
            seed=seed,
            prune_rms_threshold=None,
            use_random_coordinates=False,
        ),
        constraints=constraints,
    )


def _runtime_identity() -> dict[str, object]:
    from hotpot.cheminfo import rdWrappers
    from hotpot.cheminfo.rdWrappers import native

    runtime = rdWrappers.runtime_info()
    backend = native._backend()
    return {
        "hotpot_version": runtime.hotpot_version,
        "rdkit_version": runtime.rdkit_version,
        "rdkit_commit": backend.RDKIT_COMMIT,
        "native": runtime.native,
    }


def _run_route(
    molecule,
    constraints,
    *,
    route: str,
    conformer_count: int,
    seed: int,
    generate: Callable[..., object] = _native_generate,
) -> dict[str, object]:
    graph_before = _graph_digest(molecule)
    started = time.perf_counter_ns()
    route_constraints = None if route == "C1" else constraints
    try:
        result = generate(
            molecule,
            conformer_count=conformer_count,
            seed=seed,
            constraints=route_constraints,
        )
    except Exception as error:
        return {
            "route": route,
            "requested_conformer_count": conformer_count,
            "status": "GENERATION_EXCEPTION",
            "diagnostic": f"{type(error).__name__}: {error}",
            "returned_conformer_count": 0,
            "generate_seconds": (time.perf_counter_ns() - started) / 1.0e9,
            "input_graph_preserved": graph_before == _graph_digest(molecule),
        }
    elapsed = (time.perf_counter_ns() - started) / 1.0e9
    coordinates = np.asarray(result.coordinates, dtype=np.float64)
    rerun_started = time.perf_counter_ns()
    rerun_diagnostic = None
    try:
        rerun = generate(
            molecule,
            conformer_count=conformer_count,
            seed=seed,
            constraints=route_constraints,
        )
    except Exception as error:
        rerun = None
        rerun_diagnostic = f"{type(error).__name__}: {error}"
    rerun_elapsed = (time.perf_counter_ns() - rerun_started) / 1.0e9
    observation = _bound_observation(coordinates, constraints)
    return {
        "route": route,
        "requested_conformer_count": conformer_count,
        "outer_workers": 1,
        "etkdg_internal_threads": 1,
        "status": result.status.name,
        "status_code": int(result.status),
        "diagnostic": result.diagnostic,
        "returned_conformer_count": int(len(coordinates)),
        "generate_seconds": elapsed,
        "determinism_rerun_seconds": rerun_elapsed,
        "same_seed_deterministic": bool(
            rerun is not None
            and result.status == rerun.status
            and result.diagnostic == rerun.diagnostic
            and result.records == rerun.records
            and np.array_equal(
                coordinates,
                np.asarray(rerun.coordinates, dtype=np.float64),
                equal_nan=True,
            )
        ),
        "determinism_rerun_diagnostic": rerun_diagnostic,
        "finite_coordinates": bool(np.isfinite(coordinates).all()),
        "exact_coordinate_shape": coordinates.shape
        == (len(coordinates), len(molecule.atoms), 3),
        "coordinate_sha256": hashlib.sha256(
            np.ascontiguousarray(coordinates, dtype="<f8").tobytes()
        ).hexdigest(),
        "input_graph_preserved": graph_before == _graph_digest(molecule),
        **observation,
    }


def _run_case(
    case: Mapping[str, object],
    structure_path: Path,
    *,
    conformer_counts: Sequence[int],
    generate: Callable[..., object] = _native_generate,
) -> list[dict[str, object]]:
    index = int(case["index"])
    structure = _load_native_structure(structure_path, case_index=index)
    molecule = _build_molecule(structure)
    constraints = _constraints(structure)
    constraint_policy = structure["constraints"]["policy"]
    if constraint_policy.get("distance_unit") != "angstrom":
        raise ValueError("coordination constraint policy must use Angstrom")
    if "atom_count" in case and len(molecule.atoms) != int(case["atom_count"]):
        raise ValueError("native structure atom_count differs from manifest")
    if "bond_count" in case and len(molecule.bonds) != int(case["bond_count"]):
        raise ValueError("native structure bond_count differs from manifest")
    topology_sha256 = _canonical_topology_digest(molecule)
    if "topology_sha256" in case and topology_sha256 != str(case["topology_sha256"]):
        raise ValueError("native structure topology differs from manifest")
    seed = int(case["seed"])
    records = []
    for conformer_count in conformer_counts:
        for route in ("C1", "C2"):
            record = _run_route(
                molecule,
                constraints,
                route=route,
                conformer_count=conformer_count,
                seed=seed,
                generate=generate,
            )
            record.update(
                case_index=index,
                seed=seed,
                structure_path=str(structure_path),
                structure_sha256=_sha256_file(structure_path),
                topology_sha256=topology_sha256,
                constraint_policy=constraint_policy,
                constraint_policy_sha256=_canonical_digest(constraint_policy),
            )
            records.append(record)
    return records


def _aggregate(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    cells = {}
    conformer_counts = sorted(
        {int(record["requested_conformer_count"]) for record in records}
    )
    for route in ("C1", "C2"):
        for count in conformer_counts:
            selected = [
                record
                for record in records
                if record["route"] == route
                and record["requested_conformer_count"] == count
            ]
            key = f"{route}:{count}"
            cells[key] = {
                "case_count": len(selected),
                "status_counts": dict(Counter(r["status"] for r in selected)),
                "returned_conformer_count": sum(
                    int(r["returned_conformer_count"]) for r in selected
                ),
                "strict_bound_satisfied_conformer_count": sum(
                    int(r.get("strict_bound_satisfied_conformer_count", 0))
                    for r in selected
                ),
                "max_bound_violation_angstrom": max(
                    (
                        float(r["max_bound_violation_angstrom"])
                        for r in selected
                        if r.get("max_bound_violation_angstrom") is not None
                    ),
                    default=None,
                ),
                "aggregate_generate_seconds": sum(
                    float(r["generate_seconds"]) for r in selected
                ),
            }
    return cells


def run(
    manifest_path: Path,
    structure_root: Path,
    *,
    smoke: bool,
    conformer_counts: Sequence[int] = DEFAULT_CONFORMER_COUNTS,
    generate: Callable[..., object] = _native_generate,
    runtime_identity: Callable[[], Mapping[str, object]] = _runtime_identity,
) -> dict[str, object]:
    cases = _load_manifest(manifest_path, smoke=smoke)
    structures = [_safe_structure_path(structure_root, case) for case in cases]
    structure_inventory = [
        {
            "case_index": int(case["index"]),
            "path": str(path.relative_to(structure_root.resolve())),
            "sha256": _sha256_file(path),
        }
        for case, path in zip(cases, structures)
    ]
    records = [
        record
        for case, path in zip(cases, structures)
        for record in _run_case(
            case,
            path,
            conformer_counts=conformer_counts,
            generate=generate,
        )
    ]
    return {
        "schema": "hotpot-plan17-c1-c2/1",
        "evidence_class": "development-smoke" if smoke else "formal",
        "formal_claim_allowed": not smoke,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "formal_manifest_sha256": FORMAL_MANIFEST_SHA256,
        "structure_root": str(structure_root.resolve()),
        "structure_inventory_sha256": _canonical_digest(structure_inventory),
        "structure_inventory": structure_inventory,
        "case_count": len(cases),
        "conformer_counts": list(conformer_counts),
        "routes": {
            "C1": "stock whole-complex ETKDGv3 coordinates only",
            "C2": "same graph with coordination bounds coordinates only",
        },
        "input_loader": (
            "direct Hotpot native-graph JSON construction; no molecular-toolkit "
            "file parser"
        ),
        "prohibited_stages": [
            "UFF/MMFF optimization calls",
            "Open Babel API calls",
            "hotpot.ff",
            "Hotpot geometry gate",
            "perturbation",
        ],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "native_runtime": dict(runtime_identity()),
        "runner_sha256": _sha256_file(Path(__file__)),
        "records": records,
        "summary": _aggregate(records),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="accept a non-formal development cohort and label it as smoke",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run(
        args.manifest,
        args.structure_root,
        smoke=args.smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
