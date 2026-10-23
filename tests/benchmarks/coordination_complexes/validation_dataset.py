"""Publish Plan-17 route results through the governed Hotpot dataset path.

Preparation is intentionally separate from publication.  ``prepare_bundle``
materializes an ignored Bundle root and never opens the target store.
``publish_bundle`` is the only ingest entry point and requires an explicit
confirmation flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


ROUTES = ("O0", "F1", "F2", "C1", "C2")
NO_CBOND_INDICES = frozenset({139, 182, 185, 186, 187})
FORMAL_CASE_INDICES = tuple(
    index for index in range(1, 188) if index not in NO_CBOND_INDICES
)
SOURCE_SHA256 = "e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7"
DATASET_NAME = "cbond-forcefield"
RELEASE_ID = "cbond-forcefield.v001"
VERSION = "001"
RESULT_SCHEMA = "hotpot-cbond-forcefield-route-result/current"
INVENTORY_SCHEMA = "hotpot-cbond-forcefield-inventory-candidate/current"
NONFINITE_KEY = "nonfinite_float"
ACCEPTANCE_IDS = tuple(f"A{index:02d}" for index in range(1, 11))
STRUCTURE_FORMATS = frozenset({"mol", "mol2", "sdf", "smi", "smiles", "xyz"})

_RECORD_FIELDS = {
    "case_index",
    "source_smiles",
    "source_sha256",
    "semantic_graph_sha256",
    "route",
    "run_id",
    "seed",
    "status",
    "passed",
    "selected_structure",
    "acceptance",
    "optimization",
    "provenance",
    "settings",
    "manual_review",
}
_REQUIRED_RECORD_FIELDS = _RECORD_FIELDS - {"selected_structure"}
_SELECTED_STRUCTURE_FIELDS = {
    "path",
    "sha256",
    "format",
    "atom_row_mapping",
}
_PROVENANCE_FIELDS = {
    "validation_round",
    "hotpot_commit",
    "dependency_versions",
    "controls",
}
_AUTHORIZATION_FIELDS = {
    "authorization_id",
    "authority",
    "actor",
    "approved",
    "reason",
    "authorized_at",
    "license",
    "access_policy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def normalize_nonfinite_json(value: object) -> object:
    """Return strict JSON data with every non-finite float made explicit."""
    if isinstance(value, float) and not math.isfinite(value):
        token = "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
        return {NONFINITE_KEY: token}
    if isinstance(value, Mapping):
        return {
            str(key): normalize_nonfinite_json(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_nonfinite_json(item) for item in value]
    return value


def route_case_id(case_index: int, route: str) -> str:
    """Return the permanent route-case identity."""
    return f"cbond-ff-{case_index:03d}-{route.lower()}"


def route_case_public_index(
    case_index: int,
    route: str,
    *,
    case_indices: Sequence[int] = FORMAL_CASE_INDICES,
) -> int:
    """Return source-major, fixed-route-order public ordering."""
    return case_indices.index(case_index) * len(ROUTES) + ROUTES.index(route)


def _require_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_authorization(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != _AUTHORIZATION_FIELDS:
        raise ValueError(
            "owner authorization fields must be exactly "
            f"{sorted(_AUTHORIZATION_FIELDS)}"
        )
    normalized = dict(value)
    if normalized["authority"] != "User":
        raise ValueError("benchmark authorization authority must be User")
    if normalized["approved"] is not True:
        raise ValueError("benchmark authorization must explicitly approve publication")
    for field in (
        "authorization_id",
        "actor",
        "reason",
        "authorized_at",
        "license",
        "access_policy",
    ):
        if not isinstance(normalized[field], str) or not normalized[field].strip():
            raise ValueError(f"owner authorization {field} must be non-empty")
    return normalized


def _validate_acceptance(value: object, *, passed: bool) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("acceptance must be an object")
    expected = set(ACCEPTANCE_IDS) | {"diagnostics", "recovery_actions"}
    if set(value) != expected:
        raise ValueError(f"acceptance fields must be exactly {sorted(expected)}")
    decisions = {identifier: value[identifier] for identifier in ACCEPTANCE_IDS}
    if any(type(decision) is not bool for decision in decisions.values()):
        raise ValueError("A01-A10 decisions must be Boolean")
    if passed and not all(decisions.values()):
        raise ValueError("a passed result requires every A01-A10 decision to pass")
    if not isinstance(value["diagnostics"], list):
        raise ValueError("acceptance.diagnostics must be an array")
    if not isinstance(value["recovery_actions"], list):
        raise ValueError("acceptance.recovery_actions must be an array")
    return dict(value)


def _validate_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise ValueError(
            f"provenance fields must be exactly {sorted(_PROVENANCE_FIELDS)}"
        )
    result = dict(value)
    for field in ("validation_round", "hotpot_commit"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"provenance.{field} must be non-empty")
    if not isinstance(result["dependency_versions"], Mapping):
        raise ValueError("provenance.dependency_versions must be an object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in result["dependency_versions"].items()
    ):
        raise ValueError("provenance dependency names and versions must be non-empty")
    controls = result["controls"]
    if not isinstance(controls, list):
        raise ValueError("provenance.controls must be an array")
    for offset, control in enumerate(controls):
        if not isinstance(control, Mapping) or set(control) != {
            "name",
            "sha256",
            "locator",
        }:
            raise ValueError(
                f"provenance.controls[{offset}] requires name, sha256, and locator"
            )
        if any(
            not isinstance(control[field], str) or not control[field]
            for field in ("name", "locator")
        ):
            raise ValueError(
                f"provenance.controls[{offset}] name and locator must be non-empty"
            )
        _require_digest(control["sha256"], f"provenance.controls[{offset}].sha256")
    return result


def _validate_record(
    value: Mapping[str, object],
    *,
    source_lines: Sequence[str],
    source_sha256: str,
) -> dict[str, object]:
    fields = set(value)
    missing = _REQUIRED_RECORD_FIELDS - fields
    extra = fields - _RECORD_FIELDS
    if missing or extra:
        raise ValueError(
            f"route result has missing={sorted(missing)} extra={sorted(extra)} fields"
        )
    case_index = value["case_index"]
    if not isinstance(case_index, int) or isinstance(case_index, bool):
        raise ValueError("case_index must be an integer")
    if not 1 <= case_index <= len(source_lines):
        raise ValueError(f"case_index {case_index} is outside the source corpus")
    route = value["route"]
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}")
    source_smiles = value["source_smiles"]
    if source_smiles != source_lines[case_index - 1]:
        raise ValueError(f"case {case_index} SMILES differs from the source corpus")
    if value["source_sha256"] != source_sha256:
        raise ValueError(f"case {case_index} source SHA-256 differs from the corpus")
    _require_digest(value["semantic_graph_sha256"], "semantic_graph_sha256")
    if not isinstance(value["run_id"], str) or not value["run_id"].strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        raise ValueError("seed must be an integer")
    if not isinstance(value["status"], str) or not value["status"].strip():
        raise ValueError("status must be non-empty")
    if type(value["passed"]) is not bool:
        raise ValueError("passed must be Boolean")
    passed = bool(value["passed"])
    if (value["status"] == "passed") != passed:
        raise ValueError("status='passed' and passed must agree")
    _validate_acceptance(value["acceptance"], passed=passed)
    if not isinstance(value["optimization"], Mapping):
        raise ValueError("optimization must be an object")
    if set(value["optimization"]) != {
        "best_energy",
        "rms_gradient",
        "max_gradient",
    }:
        raise ValueError("optimization requires only the three selected-frame scalars")
    if passed and any(
        not isinstance(value["optimization"][field], (int, float))
        or isinstance(value["optimization"][field], bool)
        or not math.isfinite(value["optimization"][field])
        for field in ("best_energy", "rms_gradient", "max_gradient")
    ):
        raise ValueError("passed selected-frame optimization values must be finite")
    _validate_provenance(value["provenance"])
    if not isinstance(value["settings"], Mapping):
        raise ValueError("settings must be an object")
    cbond = value["settings"].get("cbond")
    if not isinstance(cbond, Mapping) or set(cbond) != {
        "metal_index",
        "donor_indices",
    }:
        raise ValueError("settings.cbond requires metal_index and donor_indices")
    if (
        not isinstance(cbond["metal_index"], int)
        or isinstance(cbond["metal_index"], bool)
        or cbond["metal_index"] < 0
        or not isinstance(cbond["donor_indices"], list)
        or not cbond["donor_indices"]
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in cbond["donor_indices"]
        )
    ):
        raise ValueError("settings.cbond mapping is invalid")
    review = value["manual_review"]
    if not isinstance(review, Mapping) or not isinstance(review.get("status"), str):
        raise ValueError("manual_review requires a status")
    if not isinstance(review.get("flags"), list):
        raise ValueError("manual_review.flags must be an array")

    selected = value.get("selected_structure")
    if passed:
        if (
            not isinstance(selected, Mapping)
            or set(selected) != _SELECTED_STRUCTURE_FIELDS
        ):
            raise ValueError(
                "a passed result requires exactly path, sha256, format, and "
                "atom_row_mapping in selected_structure"
            )
        _require_digest(selected["sha256"], "selected_structure.sha256")
        if selected["format"] not in STRUCTURE_FORMATS:
            raise ValueError("selected_structure.format is unsupported by hotpot.ds")
        if not isinstance(selected["path"], str) or not selected["path"]:
            raise ValueError("selected_structure.path must be non-empty")
        mapping = selected["atom_row_mapping"]
        if (
            not isinstance(mapping, list)
            or not mapping
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in mapping
            )
            or len(mapping) != len(set(mapping))
        ):
            raise ValueError(
                "selected_structure.atom_row_mapping must be non-empty unique "
                "non-negative integers"
            )
        if review["status"] != "passed" or review["flags"]:
            raise ValueError(
                "a passed result requires completed, unflagged manual review"
            )
    elif selected is not None:
        raise ValueError("a failed result must not contain selected_structure")
    return dict(value)


def validate_route_results(
    records: Iterable[Mapping[str, object]],
    source_path: Path,
    *,
    case_indices: Sequence[int] = FORMAL_CASE_INDICES,
    expected_source_sha256: Optional[str] = SOURCE_SHA256,
) -> tuple[dict[str, object], ...]:
    """Validate and canonically order one complete route-case matrix."""
    source_path = source_path.resolve()
    source_sha256 = _sha256(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("source corpus SHA-256 differs from the publication contract")
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    if any(not line for line in source_lines):
        raise ValueError("source corpus must contain only non-empty physical lines")
    normalized: dict[tuple[int, str], dict[str, object]] = {}
    for raw in records:
        record = _validate_record(
            raw,
            source_lines=source_lines,
            source_sha256=source_sha256,
        )
        key = (int(record["case_index"]), str(record["route"]))
        if key in normalized:
            raise ValueError(f"duplicate route-case result: {key}")
        normalized[key] = record
    expected = {(index, route) for index in case_indices for route in ROUTES}
    actual = set(normalized)
    if actual != expected:
        raise ValueError(
            f"route-case matrix mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return tuple(
        normalized[(index, route)] for index in case_indices for route in ROUTES
    )


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _control_asset(
    path: Path, root: Path, category: str, name: str
) -> dict[str, object]:
    return {
        "category": category,
        "name": name,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def _result_case(
    record: Mapping[str, object],
    *,
    public_index: int,
    structure: Optional[Mapping[str, object]],
    authorization: Mapping[str, object],
) -> dict[str, object]:
    case_index = int(record["case_index"])
    route = str(record["route"])
    case_id = route_case_id(case_index, route)
    case = {
        "id": case_id,
        "local_id": case_id,
        "public_index": public_index,
        "identifiers": {
            "cbond_forcefield_route_case": case_id,
            "source_line_index": str(case_index),
        },
        "core_info": {
            "source_line_index": case_index,
            "source_smiles": record["source_smiles"],
            "route": route,
            "semantic_graph_sha256": record["semantic_graph_sha256"],
            "terminal_status": record["status"],
        },
        "metadata": {
            "schema": RESULT_SCHEMA,
            "source_sha256": record["source_sha256"],
            "semantic_graph_sha256": record["semantic_graph_sha256"],
            "run_id": record["run_id"],
            "seed": record["seed"],
            "passed": record["passed"],
            "acceptance": normalize_nonfinite_json(record["acceptance"]),
            "optimization": normalize_nonfinite_json(record["optimization"]),
            "provenance": normalize_nonfinite_json(record["provenance"]),
            "settings": normalize_nonfinite_json(record["settings"]),
            "manual_review": normalize_nonfinite_json(record["manual_review"]),
        },
        "structures": [] if structure is None else [dict(structure)],
        "attestations": [
            {
                "attestation_id": (
                    f"attestation-{case_id}-"
                    f"{authorization['authorization_id']}"
                ),
                "authority": "User",
                "actor": authorization["actor"],
                "approved": True,
                "reason": authorization["reason"],
                "status": "active",
                "authorization_id": authorization["authorization_id"],
                "authorized_at": authorization["authorized_at"],
            }
        ],
    }
    return case


def prepare_bundle(
    records: Iterable[Mapping[str, object]],
    *,
    source_path: Path,
    bundle_root: Path,
    authorization: Mapping[str, object],
    schema_path: Optional[Path] = None,
    result_root: Optional[Path] = None,
    case_indices: Sequence[int] = FORMAL_CASE_INDICES,
    expected_source_sha256: Optional[str] = SOURCE_SHA256,
) -> dict[str, object]:
    """Create an unbuilt Bundle without touching a target dataset store."""
    bundle_root = bundle_root.resolve()
    if bundle_root.exists():
        raise FileExistsError(bundle_root)
    authorization = _validate_authorization(authorization)
    ordered = validate_route_results(
        records,
        source_path,
        case_indices=case_indices,
        expected_source_sha256=expected_source_sha256,
    )
    result_root = Path.cwd() if result_root is None else result_root.resolve()
    schema_path = (
        Path(__file__).parents[3]
        / "hotpot"
        / "dataset"
        / "schemas"
        / "cbond-forcefield-benchmark.json"
        if schema_path is None
        else schema_path.resolve()
    )

    selected_sources: dict[tuple[int, str], Path] = {}
    for record in ordered:
        selected = record.get("selected_structure")
        if selected is None:
            continue
        source = Path(str(selected["path"])).expanduser()
        source = source if source.is_absolute() else result_root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if _sha256(source) != selected["sha256"]:
            raise ValueError(f"selected structure SHA-256 mismatch: {source}")
        selected_sources[(int(record["case_index"]), str(record["route"]))] = source

    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{bundle_root.name}.", dir=bundle_root.parent)
    )
    try:
        controls = staging / "evidence" / "control"
        controls.mkdir(parents=True)
        source_copy = controls / "extractants.smi"
        authorization_copy = controls / "owner-authorization.json"
        schema_copy = controls / "cbond-forcefield-benchmark.schema.json"
        shutil.copyfile(source_path.resolve(), source_copy)
        _write(authorization_copy, _canonical_bytes(authorization))
        shutil.copyfile(schema_path, schema_copy)

        normalized_records = []
        cases = []
        route_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for public_index, record in enumerate(ordered):
            case_index = int(record["case_index"])
            route = str(record["route"])
            case_id = route_case_id(case_index, route)
            selected = record.get("selected_structure")
            structure = None
            normalized_record = dict(record)
            if selected is not None:
                fmt = str(selected["format"]).lower()
                extension = "smi" if fmt == "smiles" else fmt
                relative = Path("structures") / f"{case_id}.{extension}"
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(selected_sources[(case_index, route)], destination)
                structure = {
                    "structure_id": f"structure-{case_id}",
                    "role": "selected-forcefield",
                    "format": fmt,
                    "path": relative.as_posix(),
                    "primary": True,
                    "sha256": selected["sha256"],
                    "atom_row_mapping": selected["atom_row_mapping"],
                }
                normalized_record["selected_structure"] = {
                    **selected,
                    "path": relative.as_posix(),
                }
            normalized_record = normalize_nonfinite_json(normalized_record)
            normalized_records.append(normalized_record)
            cases.append(
                _result_case(
                    record,
                    public_index=public_index,
                    structure=structure,
                    authorization=authorization,
                )
            )
            route_counts[route] += 1
            status_counts["passed" if record["passed"] else "failed"] += 1

        route_results = controls / "route-results.jsonl"
        route_results.write_bytes(
            b"".join(_canonical_bytes(record) for record in normalized_records)
        )
        intake = staging / "evidence" / "intake" / "cases.jsonl"
        intake.parent.mkdir(parents=True, exist_ok=True)
        intake.write_bytes(b"".join(_canonical_bytes(case) for case in cases))
        matrix_summary = {
            "schema": INVENTORY_SCHEMA,
            "dataset": DATASET_NAME,
            "release": RELEASE_ID,
            "source_sha256": _sha256(source_copy),
            "source_case_count": len(case_indices),
            "routes": list(ROUTES),
            "route_case_count": len(cases),
            "route_counts": dict(sorted(route_counts.items())),
            "result_status": dict(sorted(status_counts.items())),
            "coordinate_assets": status_counts["passed"],
            "missing_coordinate_assets": status_counts["failed"],
            "authorization_id": authorization["authorization_id"],
        }
        summary_copy = controls / "matrix-summary.json"
        _write(summary_copy, _canonical_bytes(matrix_summary))
        control_assets = [
            _control_asset(source_copy, staging, "source", "extractants-187"),
            _control_asset(
                route_results, staging, "validation", "route-results-matrix"
            ),
            _control_asset(authorization_copy, staging, "authorization", "owner-user"),
            _control_asset(schema_copy, staging, "schema", "result-schema"),
            _control_asset(summary_copy, staging, "inventory", "matrix-summary"),
        ]
        manifest = {
            "schema": "hotpot-ds-bundle/current",
            "ds_name": DATASET_NAME,
            "dataset_id": DATASET_NAME,
            "release_id": RELEASE_ID,
            "release_serial": 1,
            "version": VERSION,
            "title": "Hotpot CBond to force-field validation structures",
            "description": (
                "Owner-authorized O0/F1/F2/C1/C2 route-case results; failed "
                "results remain explicit and have no selected structure."
            ),
            "license": authorization["license"],
            "access_policy": authorization["access_policy"],
            "provenance": {
                "created_by": "Plan-17 validation dataset publisher",
                "authorization": {
                    "authority": "User",
                    "actor": authorization["actor"],
                    "authorization_id": authorization["authorization_id"],
                },
                "source_sha256": _sha256(source_copy),
                "result_schema_sha256": _sha256(schema_copy),
            },
            "core_labels": [],
            "core_info": [
                "source_line_index",
                "source_smiles",
                "route",
                "semantic_graph_sha256",
                "terminal_status",
            ],
            "intake": {"format": "jsonl", "path": "evidence/intake/cases.jsonl"},
            "identity_schema": {
                "components": [
                    {
                        "key": "source-line-index",
                        "path": "core_info.source_line_index",
                        "required": True,
                        "normalizer": "number",
                    },
                    {
                        "key": "route",
                        "path": "core_info.route",
                        "required": True,
                        "normalizer": "text",
                    },
                    {
                        "key": "semantic-graph-sha256",
                        "path": "core_info.semantic_graph_sha256",
                        "required": True,
                        "normalizer": "text",
                    },
                ]
            },
            "required_structure_roles": ["selected-forcefield"],
            "control_assets": control_assets,
        }
        _write(staging / "bundle.json", _canonical_bytes(manifest))
        os.replace(staging, bundle_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return matrix_summary


def _certificate_sha256(bundle: object) -> str:
    return _sha256(Path(bundle.certificate_path))


def inventory_candidate(
    bundle: object, receipt: Mapping[str, object]
) -> dict[str, object]:
    """Build an object suitable for review beside tracked current-inventory data."""
    matrix = json.loads(
        (bundle.root / "evidence" / "control" / "matrix-summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = bundle.summary
    return {
        "schema": INVENTORY_SCHEMA,
        "release": {
            "dataset": bundle.name,
            "release": bundle.info["release_id"],
            "version": bundle.info["version"],
            "release_serial": 1,
            "cases": len(bundle),
            "archive_class": summary["completeness"]["archive_class"],
            "quality_level": summary["quality"]["level"],
            "evidence_level": summary["evidence"]["level"],
            "generation_digest": bundle.status["generation_digest"],
            "certificate_sha256": _certificate_sha256(bundle),
            "receipt_id": receipt["receipt_id"],
        },
        "benchmark_matrix": matrix,
        "ingestion": {
            "include_incomplete": True,
            "reason": receipt["reason"],
            "included_count": receipt["included_count"],
            "excluded_count": receipt["excluded_count"],
        },
    }


def publish_bundle(
    bundle_root: Path,
    *,
    database: Path,
    reason: str,
    confirm_ingest: bool = False,
    inventory_output: Optional[Path] = None,
) -> dict[str, object]:
    """Synchronize, certify, and explicitly ingest a prepared Bundle."""
    if not confirm_ingest:
        raise PermissionError("publication requires confirm_ingest=True")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("include-incomplete publication requires a non-empty reason")
    import hotpot.ds as ds

    bundle_root = bundle_root.resolve()
    database = database.resolve()
    if (bundle_root / "cases" / "facts.sqlite3").is_file():
        bundle = ds.load_bundle(bundle_root, DATASET_NAME, db=database)
    else:
        bundle = ds.Bundle(bundle_root, DATASET_NAME, db=database)
    if not bundle.is_synchronized:
        bundle.synchronize()
    bundle.write_certificate()
    receipt = {
        **bundle.ingest(include_incomplete=True, reason=reason.strip()),
        "include_incomplete": True,
        "reason": reason.strip(),
    }
    inventory = inventory_candidate(bundle, receipt)
    if inventory_output is not None:
        inventory_output = inventory_output.resolve()
        _write(inventory_output, _canonical_bytes(inventory))
    return {"receipt": receipt, "inventory": inventory}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="materialize an unbuilt Bundle")
    prepare.add_argument("--results", type=Path, required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--authorization", type=Path, required=True)
    prepare.add_argument("--bundle-root", type=Path, required=True)
    prepare.add_argument("--result-root", type=Path)
    publish = commands.add_parser("publish", help="synchronize, certify, and ingest")
    publish.add_argument("--bundle-root", type=Path, required=True)
    publish.add_argument("--database", type=Path, required=True)
    publish.add_argument("--reason", required=True)
    publish.add_argument("--inventory-output", type=Path, required=True)
    publish.add_argument("--confirm-ingest", action="store_true", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        summary = prepare_bundle(
            _load_jsonl(args.results),
            source_path=args.source,
            bundle_root=args.bundle_root,
            authorization=_load_json(args.authorization),
            result_root=args.result_root,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    result = publish_bundle(
        args.bundle_root,
        database=args.database,
        reason=args.reason,
        confirm_ingest=args.confirm_ingest,
        inventory_output=args.inventory_output,
    )
    print(json.dumps(result["inventory"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FORMAL_CASE_INDICES",
    "ROUTES",
    "SOURCE_SHA256",
    "inventory_candidate",
    "normalize_nonfinite_json",
    "prepare_bundle",
    "publish_bundle",
    "route_case_id",
    "route_case_public_index",
    "validate_route_results",
)
