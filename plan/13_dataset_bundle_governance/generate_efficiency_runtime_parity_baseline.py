#!/usr/bin/env python3
"""Generate the read-only Stage 13 pre-optimization parity baseline.

The generator intentionally uses only Python's standard library and opens every
SQLite database with ``mode=ro&immutable=1``.  It never imports ``hotpot.ds`` so
the baseline remains independent of the implementation being measured.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA = "hotpot-ds-efficiency-runtime-parity-baseline/v1"
BUNDLE_RELATIVE_ROOTS = (
    "hotpot/dataset/data/bundles/stage13-migration/oxidation/oxidation.v001",
    "hotpot/dataset/data/bundles/stage13-migration/spin/spin.v001",
    "hotpot/dataset/data/bundles/stage13-migration/"
    "csd-electronic-state/csd-electronic-state.vreported-text-v1",
)
STORE_RELATIVE_PATH = "hotpot/dataset/data/store/active/datasets.sqlite3"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def update_framed_value(digest: Any, value: Any) -> None:
    if value is None:
        tag, payload = b"N", b""
    elif isinstance(value, bytes):
        tag, payload = b"B", value
    elif isinstance(value, int):
        tag, payload = b"I", str(value).encode("ascii")
    elif isinstance(value, float):
        tag, payload = b"F", repr(value).encode("ascii")
    else:
        tag, payload = b"T", str(value).encode("utf-8")
    digest.update(tag)
    digest.update(str(len(payload)).encode("ascii"))
    digest.update(b":")
    digest.update(payload)


def update_framed_row(digest: Any, row: Sequence[Any]) -> None:
    digest.update(b"R")
    for value in row:
        update_framed_value(digest, value)
    digest.update(b";")


class FileHasher:
    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, Any]] = {}

    def hash(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        cached = self._cache.get(path)
        if cached is not None:
            return dict(cached)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        result = {"sha256": digest.hexdigest(), "size": size}
        self._cache[path] = result
        return dict(result)

    def remember(self, path: Path, sha256: str, size: int) -> None:
        self._cache[path.resolve()] = {"sha256": sha256, "size": size}


def readonly_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve().as_posix()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def sqlite_identity(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        "application_id": connection.execute("PRAGMA application_id").fetchone()[0],
        "encoding": connection.execute("PRAGMA encoding").fetchone()[0],
        "schema_version": connection.execute("PRAGMA schema_version").fetchone()[0],
        "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
    }


def table_projection(
    connection: sqlite3.Connection,
    table: str,
    *,
    where: str = "",
    parameters: Sequence[Any] = (),
    excluded_columns: Iterable[str] = (),
) -> dict[str, Any]:
    schema = list(connection.execute(f'PRAGMA table_info("{table}")'))
    excluded = set(excluded_columns)
    columns = [row[1] for row in schema if row[1] not in excluded]
    primary_key = [
        row[1] for row in sorted(schema, key=lambda item: item[5]) if row[5] > 0
    ]
    order_columns = primary_key or columns
    select = ",".join(f'"{column}"' for column in columns)
    order = ",".join(f'"{column}"' for column in order_columns)
    query = f'SELECT {select} FROM "{table}"'
    if where:
        query += f" WHERE {where}"
    query += f" ORDER BY {order}"
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(query, tuple(parameters)):
        update_framed_row(digest, tuple(row))
        count += 1
    return {
        "columns": columns,
        "count": count,
        "order_by": order_columns,
        "sha256": digest.hexdigest(),
    }


def canonical_line_digest(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for record in records:
        digest.update(canonical_json(record))
        digest.update(b"\n")
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def iter_asset_descriptors(record: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if "path" in record or record.get("remote_not_materialized"):
        yield dict(record)
    assets = record.get("assets", [])
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, Mapping):
                yield dict(asset)
    representations = record.get("representations", {})
    if isinstance(representations, Mapping):
        for format_name, raw in representations.items():
            if isinstance(raw, str):
                yield {"format": format_name, "path": raw}
            elif isinstance(raw, Mapping):
                yield {"format": format_name, **dict(raw)}
    elif isinstance(representations, list):
        for raw in representations:
            if isinstance(raw, Mapping):
                yield dict(raw)


def record_id(record: Mapping[str, Any], collection: str) -> str:
    field = "structure_id" if collection == "structures" else "evidence_id"
    return str(record.get(field, ""))


def scan_case_stream(
    path: Path, root: Path, file_hasher: FileHasher
) -> dict[str, Any]:
    raw_digest = hashlib.sha256()
    logical_digest = hashlib.sha256()
    identifier_digest = hashlib.sha256()
    declaration_digest = hashlib.sha256()
    structure_ref_digest = hashlib.sha256()
    scientific_ref_digest = hashlib.sha256()
    identifier_namespaces: Counter[str] = Counter()
    declaration_states: Counter[str] = Counter()
    declaration_collections: Counter[str] = Counter()
    unique_materialized: dict[str, int] = {}
    counts = Counter()
    next_index = 0

    with path.open("rb") as stream:
        for raw_line in stream:
            raw_digest.update(raw_line)
            case = json.loads(raw_line)
            normalized = canonical_json(case) + b"\n"
            logical_digest.update(normalized)
            counts["cases"] += 1
            counts["bytes"] += len(raw_line)
            if int(case["public_index"]) != next_index:
                raise RuntimeError(f"non-contiguous public_index in {path}")
            next_index += 1
            case_id = str(case["id"])

            identifiers = case.get("identifiers", {})
            for namespace in sorted(identifiers):
                raw_values = identifiers[namespace]
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                for value in sorted(str(item) for item in values):
                    identifier_digest.update(
                        canonical_json(
                            {
                                "case_id": case_id,
                                "namespace": namespace,
                                "value": value,
                            }
                        )
                        + b"\n"
                    )
                    identifier_namespaces[namespace] += 1
                    counts["identifiers"] += 1

            case_scientific_records = []
            case_structure_records = []
            for collection in ("structures", "evidence"):
                for record in case.get(collection, []):
                    declaration_id = record_id(record, collection)
                    if collection == "structures":
                        counts["structure_records"] += 1
                    for asset_index, asset in enumerate(iter_asset_descriptors(record)):
                        path_text = asset.get("path")
                        materialized = bool(
                            path_text and (root / str(path_text)).resolve().is_file()
                        )
                        state = "present" if materialized else (
                            "remote_not_materialized"
                            if not path_text
                            else "missing"
                        )
                        declaration = {
                            "asset_index": asset_index,
                            "case_id": case_id,
                            "collection": collection,
                            "declaration_id": declaration_id,
                            "expected_sha256": asset.get("expected_sha256")
                            or asset.get("sha256"),
                            "format": asset.get("format"),
                            "path": path_text,
                            "role": str(
                                asset.get("role")
                                or asset.get("format")
                                or record.get("role")
                                or "asset"
                            ),
                            "state": state,
                            "status": str(record.get("status", "active")),
                        }
                        declaration_digest.update(canonical_json(declaration) + b"\n")
                        declaration_states[state] += 1
                        declaration_collections[collection] += 1
                        counts["asset_declarations"] += 1
                        if not materialized:
                            continue
                        asset_path = (root / str(path_text)).resolve()
                        file_info = file_hasher.hash(asset_path)
                        expected = declaration["expected_sha256"]
                        if expected and expected != file_info["sha256"]:
                            raise RuntimeError(
                                f"declared asset digest mismatch: {asset_path}"
                            )
                        unique_materialized[file_info["sha256"]] = file_info["size"]
                        scientific = {
                            "asset_index": asset_index,
                            "case_id": case_id,
                            "collection": collection,
                            "content_sha256": file_info["sha256"],
                            "declaration_id": declaration_id,
                            "role": declaration["role"],
                            "size": file_info["size"],
                            "status": declaration["status"],
                        }
                        case_scientific_records.append(scientific)
                        counts["materialized_scientific_refs"] += 1

                        if collection != "structures" or record.get(
                            "status", "active"
                        ) != "active":
                            continue
                        structure = {
                            "case_id": case_id,
                            "content_sha256": file_info["sha256"],
                            "format": str(
                                asset.get("format")
                                or record.get("format")
                                or asset_path.suffix.lstrip(".")
                            ),
                            "is_primary": int(bool(record.get("primary", False))),
                            "role": str(record.get("role", "structure")),
                            "size": file_info["size"],
                            "structure_id": str(record["structure_id"]),
                        }
                        case_structure_records.append(structure)
                        counts["materialized_structure_refs"] += 1

            for scientific in sorted(
                case_scientific_records,
                key=lambda item: (
                    item["collection"],
                    item["declaration_id"],
                    item["asset_index"],
                ),
            ):
                scientific_ref_digest.update(canonical_json(scientific) + b"\n")
            for structure in sorted(
                case_structure_records,
                key=lambda item: (
                    item["structure_id"],
                    item["format"],
                    item["role"],
                    item["content_sha256"],
                ),
            ):
                structure_ref_digest.update(canonical_json(structure) + b"\n")

    raw_sha256 = raw_digest.hexdigest()
    file_hasher.remember(path, raw_sha256, counts["bytes"])
    return {
        "asset_declarations": {
            "by_collection": dict(sorted(declaration_collections.items())),
            "by_state": dict(sorted(declaration_states.items())),
            "count": counts["asset_declarations"],
            "sha256": declaration_digest.hexdigest(),
        },
        "case_count": counts["cases"],
        "identifiers": {
            "by_namespace": dict(sorted(identifier_namespaces.items())),
            "count": counts["identifiers"],
            "sha256": identifier_digest.hexdigest(),
        },
        "logical_canonical_jsonl_sha256": logical_digest.hexdigest(),
        "materialized_assets": {
            "scientific_references": {
                "count": counts["materialized_scientific_refs"],
                "sha256": scientific_ref_digest.hexdigest(),
            },
            "structure_references": {
                "count": counts["materialized_structure_refs"],
                "sha256": structure_ref_digest.hexdigest(),
            },
            "unique_content": {
                "bytes": sum(unique_materialized.values()),
                "count": len(unique_materialized),
                "sha256": canonical_digest(
                    [
                        {"content_sha256": digest, "size": size}
                        for digest, size in sorted(unique_materialized.items())
                    ]
                ),
            },
        },
        "raw_file": {"sha256": raw_sha256, "size": counts["bytes"]},
        "structure_record_count": counts["structure_records"],
    }


def facts_state(path: Path) -> dict[str, Any]:
    with readonly_sqlite(path) as connection:
        control = dict(connection.execute("SELECT * FROM control").fetchone())
        digest = hashlib.sha256()
        case_ids = hashlib.sha256()
        count = 0
        invalid_json = 0
        for row in connection.execute(
            "SELECT public_index,case_id,local_id,payload_json "
            "FROM cases ORDER BY public_index"
        ):
            digest.update(row["payload_json"].encode("utf-8") + b"\n")
            case_ids.update(
                canonical_json(
                    {
                        "case_id": row["case_id"],
                        "local_id": row["local_id"],
                        "public_index": row["public_index"],
                    }
                )
                + b"\n"
            )
            count += 1
        invalid_json = connection.execute(
            "SELECT count(*) FROM cases WHERE NOT json_valid(payload_json)"
        ).fetchone()[0]
        return {
            "case_count": count,
            "case_identity_sha256": case_ids.hexdigest(),
            "control": control,
            "invalid_payload_json_count": invalid_json,
            "payload_jsonl_sha256": digest.hexdigest(),
            "sqlite": sqlite_identity(connection),
        }


def assessment_state(path: Path) -> dict[str, Any]:
    tables = (
        "asset_assessments",
        "case_assessments",
        "conflict_assessments",
        "conflict_case_links",
        "conflict_groups",
        "evidence_assessments",
    )
    with readonly_sqlite(path) as connection:
        projections = {
            table: table_projection(connection, table) for table in tables
        }
        distributions = {
            "archive_class": dict(
                connection.execute(
                    "SELECT archive_class,count(*) FROM case_assessments "
                    "GROUP BY archive_class ORDER BY archive_class"
                ).fetchall()
            ),
            "benchmark": {
                str(key): value
                for key, value in connection.execute(
                    "SELECT benchmark,count(*) FROM case_assessments "
                    "GROUP BY benchmark ORDER BY benchmark"
                )
            },
            "evidence_level": {
                str(key): value
                for key, value in connection.execute(
                    "SELECT evidence_level,count(*) FROM case_assessments "
                    "GROUP BY evidence_level ORDER BY evidence_level"
                )
            },
            "quality_level": dict(
                connection.execute(
                    "SELECT quality_level,count(*) FROM case_assessments "
                    "GROUP BY quality_level ORDER BY quality_level"
                ).fetchall()
            ),
        }
        asset_distribution = [
            {
                "count": count,
                "hash_result": hash_result,
                "parse_result": parse_result,
                "presence": presence,
            }
            for presence, hash_result, parse_result, count in connection.execute(
                "SELECT presence,hash_result,parse_result,count(*) "
                "FROM asset_assessments GROUP BY presence,hash_result,parse_result "
                "ORDER BY presence,hash_result,parse_result"
            )
        ]
        return {
            "asset_result_distribution": asset_distribution,
            "case_distributions": distributions,
            "projections": projections,
            "sqlite": sqlite_identity(connection),
        }


def generation_root(bundle_root: Path) -> Path:
    generations = sorted((bundle_root / "assessments" / "generations").glob("*"))
    if len(generations) != 1:
        raise RuntimeError(f"expected one generation in {bundle_root}")
    return generations[0]


def input_file_inventory(
    repository_root: Path,
    bundle_root: Path,
    generation: Path,
    manifest: Mapping[str, Any],
    file_hasher: FileHasher,
) -> list[dict[str, Any]]:
    paths = {
        bundle_root / "bundle.json",
        bundle_root / "cases.jsonl",
        bundle_root / "cases" / "facts.sqlite3",
        bundle_root / "policy" / "policy.json",
        bundle_root / "certificate.json",
        bundle_root / "ingestion-receipt.json",
    }
    paths.update(path for path in generation.rglob("*") if path.is_file())
    paths.update(
        bundle_root / descriptor["path"]
        for descriptor in manifest.get("control_assets", [])
    )
    result = []
    for path in sorted(paths):
        info = file_hasher.hash(path)
        result.append(
            {
                "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
                **info,
            }
        )
    return result


def store_case_projection(
    connection: sqlite3.Connection, dataset: str, release: str
) -> dict[str, Any]:
    projection = hashlib.sha256()
    payloads = hashlib.sha256()
    identities = hashlib.sha256()
    count = 0
    query = (
        "SELECT dataset_id,release_id,case_id,public_index,local_id,source_payload_json "
        "FROM cases WHERE dataset_id=? AND release_id=? ORDER BY public_index"
    )
    for row in connection.execute(query, (dataset, release)):
        update_framed_row(projection, tuple(row))
        payloads.update(row["source_payload_json"].encode("utf-8") + b"\n")
        identities.update(
            canonical_json(
                {
                    "case_id": row["case_id"],
                    "local_id": row["local_id"],
                    "public_index": row["public_index"],
                }
            )
            + b"\n"
        )
        count += 1
    return {
        "case_count": count,
        "case_identity_sha256": identities.hexdigest(),
        "columns": [
            "dataset_id",
            "release_id",
            "case_id",
            "public_index",
            "local_id",
            "source_payload_json",
        ],
        "order_by": ["public_index"],
        "payload_jsonl_sha256": payloads.hexdigest(),
        "sha256": projection.hexdigest(),
    }


def store_identifier_inventory(
    connection: sqlite3.Connection, dataset: str, release: str
) -> dict[str, Any]:
    digest = hashlib.sha256()
    namespaces: Counter[str] = Counter()
    count = 0
    query = (
        "SELECT i.case_id,i.namespace,i.value FROM identifiers i "
        "JOIN cases c ON c.dataset_id=i.dataset_id AND c.release_id=i.release_id "
        "AND c.case_id=i.case_id WHERE i.dataset_id=? AND i.release_id=? "
        "ORDER BY c.public_index,i.namespace,i.value"
    )
    for row in connection.execute(query, (dataset, release)):
        digest.update(canonical_json(dict(row)) + b"\n")
        namespaces[row["namespace"]] += 1
        count += 1
    return {
        "by_namespace": dict(sorted(namespaces.items())),
        "count": count,
        "sha256": digest.hexdigest(),
    }


def store_asset_reference_inventory(
    connection: sqlite3.Connection,
    dataset: str,
    release: str,
    kind: str,
) -> dict[str, Any]:
    if kind == "structure":
        query = (
            "SELECT r.case_id,r.structure_id,r.format,r.role,r.is_primary,"
            "r.content_sha256,t.size FROM structure_asset_refs r "
            "JOIN target_assets t USING(content_sha256) "
            "JOIN cases c ON c.dataset_id=r.dataset_id AND c.release_id=r.release_id "
            "AND c.case_id=r.case_id WHERE r.dataset_id=? AND r.release_id=? "
            "ORDER BY c.public_index,r.structure_id,r.format,r.role,r.content_sha256"
        )
    elif kind == "scientific":
        query = (
            "SELECT r.case_id,r.collection,r.declaration_id,r.asset_index,r.role,"
            "r.status,r.content_sha256,t.size FROM scientific_asset_refs r "
            "JOIN target_assets t USING(content_sha256) "
            "JOIN cases c ON c.dataset_id=r.dataset_id AND c.release_id=r.release_id "
            "AND c.case_id=r.case_id WHERE r.dataset_id=? AND r.release_id=? "
            "ORDER BY c.public_index,r.collection,r.declaration_id,r.asset_index"
        )
    elif kind == "control":
        query = (
            "SELECT r.category,r.name,r.content_sha256,t.size,t.relative_path "
            "FROM release_control_asset_refs r JOIN target_assets t USING(content_sha256) "
            "WHERE r.dataset_id=? AND r.release_id=? "
            "ORDER BY r.category,r.name"
        )
    else:
        raise ValueError(kind)
    digest = hashlib.sha256()
    count = 0
    unique: dict[str, int] = {}
    for row in connection.execute(query, (dataset, release)):
        record = dict(row)
        digest.update(canonical_json(record) + b"\n")
        unique[record["content_sha256"]] = record["size"]
        count += 1
    return {
        "count": count,
        "sha256": digest.hexdigest(),
        "unique_content": {
            "bytes": sum(unique.values()),
            "count": len(unique),
            "sha256": canonical_digest(
                [
                    {"content_sha256": item, "size": size}
                    for item, size in sorted(unique.items())
                ]
            ),
        },
    }


def receipt_state(
    connection: sqlite3.Connection, dataset: str, release: str
) -> dict[str, Any]:
    row = dict(
        connection.execute(
            "SELECT * FROM ingestion_receipts WHERE dataset_id=? AND release_id=?",
            (dataset, release),
        ).fetchone()
    )
    row["binding"] = json.loads(row.pop("binding_json"))
    selected = [
        item[0]
        for item in connection.execute(
            "SELECT c.case_id FROM cases c JOIN case_assessments a "
            "ON a.dataset_id=c.dataset_id AND a.release_id=c.release_id "
            "AND a.case_id=c.case_id WHERE c.dataset_id=? AND c.release_id=? "
            "AND (?=1 OR a.archive_class!='incomplete') ORDER BY c.case_id",
            (dataset, release, row["include_incomplete"]),
        )
    ]
    excluded = [
        item[0]
        for item in connection.execute(
            "SELECT c.case_id FROM cases c JOIN case_assessments a "
            "ON a.dataset_id=c.dataset_id AND a.release_id=c.release_id "
            "AND a.case_id=c.case_id WHERE c.dataset_id=? AND c.release_id=? "
            "AND ?=0 AND a.archive_class='incomplete' ORDER BY c.case_id",
            (dataset, release, row["include_incomplete"]),
        )
    ]
    recomputed = {
        "excluded_cases_digest": canonical_digest(excluded),
        "excluded_count": len(excluded),
        "included_cases_digest": canonical_digest(selected),
        "included_count": len(selected),
    }
    return {
        "database_row": row,
        "recomputed_selection": recomputed,
        "selection_binding_matches": all(
            row[key] == value for key, value in recomputed.items()
        ),
    }


def store_dataset_projection(
    connection: sqlite3.Connection, dataset: str, release: str
) -> dict[str, Any]:
    filters = {
        "datasets": ("dataset_id=?", (dataset,), ("created_at",)),
        "dataset_policies": ("dataset_id=?", (dataset,), ("updated_at",)),
        "releases": (
            "dataset_id=? AND release_id=?",
            (dataset, release),
            ("created_at",),
        ),
        "case_assessments": (
            "dataset_id=? AND release_id=?",
            (dataset, release),
            (),
        ),
        "ordinary_case_index": (
            "dataset_id=? AND release_id=?",
            (dataset, release),
            (),
        ),
        "benchmark_case_index": (
            "dataset_id=? AND release_id=?",
            (dataset, release),
            (),
        ),
        "incomplete_case_index": (
            "dataset_id=? AND release_id=?",
            (dataset, release),
            (),
        ),
    }
    return {
        table: table_projection(
            connection,
            table,
            where=where,
            parameters=parameters,
            excluded_columns=excluded,
        )
        for table, (where, parameters, excluded) in filters.items()
    }


def verify_store_assets(
    connection: sqlite3.Connection, asset_root: Path, file_hasher: FileHasher
) -> dict[str, Any]:
    records = []
    failures = []
    total_bytes = 0
    verified_count = 0
    for row in connection.execute(
        "SELECT content_sha256,size,relative_path FROM target_assets "
        "ORDER BY content_sha256"
    ):
        path = asset_root / row["relative_path"]
        record = dict(row)
        records.append(record)
        total_bytes += row["size"]
        if not path.is_file():
            failures.append({"content_sha256": row["content_sha256"], "error": "missing"})
            continue
        actual = file_hasher.hash(path)
        if actual != {"sha256": row["content_sha256"], "size": row["size"]}:
            failures.append(
                {
                    "content_sha256": row["content_sha256"],
                    "error": "content_mismatch",
                    "actual": actual,
                }
            )
            continue
        verified_count += 1
    projection = canonical_line_digest(records)
    expected_paths = {record["relative_path"] for record in records}
    filesystem_content_paths = {
        path.relative_to(asset_root).as_posix()
        for path in asset_root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    lock_file_count = sum(
        1
        for path in asset_root.rglob("*")
        if path.is_file() and path.name.startswith(".") and path.name.endswith(".lock")
    )
    unreferenced = sorted(filesystem_content_paths - expected_paths)
    return {
        "bytes": total_bytes,
        "content_verified_count": verified_count,
        "count": len(records),
        "failures": failures,
        "filesystem_content_file_count": len(filesystem_content_paths),
        "inventory_sha256": projection["sha256"],
        "lock_file_count": lock_file_count,
        "unreferenced_content_file_count": len(unreferenced),
        "unreferenced_content_paths_sha256": canonical_digest(unreferenced),
    }


def declared_control_assets(
    root: Path, manifest: Mapping[str, Any], file_hasher: FileHasher
) -> dict[str, Any]:
    records = []
    failures = []
    for descriptor in sorted(
        manifest.get("control_assets", []),
        key=lambda item: (item["category"], item["name"]),
    ):
        actual = file_hasher.hash(root / descriptor["path"])
        record = {
            "category": descriptor["category"],
            "name": descriptor["name"],
            "path": descriptor["path"],
            **actual,
        }
        records.append(record)
        if actual != {
            "sha256": descriptor["sha256"],
            "size": descriptor["size"],
        }:
            failures.append(record)
    inventory = canonical_line_digest(records)
    return {**inventory, "failures": failures}


def bundle_baseline(
    repository_root: Path, relative_root: str, file_hasher: FileHasher
) -> dict[str, Any]:
    root = repository_root / relative_root
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    generation = generation_root(root)
    generation_manifest = json.loads(
        (generation / "generation.json").read_text(encoding="utf-8")
    )
    summary = json.loads((generation / "summary.json").read_text(encoding="utf-8"))
    receipt_file = json.loads(
        (root / "ingestion-receipt.json").read_text(encoding="utf-8")
    )
    case_stream = scan_case_stream(root / "cases.jsonl", root, file_hasher)
    facts = facts_state(root / "cases" / "facts.sqlite3")
    assessment = assessment_state(generation / "assessment.sqlite3")
    result = {
        "bundle_root": relative_root,
        "identity": {
            "dataset_id": manifest["dataset_id"],
            "release_id": manifest["release_id"],
            "release_serial": manifest.get("release_serial"),
            "version": manifest["version"],
        },
        "schemas": {
            "bundle": manifest["schema"],
            "assessment_runtime": generation_manifest["runtime_policy"]["assessment"][
                "schema"
            ],
            "assessment_store": generation_manifest["runtime_policy"]["assessment"][
                "store_protocol"
            ]["schema"],
            "runtime_policy": generation_manifest["runtime_policy"]["schema"],
        },
        "digests": {
            "generation": generation_manifest["generation_digest"],
            "governance_registry": generation_manifest["governance"]["digest"],
            "ordered_assessment": generation_manifest["ordered_assessment_digest"],
            "policy": generation_manifest["policy"]["digest"],
            "runtime_policy": generation_manifest["runtime_policy"]["digest"],
            "source": facts["control"]["source_digest"],
        },
        "revisions": {
            "governance_registry": facts["control"]["governance_registry_revision"],
            "policy": facts["control"]["policy_revision"],
            "source": facts["control"]["source_revision"],
        },
        "case_stream": case_stream,
        "facts": facts,
        "assessment": assessment,
        "generation_summary": summary,
        "declared_control_assets": declared_control_assets(root, manifest, file_hasher),
        "receipt_file": receipt_file,
        "input_files": input_file_inventory(
            repository_root, root, generation, manifest, file_hasher
        ),
    }
    result["internal_parity"] = {
        "assessment_count_matches_cases": assessment["projections"][
            "case_assessments"
        ]["count"]
        == case_stream["case_count"],
        "facts_count_matches_cases": facts["case_count"] == case_stream["case_count"],
        "facts_payload_matches_raw": facts["payload_jsonl_sha256"]
        == case_stream["raw_file"]["sha256"],
        "raw_stream_is_canonical_jsonl": case_stream["raw_file"]["sha256"]
        == case_stream["logical_canonical_jsonl_sha256"],
        "summary_case_count_matches": summary["cases"] == case_stream["case_count"],
    }
    return result


def store_baseline(
    repository_root: Path,
    bundles: Sequence[Mapping[str, Any]],
    file_hasher: FileHasher,
) -> dict[str, Any]:
    store_path = (repository_root / STORE_RELATIVE_PATH).resolve()
    asset_root = Path(str(store_path) + ".assets")
    with readonly_sqlite(store_path) as connection:
        metadata = dict(connection.execute("SELECT * FROM store_metadata").fetchone())
        datasets = {}
        for bundle in bundles:
            identity = bundle["identity"]
            dataset = identity["dataset_id"]
            release = identity["release_id"]
            cases = store_case_projection(connection, dataset, release)
            identifiers = store_identifier_inventory(connection, dataset, release)
            assets = {
                kind: store_asset_reference_inventory(
                    connection, dataset, release, kind
                )
                for kind in ("structure", "scientific", "control")
            }
            datasets[dataset] = {
                "identity": identity,
                "cases": cases,
                "identifiers": identifiers,
                "asset_references": assets,
                "projections": store_dataset_projection(
                    connection, dataset, release
                ),
                "receipt": receipt_state(connection, dataset, release),
                "bundle_parity": {
                    "case_count": cases["case_count"]
                    == bundle["case_stream"]["case_count"],
                    "case_payload": cases["payload_jsonl_sha256"]
                    == bundle["case_stream"]["logical_canonical_jsonl_sha256"],
                    "identifiers": identifiers["sha256"]
                    == bundle["case_stream"]["identifiers"]["sha256"],
                    "scientific_asset_references": assets["scientific"]["sha256"]
                    == bundle["case_stream"]["materialized_assets"][
                        "scientific_references"
                    ]["sha256"],
                    "structure_asset_references": assets["structure"]["sha256"]
                    == bundle["case_stream"]["materialized_assets"][
                        "structure_references"
                    ]["sha256"],
                },
            }
        target_assets = verify_store_assets(connection, asset_root, file_hasher)
        global_projections = {
            "store_metadata": table_projection(connection, "store_metadata"),
            "target_assets": table_projection(connection, "target_assets"),
        }
        return {
            "active_database": store_path.relative_to(repository_root).as_posix(),
            "active_generation": store_path.parent.relative_to(repository_root).as_posix(),
            "asset_root": asset_root.relative_to(repository_root).as_posix(),
            "database_file": file_hasher.hash(store_path),
            "datasets": datasets,
            "global_projections": global_projections,
            "metadata": metadata,
            "sqlite": sqlite_identity(connection),
            "target_assets": target_assets,
        }


def git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def generate(repository_root: Path) -> dict[str, Any]:
    file_hasher = FileHasher()
    bundles = [
        bundle_baseline(repository_root, relative_root, file_hasher)
        for relative_root in BUNDLE_RELATIVE_ROOTS
    ]
    store = store_baseline(repository_root, bundles, file_hasher)
    all_checks = []
    for bundle in bundles:
        all_checks.extend(bundle["internal_parity"].values())
        all_checks.append(not bundle["declared_control_assets"]["failures"])
    for dataset in store["datasets"].values():
        all_checks.extend(dataset["bundle_parity"].values())
        all_checks.append(dataset["receipt"]["selection_binding_matches"])
    all_checks.append(not store["target_assets"]["failures"])
    return {
        "schema": SCHEMA,
        "purpose": (
            "Read-only pre-Wave-B semantic and storage parity baseline for clean "
            "Bundle and TargetStore rebuild comparison."
        ),
        "repository_commit_at_capture": git_head(repository_root),
        "algorithms": {
            "canonical_json": (
                "UTF-8 JSON with sorted keys, compact separators, ensure_ascii=false, "
                "allow_nan=false"
            ),
            "canonical_json_lines_sha256": (
                "SHA-256 over each canonical JSON record followed by one LF, in the "
                "documented query or Case public_index order"
            ),
            "file_sha256": "SHA-256 over exact file bytes",
            "sqlite_projection_sha256": (
                "sha256-framed-sql-values-v1: rows in documented order; R prefix, "
                "N/B/I/F/T value tag, decimal byte length, colon, payload, semicolon "
                "row terminator"
            ),
        },
        "capture_constraints": {
            "hotpot_runtime_imported": False,
            "sqlite_open_mode": "mode=ro&immutable=1",
            "writes_to_bundle_or_store": False,
        },
        "bundles": {bundle["identity"]["dataset_id"]: bundle for bundle in bundles},
        "active_store": store,
        "validation": {
            "all_parity_checks_passed": all(all_checks),
            "check_count": len(all_checks),
            "failed_check_count": sum(not check for check in all_checks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = json.dumps(generate(arguments.repository_root), indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
