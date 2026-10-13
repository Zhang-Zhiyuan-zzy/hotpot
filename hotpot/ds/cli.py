"""Command-line access to the concise Hotpot dataset API."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

from ._communication import emit_envelope, read_input
from ._export import export_records, render_structure_record
from .database import DatasetStore
from .levels import EvidenceLevel, QualityLevel
from .quality import QualityGateError
from .public.catalog import DEFAULT_DATABASE, DatasetCatalog, load
from .public.bundle import load_bundle
from .public.certificate import Certificate
from .public.search import DataSearcher, DatasetSearcher, SmartsSearcher
from .public.store import Store, store as public_store


HELP_FORMATTER = argparse.ArgumentDefaultsHelpFormatter
_AUTO_IDENTIFIER = object()


def _command(subparsers, name: str, *, help: str, description: str):
    return subparsers.add_parser(
        name,
        help=help,
        description=description,
        formatter_class=HELP_FORMATTER,
    )


def _database(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="Hotpot SQLite dataset-store path; defaults to the project store",
    )


def _dataset(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    options = {} if required else {"nargs": "?"}
    parser.add_argument("dataset", metavar="DATASET", help="public dataset name", **options)
    parser.add_argument(
        "--version",
        default=argparse.SUPPRESS,
        metavar="VERSION",
        help="source version, release ID, or three-digit public release number",
    )
    _database(parser)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Discover, query, search, export, certify, and ingest governed scientific datasets."
    )
    parser.formatter_class = HELP_FORMATTER
    actions = parser.add_subparsers(
        title="commands", dest="ds_action", metavar="COMMAND", required=True
    )

    listing = _command(
        actions,
        "list",
        help="list available datasets",
        description="List the public names of all datasets in one store.",
    )
    _database(listing)

    info = _command(
        actions,
        "info",
        help="show dataset information",
        description="Show public metadata, labels, handles, and counts for one dataset release.",
    )
    _dataset(info)

    get = _command(
        actions,
        "get",
        help="get one data item",
        description="Get one data item by sequence index, public ID, or dataset identifier.",
    )
    _dataset(get)
    get.add_argument(
        "selector",
        metavar="SELECTOR",
        help="sequence index by default, an ID selected by a mode flag, or - for protocol stdin",
    )
    selection = get.add_mutually_exclusive_group()
    selection.add_argument(
        "--index",
        action="store_true",
        help="interpret SELECTOR as a zero-based sequence index (the default)",
    )
    selection.add_argument(
        "--id",
        action="store_true",
        help="interpret SELECTOR as a stable Hotpot data ID",
    )
    selection.add_argument(
        "--identifier",
        nargs="?",
        const=_AUTO_IDENTIFIER,
        metavar="NAMESPACE",
        help="interpret SELECTOR as an external identifier; infer its namespace when unique",
    )
    display = get.add_mutually_exclusive_group()
    display.add_argument(
        "--smi",
        action="store_true",
        help="include SMILES in the JSON result and mirror it on an interactive terminal",
    )
    display.add_argument(
        "--xyz",
        action="store_true",
        help="include XYZ in the JSON result and mirror it on an interactive terminal",
    )
    structure_selection = get.add_mutually_exclusive_group()
    structure_selection.add_argument(
        "--structure-index",
        type=int,
        metavar="N",
        help="select the attached structure at zero-based position N in data.info order",
    )
    structure_selection.add_argument(
        "--structure-role",
        metavar="ROLE",
        help="select the unique attached structure carrying ROLE",
    )
    structure_selection.add_argument(
        "--structure-id",
        metavar="ID",
        help="select an attached structure by its exact stored structure ID",
    )

    labels = _command(
        actions,
        "labels",
        help="show label coverage",
        description="Show each structured label and the number of data items containing it.",
    )
    _dataset(labels)

    subset = _command(
        actions,
        "subset",
        help="filter data fields",
        description="Return a Dataset subset; repeated field expressions are combined with AND.",
    )
    _dataset(subset)
    subset.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="field equality or IN filter; repeat for multiple fields",
    )
    subset.add_argument(
        "-q",
        "--quality",
        type=int,
        choices=range(5),
        default=0,
        metavar="LEVEL",
        help="minimum data quality level, from 0 (B0) through 4 (B4)",
    )
    subset.add_argument(
        "-e",
        "--evidence",
        type=int,
        choices=range(5),
        default=0,
        metavar="LEVEL",
        help="minimum data evidence level, from 0 (E0) through 4 (E4)",
    )
    subset.add_argument("--limit", type=int, metavar="N", help="emit at most N data summaries")

    search = _command(
        actions,
        "search",
        help="search datasets, data fields, or molecular substructures",
        description="Search dataset metadata, data.info fields, or structures with SMARTS.",
    )
    _dataset(search, required=False)
    search.add_argument(
        "--scope",
        choices=("data", "dataset", "smarts"),
        default="data",
        help="kind of object to search",
    )
    search.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="field equality or IN filter; repeat for multiple fields",
    )
    search.add_argument("--smarts", metavar="PATTERN", help="SMARTS pattern for smarts scope")
    search.add_argument("--limit", type=int, metavar="N", help="emit at most N matches")

    export = _command(
        actions,
        "export",
        help="export structures and record information",
        description=(
            "Atomically export one record, a range, a field-filtered subset, or an entire "
            "dataset. Explicit formats are converted with Hotpot Molecule."
        ),
    )
    _dataset(export)
    export.add_argument("output", metavar="OUTPUT", help="new output directory")
    export_scope = export.add_mutually_exclusive_group()
    export_scope.add_argument(
        "item",
        nargs="?",
        metavar="ITEM",
        help="one sequence index or identifier, - for protocol stdin, or omit for several records",
    )
    export_scope.add_argument(
        "--range",
        nargs=2,
        type=int,
        dest="index_range",
        metavar=("START", "STOP"),
        help="export the zero-based half-open sequence range [START, STOP)",
    )
    export_scope.add_argument(
        "--where",
        action="append",
        metavar="FIELD=JSON",
        help="export a field-equality or IN subset; repeat to combine fields with AND",
    )
    export_selection = export.add_mutually_exclusive_group()
    export_selection.add_argument(
        "--index",
        action="store_true",
        help="interpret ITEM as a zero-based sequence index (the default)",
    )
    export_selection.add_argument(
        "--id",
        action="store_true",
        help="interpret ITEM as a stable Hotpot data ID",
    )
    export_selection.add_argument(
        "--identifier",
        nargs="?",
        const=_AUTO_IDENTIFIER,
        metavar="NAMESPACE",
        help="interpret ITEM as an external identifier; infer its namespace when unique",
    )
    export.add_argument(
        "--format",
        metavar="FORMAT",
        help="convert every structure with Hotpot Molecule; otherwise copy its native format",
    )

    certify = _command(
        actions,
        "certify",
        help="assess a submission and write its certificate",
        description=(
            "Assess a folder, compact SQLite database, or bundle. New datasets require an "
            "explicit quality level, core labels, or core information declaration."
        ),
    )
    certify.add_argument("source", metavar="SOURCE", help="submission folder, SQLite, or JSON")
    certify.add_argument("dataset", metavar="DATASET", help="target dataset name")
    certify.add_argument(
        "--existing",
        action="store_true",
        help="assess as a new immutable release of an established dataset",
    )
    certify.add_argument(
        "--quality-level",
        choices=tuple(level.value for level in QualityLevel),
        help="declared quality level for a new dataset",
    )
    certify.add_argument(
        "--core-label",
        action="append",
        dest="core_labels",
        metavar="NAME",
        help="label required on every data item; repeat as needed",
    )
    certify.add_argument(
        "--core-info",
        action="append",
        dest="core_info",
        metavar="PATH",
        help="information path required on every data item; repeat as needed",
    )
    certify.add_argument(
        "--no-core-fields",
        action="store_true",
        help="explicitly declare empty core_labels and core_info",
    )
    certify.add_argument(
        "--configure",
        metavar="PATH",
        help="policy configuration JSON for a new dataset",
    )
    certify.add_argument(
        "--certificate",
        required=True,
        metavar="PATH",
        help="write the resulting assessment certificate here",
    )
    certify.add_argument("--asset-root", metavar="DIR", help="override the structure-file root")
    _database(certify)

    ingest = _command(
        actions,
        "ingest",
        help="ingest a certified submission",
        description="Verify a public certificate and atomically ingest its exact submission.",
    )
    ingest.add_argument("source", metavar="SOURCE", help="the exact assessed submission")
    ingest.add_argument(
        "--certificate",
        required=True,
        metavar="PATH",
        help="passing public certificate issued for SOURCE",
    )
    ingest.add_argument("--asset-root", metavar="DIR", help="override the structure-file root")
    ingest.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="target Hotpot SQLite dataset-store path",
    )

    detail = _command(
        actions,
        "detail",
        help="advanced storage diagnostics",
        description="Advanced integrity audit and normalized-store diagnostics.",
    )
    detail_actions = detail.add_subparsers(
        title="detail commands",
        dest="detail_action",
        metavar="COMMAND",
        required=False,
    )
    for name, help_text, description in (
        ("status", "show normalized table counts", "Show counts from the normalized backend."),
        ("audit", "audit relations and assets", "Audit foreign keys, certificates, and assets."),
    ):
        child = _command(
            detail_actions,
            name,
            help=help_text,
            description=description,
        )
        _database(child)
    raw_query = _command(
        detail_actions,
        "query",
        help="query normalized backend rows",
        description="Advanced queries over normalized observation and claim rows.",
    )
    raw_targets = raw_query.add_subparsers(
        title="query targets",
        dest="detail_query_target",
        metavar="TARGET",
        required=True,
    )
    raw_claims = _command(
        raw_targets,
        "claims",
        help="query unresolved claims",
        description="Query unresolved or extracted claims from the normalized backend.",
    )
    _database(raw_claims)
    raw_claims.add_argument(
        "--dataset", metavar="NAME", help="filter by backend dataset identifier"
    )
    raw_claims.add_argument(
        "--evidence-level", metavar="LEVEL", help="filter by evidence level"
    )
    raw_claims.add_argument("--limit", type=int, metavar="N", help="return at most N rows")
    raw_claims.add_argument(
        "--format",
        choices=("json",),
        default="json",
        help="output encoding; protocol stdout is one JSON document",
    )


def _db_argument(args: argparse.Namespace):
    return getattr(args, "db", None)


def _version_argument(args: argparse.Namespace):
    return getattr(args, "version", None)


def _active_database(args: argparse.Namespace):
    return _db_argument(args) or DEFAULT_DATABASE


def _resolved_path(value: Any) -> str:
    return str(Path(value).expanduser().resolve())


def _dataset_locator(dataset) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "release_id": dataset.release_id,
        "version": dataset.version,
        "version_serial": f"{dataset.version_serial:03d}",
        "database": _resolved_path(dataset._store.path),
    }


def _emit(key: str, payload: Any, *, channel: str = "default") -> None:
    emit_envelope(key, payload, channel=channel)


def _criteria(expressions: Iterable[str]) -> dict[str, Any]:
    result = {}
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(f"filter must be FIELD=JSON: {expression!r}")
        field, raw = expression.split("=", 1)
        if not field:
            raise ValueError("filter field cannot be empty")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        result[field] = value
    return result


def _summaries(values, limit=None):
    selected = values if limit is None else values[:limit]
    return [
        {
            "id": value.id,
            "local_id": value.local_id,
            "labels": value.label,
            "quality_level": value.quality_level,
            "evidence_level": value.evidence_level,
        }
        for value in selected
    ]


def _identifier_namespace(dataset, requested) -> str:
    namespaces = dataset.identifier_namespaces
    choices = ", ".join(namespaces) or "none"
    if requested is _AUTO_IDENTIFIER:
        if len(namespaces) != 1:
            raise ValueError(
                "bare --identifier requires exactly one identifier namespace; "
                f"available: {choices}"
            )
        return namespaces[0]

    name = str(requested).strip()
    lowered = name.lower()
    exact = [namespace for namespace in namespaces if namespace.lower() == lowered]
    if exact:
        return exact[0]
    prefix = (
        name[: -len("_identifier")]
        if lowered.endswith("_identifier")
        else name
    ).lower()
    matches = [
        namespace
        for namespace in namespaces
        if namespace.lower().startswith(prefix + "_")
        or namespace.split("_", 1)[0].lower() == prefix
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"identifier namespace {name!r} is ambiguous; available: {choices}"
        )
    raise ValueError(f"unknown identifier namespace {name!r}; available: {choices}")


def _select_data(dataset, selector: str, *, identifier=None, by_id: bool = False):
    if identifier is not None:
        namespace = _identifier_namespace(dataset, identifier)
        matches = dataset.subset(**{namespace: selector})
        if not matches:
            raise ValueError(
                f"external identifier {selector!r} was not found in {namespace!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"external identifier {selector!r} in {namespace!r} matches "
                f"{len(matches)} records"
            )
        return matches[0], {
            "kind": "identifier",
            "namespace": namespace,
            "value": selector,
        }

    if by_id:
        data = next((item for item in dataset if item.id == selector), None)
        if data is None:
            raise ValueError(f"Hotpot data ID {selector!r} was not found")
        return data, {"kind": "id", "value": selector}

    try:
        index = int(selector)
    except ValueError as error:
        raise ValueError(
            f"index selector {selector!r} is not an integer; use --identifier or --id"
        ) from error
    try:
        data = dataset[index]
    except IndexError as error:
        raise ValueError(f"sequence index {index} is out of range") from error
    return data, {"kind": "index", "index": index}


def _present_representations(representations: list[dict[str, Any]]) -> None:
    """Mirror requested structures to the interactive presentation stream."""
    try:
        if not representations or not sys.stderr.isatty():
            return
        multiple = len(representations) > 1
        for representation in representations:
            if multiple:
                sys.stderr.write(
                    "--- "
                    f"{representation['data_id']} "
                    f"{representation['structure_id']} "
                    f"[{representation['format']}] ---\n"
                )
            content = representation["content"]
            sys.stderr.write(content)
            if not content.endswith("\n"):
                sys.stderr.write("\n")
    except Exception:
        return


def _load_direct_dataset(args: argparse.Namespace):
    return load(
        args.dataset,
        version=_version_argument(args),
        db=_db_argument(args),
    )


def _load_protocol_selection(args: argparse.Namespace, consumer_key: str):
    resolved = read_input(consumer_key, sys.stdin)
    bindings = resolved["bindings"]
    locator = bindings["dataset"]
    if locator["name"] != args.dataset:
        raise ValueError(
            f"stdin dataset {locator['name']!r} does not match "
            f"positional DATASET {args.dataset!r}"
        )

    upstream_database = _resolved_path(locator["database"])
    explicit_database = _db_argument(args)
    if explicit_database is not None and _resolved_path(explicit_database) != upstream_database:
        raise ValueError(
            "--db does not match the database carried by protocol stdin"
        )
    database = explicit_database or upstream_database
    version = _version_argument(args) or locator["release_id"]
    dataset = load(args.dataset, version=version, db=database)
    normalized_locator = {**locator, "database": upstream_database}
    expected_locator = _dataset_locator(dataset)
    mismatched = [
        name
        for name, value in expected_locator.items()
        if normalized_locator[name] != value
    ]
    if mismatched:
        raise ValueError(
            "selected dataset does not match protocol stdin fields: "
            + ", ".join(mismatched)
        )

    by_id = {data.id: data for data in dataset}
    records = []
    for selector in bindings["selectors"]:
        data = by_id.get(selector)
        if data is None:
            raise ValueError(f"Hotpot data ID {selector!r} was not found")
        records.append(data)
    return dataset, records


def _export_selection(dataset, args: argparse.Namespace):
    has_many_selector = args.index_range is not None or args.where is not None
    if args.item is not None:
        if has_many_selector:
            raise ValueError("ITEM cannot be combined with --range or --where")
        data, selection = _select_data(
            dataset,
            args.item,
            identifier=args.identifier,
            by_id=args.id,
        )
        return dataset._select([data]), selection

    if args.index or args.id or args.identifier is not None:
        raise ValueError("ITEM is required with --index, --id, or --identifier")
    if args.index_range is not None:
        start, stop = args.index_range
        if start < 0 or stop < start or stop > len(dataset):
            raise ValueError(
                f"range must satisfy 0 <= START <= STOP <= {len(dataset)}"
            )
        selected = dataset[start:stop]
        selection = {"kind": "range", "start": start, "stop": stop}
    elif args.where is not None:
        criteria = _criteria(args.where)
        selected = dataset.subset(**criteria)
        selection = {"kind": "subset", "criteria": criteria}
    else:
        selected = dataset
        selection = {"kind": "all"}
    if not selected:
        raise ValueError("the export selection is empty")
    return selected, selection


def run(args: argparse.Namespace) -> int:
    try:
        if args.ds_action == "list":
            database = _active_database(args)
            _emit(
                "hotpot.ds.list",
                {
                    "database": _resolved_path(database),
                    "datasets": list(DatasetCatalog(database)),
                },
            )
            return 0

        if args.ds_action in {"info", "get", "labels", "subset"}:
            if args.ds_action == "info":
                dataset = _load_direct_dataset(args)
                _emit(
                    "hotpot.ds.info",
                    {"dataset": _dataset_locator(dataset), "info": dataset.info},
                )
            elif args.ds_action == "labels":
                dataset = _load_direct_dataset(args)
                _emit(
                    "hotpot.ds.labels",
                    {
                        "dataset": _dataset_locator(dataset),
                        "core_labels": dataset.core_labels,
                        "labels": dataset.labels,
                    }
                )
            elif args.ds_action == "subset":
                dataset = _load_direct_dataset(args)
                result = dataset.subset(**_criteria(args.where))
                result = result.subset(
                    quality_level=tuple(
                        level.value for level in tuple(QualityLevel)[args.quality :]
                    ),
                    evidence_level=tuple(
                        level.value for level in tuple(EvidenceLevel)[args.evidence :]
                    ),
                )
                items = _summaries(result, args.limit)
                _emit(
                    "hotpot.ds.subset",
                    {
                        "dataset": _dataset_locator(dataset),
                        "items": items,
                        "count": len(items),
                    },
                )
            else:
                if args.selector == "-":
                    if args.index or args.id or args.identifier is not None:
                        raise ValueError(
                            "SELECTOR '-' cannot be combined with --index, --id, "
                            "or --identifier"
                        )
                    dataset, records = _load_protocol_selection(args, "hotpot.ds.get")
                else:
                    dataset = _load_direct_dataset(args)
                    data, _ = _select_data(
                        dataset,
                        args.selector,
                        identifier=args.identifier,
                        by_id=args.id,
                    )
                    records = [data]
                structure_selected = any(
                    value is not None
                    for value in (
                        args.structure_index,
                        args.structure_role,
                        args.structure_id,
                    )
                )
                if structure_selected and not (args.smi or args.xyz):
                    raise ValueError(
                        "--structure-index, --structure-role, and --structure-id "
                        "require --smi or --xyz"
                    )
                representations = []
                if args.smi:
                    representations = [
                        render_structure_record(
                            data,
                            "smi",
                            structure_index=args.structure_index,
                            structure_role=args.structure_role,
                            structure_id=args.structure_id,
                        )
                        for data in records
                    ]
                elif args.xyz:
                    representations = [
                        render_structure_record(
                            data,
                            "xyz",
                            structure_index=args.structure_index,
                            structure_role=args.structure_role,
                            structure_id=args.structure_id,
                        )
                        for data in records
                    ]
                _emit(
                    "hotpot.ds.get",
                    {
                        "dataset": _dataset_locator(dataset),
                        "items": [data.info for data in records],
                        "representations": representations,
                        "count": len(records),
                    },
                    channel="structure" if args.smi or args.xyz else "default",
                )
                _present_representations(representations)
            return 0

        if args.ds_action == "export":
            if args.item == "-":
                if args.index or args.id or args.identifier is not None:
                    raise ValueError(
                        "ITEM '-' cannot be combined with --index, --id, or --identifier"
                    )
                dataset, records = _load_protocol_selection(args, "hotpot.ds.export")
                selected = dataset._select(records)
                selection = {
                    "kind": "protocol",
                    "ids": [data.id for data in records],
                }
            else:
                dataset = _load_direct_dataset(args)
                selected, selection = _export_selection(dataset, args)
            summary = export_records(
                selected,
                Path(args.output),
                fmt=args.format,
                selection=selection,
            )
            _emit(
                "hotpot.ds.export",
                {
                    "output": str(Path(args.output).expanduser().resolve()),
                    "dataset": _dataset_locator(selected),
                    "record_count": summary["record_count"],
                    "structure_count": summary["structure_count"],
                    "requested_format": summary["requested_format"],
                }
            )
            return 0

        if args.ds_action == "search":
            criteria = _criteria(args.where)
            if args.scope == "dataset":
                values = DatasetSearcher(**criteria).search(
                    DatasetCatalog(_active_database(args))
                )
                items = [value.info for value in values[: args.limit]]
                _emit(
                    "hotpot.ds.search",
                    {"items": items, "count": len(items)},
                    channel="dataset",
                )
                return 0
            if not args.dataset:
                raise ValueError("DATASET is required for data and SMARTS searches")
            dataset = _load_direct_dataset(args)
            if args.scope == "smarts":
                if not args.smarts:
                    raise ValueError("--smarts is required for SMARTS search")
                result = SmartsSearcher(args.smarts).search(dataset)
            else:
                result = DataSearcher(**criteria).search(dataset)
            items = _summaries(result, args.limit)
            _emit(
                "hotpot.ds.search",
                {
                    "dataset": _dataset_locator(dataset),
                    "items": items,
                    "count": len(items),
                },
                channel=args.scope,
            )
            return 0

        if args.ds_action == "certify":
            if args.no_core_fields and (
                args.core_labels is not None or args.core_info is not None
            ):
                raise ValueError(
                    "--no-core-fields cannot be combined with --core-label or --core-info"
                )
            if args.existing:
                incompatible = [
                    option
                    for option, supplied in (
                        ("--quality-level", args.quality_level is not None),
                        ("--core-label", args.core_labels is not None),
                        ("--core-info", args.core_info is not None),
                        ("--no-core-fields", args.no_core_fields),
                        ("--configure", args.configure is not None),
                    )
                    if supplied
                ]
                if incompatible:
                    raise ValueError(
                        "--existing loads the established dataset policy and cannot be "
                        "combined with " + ", ".join(incompatible)
                    )
            configure = (
                Certificate.Configure.from_file(args.configure) if args.configure else None
            )
            if args.existing:
                certificate = Certificate.assess_existing(
                    args.source,
                    ds_name=args.dataset,
                    store=Store(_active_database(args)),
                    asset_root=args.asset_root,
                )
            else:
                declarations = {}
                if args.quality_level is not None:
                    declarations["quality_level"] = args.quality_level
                if args.no_core_fields:
                    declarations["core_labels"] = []
                    declarations["core_info"] = []
                else:
                    if args.core_labels is not None:
                        declarations["core_labels"] = args.core_labels
                    if args.core_info is not None:
                        declarations["core_info"] = args.core_info
                certificate = Certificate.assess_new(
                    args.source,
                    ds_name=args.dataset,
                    configure=configure,
                    asset_root=args.asset_root,
                    **declarations,
                )
            certificate.write(args.certificate)
            _emit(
                "hotpot.ds.certify",
                {
                    "certificate": certificate.to_dict(),
                    "certificate_path": _resolved_path(args.certificate),
                    "source": _resolved_path(args.source),
                },
            )
            return 0 if certificate.passed else 2

        if args.ds_action == "ingest":
            bundle = load_bundle(args.source, asset_root=args.asset_root)
            result = public_store.ingest(
                bundle,
                certificate=args.certificate,
                db=args.db,
            )
            _emit(
                "hotpot.ds.ingest",
                {
                    "database": _resolved_path(args.db),
                    "result": result,
                },
            )
            return 0

        if args.ds_action == "detail":
            if args.detail_action is None:
                _emit(
                    "hotpot.ds.detail",
                    {"commands": ["status", "audit"]},
                )
                return 0
            backend = DatasetStore(_active_database(args))
            if args.detail_action == "query":
                rows = list(
                    backend.iter_claims(
                        dataset_id=args.dataset,
                        evidence_level=args.evidence_level,
                        limit=args.limit,
                    )
                )
                _emit("hotpot.ds.detail.query.claims", rows)
                return 0
            if args.detail_action == "status":
                _emit(
                    "hotpot.ds.detail.status",
                    {"database": str(backend.path), "counts": backend.counts()},
                )
                return 0
            issues = backend.integrity_issues()
            _emit(
                "hotpot.ds.detail.audit",
                {
                    "database": str(backend.path),
                    "passed": not issues,
                    "issues": issues,
                    "counts": backend.counts(),
                }
            )
            return 0 if not issues else 2
    except (
        KeyError,
        ValueError,
        OSError,
        sqlite3.DatabaseError,
        QualityGateError,
    ) as error:
        print(f"hotpot ds: {error}", file=sys.stderr)
        return 2
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HELP_FORMATTER)
    add_arguments(parser)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
