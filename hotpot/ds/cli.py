"""Command-line access to current Hotpot datasets and governed Bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ._communication import emit_envelope, read_input
from ._export import export_records, render_structure_record
from ._paths import DEFAULT_STORE_PATH
from .gates import BenchmarkGate, EvidenceGate, QualityGate
from .gates._checker import CHECKER_PROTOCOL_DIGEST, _CheckerDefinition, _inspect_source
from .public.bundle import Bundle, load_bundle
from .public.catalog import DatasetCatalog, load
from .public.certificate import certificate_digest
from .public.search import DataSearcher, DatasetSearcher, SmartsSearcher
from .storage import TargetStore


HELP_FORMATTER = argparse.ArgumentDefaultsHelpFormatter
_AUTO_IDENTIFIER = object()
_B_LEVELS = tuple(f"B{index}" for index in range(5))
_E_LEVELS = tuple(f"E{index}" for index in range(5))


def _command(subparsers, name: str, *, help: str, description: str):
    return subparsers.add_parser(
        name,
        help=help,
        description=description,
        formatter_class=HELP_FORMATTER,
    )


def _database(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--db",
        required=required,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="current Hotpot dataset store",
    )


def _dataset(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "dataset",
        nargs=None if required else "?",
        metavar="DATASET",
        help="public dataset name",
    )
    parser.add_argument(
        "--version",
        default=argparse.SUPPRESS,
        metavar="VERSION",
        help="release ID, semantic version, or three-digit release serial",
    )
    parser.add_argument(
        "--archive-class",
        action="append",
        choices=("ordinary", "benchmark", "incomplete"),
        default=argparse.SUPPRESS,
        help="opt into explicit persisted archive classes; repeat to combine",
    )
    _database(parser)


def _bundle_locator_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        dest="bundle_root_option",
        default=argparse.SUPPRESS,
        metavar="BUNDLE_ROOT",
        help="built Bundle root; defaults to the current directory",
    )
    parser.add_argument(
        "--dataset",
        dest="bundle_dataset",
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="assert the authoritative bundle.json dataset name",
    )
    parser.add_argument(
        "--db",
        dest="bundle_db",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="current Hotpot dataset store",
    )


def _selection_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--archive-class",
        action="append",
        choices=("incomplete", "ordinary", "benchmark"),
        help="select archive classes; repeat for IN semantics",
    )
    parser.add_argument("--quality", action="append", choices=_B_LEVELS, help="select exact B levels")
    parser.add_argument("--min-quality", choices=_B_LEVELS, help="select a minimum B level")
    parser.add_argument("--evidence", action="append", choices=_E_LEVELS, help="select exact E levels")
    parser.add_argument("--min-evidence", choices=_E_LEVELS, help="select a minimum E level")
    parser.add_argument("--benchmark-status", action="append", help="select benchmark statuses")
    parser.add_argument("--reason", action="append", help="select assessment reason codes")
    parser.add_argument("--verification", action="append", choices=("AI", "Human"), help="select evidence verification methods")
    parser.add_argument("--attestation", action="append", choices=("Hotpot", "User", "Other"), help="select attestation authorities")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = "Discover, query, assess, certify, and ingest current datasets."
    parser.formatter_class = HELP_FORMATTER
    actions = parser.add_subparsers(dest="ds_action", metavar="COMMAND", required=True)

    listing = _command(actions, "list", help="list datasets", description="List current datasets.")
    _database(listing)

    info = _command(actions, "info", help="show dataset information", description="Show one dataset release.")
    _dataset(info)

    labels = _command(actions, "labels", help="show label coverage", description="Show structured label counts.")
    _dataset(labels)

    get = _command(actions, "get", help="get data", description="Get data by index, Hotpot ID, or identifier.")
    _dataset(get)
    get.add_argument("selector", metavar="SELECTOR", help="index, identifier, Hotpot ID, or - for protocol stdin")
    modes = get.add_mutually_exclusive_group()
    modes.add_argument("--index", action="store_true", help="interpret SELECTOR as a sequence index (default)")
    modes.add_argument("--id", action="store_true", help="interpret SELECTOR as a Hotpot data ID")
    modes.add_argument("--identifier", nargs="?", const=_AUTO_IDENTIFIER, metavar="NAMESPACE", help="interpret SELECTOR as an external identifier")
    presentation = get.add_mutually_exclusive_group()
    presentation.add_argument("--smi", action="store_true", help="render SMILES")
    presentation.add_argument("--xyz", action="store_true", help="render XYZ")
    structures = get.add_mutually_exclusive_group()
    structures.add_argument("--structure-index", type=int, metavar="N", help="select a zero-based attached structure")
    structures.add_argument("--structure-role", metavar="ROLE", help="select the unique structure with ROLE")
    structures.add_argument("--structure-id", metavar="ID", help="select an exact attached structure ID")

    subset = _command(actions, "subset", help="filter data", description="Select data by facts and quality thresholds.")
    _dataset(subset)
    subset.add_argument("--where", action="append", default=[], metavar="FIELD=JSON", help="field equality or IN filter; repeat to combine")
    subset.add_argument("-q", "--quality", type=int, choices=range(5), default=0, metavar="LEVEL", help="minimum B level; zero disables the threshold")
    subset.add_argument("-e", "--evidence", type=int, choices=range(5), default=0, metavar="LEVEL", help="minimum E level; zero preserves not-applicable records")
    subset.add_argument("--limit", type=int, metavar="N", help="emit at most N records")

    search = _command(actions, "search", help="search data", description="Search dataset metadata, facts, or SMARTS.")
    _dataset(search, required=False)
    search.add_argument("--scope", choices=("data", "dataset", "smarts"), default="data", help="object class to search")
    search.add_argument("--where", action="append", default=[], metavar="FIELD=JSON", help="field equality or IN filter")
    search.add_argument("--smarts", metavar="PATTERN", help="SMARTS query for smarts scope")
    search.add_argument("--limit", type=int, metavar="N", help="emit at most N matches")

    export = _command(actions, "export", help="export structures", description="Export one selection atomically.")
    _dataset(export)
    export.add_argument("output", metavar="OUTPUT", help="new export directory")
    scope = export.add_mutually_exclusive_group()
    scope.add_argument("item", nargs="?", metavar="ITEM", help="one index/identifier or - for protocol stdin")
    scope.add_argument("--range", nargs=2, type=int, dest="index_range", metavar=("START", "STOP"), help="half-open sequence range")
    scope.add_argument("--where", action="append", metavar="FIELD=JSON", help="field subset filter")
    modes = export.add_mutually_exclusive_group()
    modes.add_argument("--index", action="store_true", help="interpret ITEM as an index (default)")
    modes.add_argument("--id", action="store_true", help="interpret ITEM as a Hotpot ID")
    modes.add_argument("--identifier", nargs="?", const=_AUTO_IDENTIFIER, metavar="NAMESPACE", help="interpret ITEM as an external identifier")
    export.add_argument("--format", metavar="FORMAT", help="convert structures to FORMAT")

    detail = _command(actions, "detail", help="inspect current storage", description="Current-store diagnostics.")
    detail_actions = detail.add_subparsers(dest="detail_action", metavar="COMMAND")
    for name in ("status", "audit"):
        child = _command(detail_actions, name, help=name, description=f"Current-store {name}.")
        _database(child)
    query = _command(detail_actions, "query", help="query facts", description="Query current stored facts.")
    targets = query.add_subparsers(dest="detail_query_target", metavar="TARGET", required=True)
    claims = _command(targets, "claims", help="query claims", description="Query Case claims.")
    _database(claims)
    claims.add_argument("--dataset", metavar="NAME", help="restrict claims to one dataset")
    claims.add_argument("--evidence-level", metavar="LEVEL", help="restrict claims to one Case E level")
    claims.add_argument("--limit", type=int, metavar="N", help="return at most N claims")

    bundle = _command(actions, "bundle", help="manage a Bundle", description="Build, inspect, edit, and synchronize one Bundle.")
    _bundle_locator_options(bundle)
    bundle_actions = bundle.add_subparsers(dest="bundle_action", metavar="COMMAND", required=True)
    build = _command(bundle_actions, "build", help="build canonical facts", description="Build an unbuilt current intake root once.")
    build.add_argument("bundle_root", metavar="BUNDLE_ROOT", help="unbuilt current-format intake root")
    build.add_argument(
        "--dataset",
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="assert the authoritative bundle.json dataset name",
    )
    _database(build)
    for name in ("status", "synchronize"):
        _command(bundle_actions, name, help=name, description=f"Bundle {name}.")

    bundle_get = _command(bundle_actions, "get", help="query Bundle state", description="Read one allowlisted Bundle view.")
    bundle_get.add_argument(
        "view",
        choices=("info", "summary", "cases", "case.info", "case.assessment"),
        help="allowlisted Bundle or Case view",
    )
    bundle_get.add_argument("case_id", nargs="?", metavar="CASE_ID", help="Case ID for a case.* view")
    _selection_flags(bundle_get)

    bundle_set = _command(bundle_actions, "set", help="edit Case facts", description="Invoke one allowlisted Case mutation from JSON.")
    bundle_set.add_argument(
        "method",
        choices=(
            "case.set_core_info",
            "case.replace_structure",
            "case.add_observation",
            "case.add_evidence",
            "case.verify_evidence",
            "case.attest_benchmark",
        ),
        help="allowlisted Case method",
    )
    bundle_set.add_argument("input", metavar="JSON_FILE", help="closed method-argument JSON object")

    gate = _command(bundle_actions, "gate", help="inspect or edit Gates", description="Inspect or atomically replace one new-dataset Gate.")
    gate_actions = gate.add_subparsers(dest="gate_name", metavar="GATE", required=True)
    for gate_name in ("benchmark", "quality", "evidence"):
        child = _command(gate_actions, gate_name, help=gate_name, description=f"Inspect or edit the {gate_name} Gate.")
        child.add_argument("--show", action="store_true", help="show the current immutable Gate")
        child.add_argument("--add-check", action="append", default=[], metavar="NAME[=BOOL]", help="add or replace an ordinary check")
        child.add_argument("--rm-check", action="append", default=[], metavar="NAME", help="remove an ordinary check")
        child.add_argument("--checker", action="append", default=[], metavar="NAME=FILE:ENTRYPOINT", help="bind source for a same-named custom check")
        if gate_name in {"quality", "evidence"}:
            child.add_argument("--set-level", action="append", default=[], metavar="LEVEL=NAMES", help="set one cumulative B/E check list")
            child.add_argument("--rm-level", action="append", default=[], metavar="LEVEL", help="remove a level before replacing it in the same edit")
        if gate_name == "quality":
            child.add_argument("--add-structure-check", action="append", default=[], metavar="NAME[=BOOL]", help="add or replace a structure check")
            child.add_argument("--rm-structure-check", action="append", default=[], metavar="NAME", help="remove a structure check")
        if gate_name == "benchmark":
            child.add_argument("--add-direct", action="append", default=[], metavar="NAME[=BOOL]", help="add or replace a direct-entry check")
            child.add_argument("--rm-direct", action="append", default=[], metavar="NAME", help="remove a direct-entry check")
            child.add_argument("--add-veto", action="append", default=[], metavar="NAME", help="add a benchmark veto")
            child.add_argument("--rm-veto", action="append", default=[], metavar="NAME", help="remove a benchmark veto")
            child.add_argument("--set-tolerance", action="append", default=[], metavar="PROPERTY=ABS[,REL]", help="set numeric conflict tolerances")
            child.add_argument("--rm-tolerance", action="append", default=[], metavar="PROPERTY", help="remove numeric conflict tolerances")

    certify = _command(actions, "certify", help="write the Bundle certificate", description="Certify one synchronized built Bundle.")
    certify.add_argument("bundle_root", metavar="BUNDLE_ROOT", help="built synchronized Bundle root")
    certify.add_argument("--dataset", default=argparse.SUPPRESS, metavar="NAME", help="assert bundle.json.ds_name")
    _database(certify)

    ingest = _command(actions, "ingest", help="ingest a certified Bundle", description="Ingest without implicit build, synchronize, or certify.")
    ingest.add_argument("bundle_root", metavar="BUNDLE_ROOT", help="built synchronized certified Bundle root")
    ingest.add_argument("--dataset", default=argparse.SUPPRESS, metavar="NAME", help="assert bundle.json.ds_name")
    _database(ingest, required=True)
    ingest.add_argument("--include-incomplete", action="store_true", help="force incomplete archival into its separate partition")
    ingest.add_argument("--reason", help="required justification for --include-incomplete")


def _db(args: argparse.Namespace) -> Optional[str]:
    return getattr(args, "db", None)


def _version(args: argparse.Namespace) -> Optional[str]:
    return getattr(args, "version", None)


def _resolved(value: Any) -> str:
    return str(Path(value).expanduser().resolve())


def _active_database(args: argparse.Namespace) -> Path:
    return Path(_db(args) or DEFAULT_STORE_PATH).expanduser().resolve()


def _dataset_locator(dataset: Any) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "release_id": dataset.release_id,
        "version": dataset.version,
        "version_serial": f"{dataset.version_serial:03d}",
        "database": str(dataset._store.path),
    }


def _bundle_locator(bundle: Bundle) -> dict[str, Any]:
    return {
        "root": str(bundle.root),
        "dataset": bundle.name,
        "release_id": str(bundle._state.manifest["release_id"]),
        "database": str(bundle._database),
    }


def _manifest_dataset(root: Path, asserted: Optional[str]) -> str:
    path = root / "bundle.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    name = document.get("ds_name")
    if not isinstance(name, str) or not name:
        raise ValueError("bundle.json.ds_name must be a non-empty string")
    if asserted is not None and asserted != name:
        raise ValueError(
            f"--dataset {asserted!r} does not match bundle.json.ds_name {name!r}"
        )
    return name


def _bundle_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "bundle_root_option", ".")).expanduser().resolve()


def _open_bundle(args: argparse.Namespace) -> Bundle:
    root = _bundle_root(args)
    name = _manifest_dataset(root, getattr(args, "bundle_dataset", None))
    return load_bundle(root, name, db=getattr(args, "bundle_db", None))


def _load_dataset(args: argparse.Namespace):
    return load(
        args.dataset,
        version=_version(args),
        db=_db(args),
        archive_class=getattr(args, "archive_class", None),
    )


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


def _summaries(values: Any, limit: Optional[int] = None) -> list[dict[str, Any]]:
    selected = values if limit is None else values[:limit]
    return [
        {
            "id": value.id,
            "record_id": value.record_id,
            "local_id": value.local_id,
            "labels": value.label,
            "quality_level": value.quality_level,
            "evidence_level": value.evidence_level,
            "archive_class": value.archive_class,
        }
        for value in selected
    ]


def _identifier_namespace(dataset: Any, requested: Any) -> str:
    namespaces = dataset.identifier_namespaces
    if requested is _AUTO_IDENTIFIER:
        if len(namespaces) != 1:
            raise ValueError(
                "bare --identifier requires exactly one namespace; available: "
                + (", ".join(namespaces) or "none")
            )
        return namespaces[0]
    name = str(requested)
    exact = [item for item in namespaces if item.casefold() == name.casefold()]
    if len(exact) == 1:
        return exact[0]
    raise ValueError(
        f"unknown identifier namespace {name!r}; available: "
        + (", ".join(namespaces) or "none")
    )


def _select_data(dataset: Any, selector: str, *, identifier: Any, by_id: bool):
    if identifier is not None:
        namespace = _identifier_namespace(dataset, identifier)
        matches = dataset.subset(**{namespace: selector})
        if len(matches) != 1:
            raise ValueError(
                f"identifier {selector!r} in {namespace!r} matched {len(matches)} records"
            )
        return matches[0], {"kind": "identifier", "namespace": namespace, "value": selector}
    if by_id:
        matches = [item for item in dataset if item.id == selector]
        if len(matches) != 1:
            raise ValueError(f"Hotpot data ID {selector!r} was not found")
        return matches[0], {"kind": "id", "value": selector}
    try:
        index = int(selector)
    except ValueError as error:
        raise ValueError(
            f"index selector {selector!r} is not an integer; use --identifier or --id"
        ) from error
    try:
        return dataset[index], {"kind": "index", "index": index}
    except IndexError as error:
        raise ValueError(f"sequence index {index} is out of range") from error


def _protocol_selection(args: argparse.Namespace, consumer: str):
    resolved = read_input(consumer, sys.stdin)
    locator = resolved["bindings"]["dataset"]
    if locator["name"] != args.dataset:
        raise ValueError("protocol dataset does not match positional DATASET")
    explicit = _db(args)
    if explicit is not None and _resolved(explicit) != _resolved(locator["database"]):
        raise ValueError("--db does not match protocol stdin")
    dataset = load(
        args.dataset,
        version=_version(args) or locator["release_id"],
        db=explicit or locator["database"],
        archive_class=getattr(args, "archive_class", None),
    )
    if _dataset_locator(dataset) != {**locator, "database": _resolved(locator["database"])}:
        raise ValueError("protocol dataset locator does not match the current store")
    by_id = {item.id: item for item in dataset}
    records = []
    for selector in resolved["bindings"]["selectors"]:
        if selector not in by_id:
            raise ValueError(f"Hotpot data ID {selector!r} was not found")
        records.append(by_id[selector])
    return dataset, records


def _present(representations: list[dict[str, Any]]) -> None:
    if not representations or not sys.stderr.isatty():
        return
    for value in representations:
        if len(representations) > 1:
            sys.stderr.write(
                f"--- {value['data_id']} {value['structure_id']} [{value['format']}] ---\n"
            )
        sys.stderr.write(value["content"])
        if not value["content"].endswith("\n"):
            sys.stderr.write("\n")


def _export_selection(dataset: Any, args: argparse.Namespace):
    if args.item is not None:
        data, selection = _select_data(
            dataset, args.item, identifier=args.identifier, by_id=args.id
        )
        return dataset._select([data]), selection
    if args.index or args.id or args.identifier is not None:
        raise ValueError("ITEM is required with --index, --id, or --identifier")
    if args.index_range is not None:
        start, stop = args.index_range
        if start < 0 or stop < start or stop > len(dataset):
            raise ValueError(f"range must satisfy 0 <= START <= STOP <= {len(dataset)}")
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


def _case_criteria(args: argparse.Namespace) -> dict[str, Any]:
    names = {
        "archive_class": args.archive_class,
        "quality_level": args.quality,
        "min_quality": args.min_quality,
        "evidence_level": args.evidence,
        "min_evidence": None if args.min_evidence == "E0" else args.min_evidence,
        "benchmark_status": args.benchmark_status,
        "reason": args.reason,
        "verification": args.verification,
        "attestation": args.attestation,
    }
    return {name: value for name, value in names.items() if value is not None}


def _case_by_id(bundle: Bundle, case_id: str):
    return bundle._case_object(case_id)


def _read_mutation(path: str) -> dict[str, Any]:
    def reject_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate mutation JSON key {key!r}")
            result[key] = item
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite mutation JSON number {value!r} is not allowed")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("mutation JSON must be one object")
    return value


_MUTATIONS = {
    "case.set_core_info": ({"case_id", "name", "value"}, {"reason"}, "set_core_info"),
    "case.replace_structure": ({"case_id", "structure_id", "data", "reason"}, set(), "replace_structure"),
    "case.add_observation": ({"case_id", "data"}, set(), "add_observation"),
    "case.add_evidence": (
        {"case_id", "target", "source"},
        {"relation", "locator", "note", "details"},
        "add_evidence",
    ),
    "case.verify_evidence": (
        {"case_id", "evidence_id", "method", "verifier"},
        {"note", "details"},
        "verify_evidence",
    ),
    "case.attest_benchmark": ({"case_id", "approved", "authority", "actor"}, {"reason"}, "attest_benchmark"),
}


def _validate_mutation(action: str, document: Mapping[str, Any]) -> None:
    string_fields = {
        "case_id", "name", "structure_id", "reason", "target", "source",
        "relation", "locator", "note", "evidence_id", "method", "verifier",
        "authority", "actor",
    }
    nullable = {"reason", "locator", "note"}
    for name in string_fields.intersection(document):
        value = document[name]
        if value is None and name in nullable:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{action}.{name} must be a non-empty string")
    if "data" in document and not isinstance(document["data"], Mapping):
        raise ValueError(f"{action}.data must be a JSON object")
    if "details" in document and not isinstance(document["details"], Mapping):
        raise ValueError(f"{action}.details must be a JSON object")
    if action == "case.attest_benchmark":
        if type(document["approved"]) is not bool:
            raise ValueError("case.attest_benchmark.approved must be a boolean")
        if document["authority"] not in {"User", "Other"}:
            raise ValueError("case.attest_benchmark.authority must be User or Other")
    if action == "case.verify_evidence" and document["method"] not in {"AI", "Human"}:
        raise ValueError("case.verify_evidence.method must be AI or Human")


def _mutate_case(bundle: Bundle, action: str, path: str) -> tuple[str, str]:
    required, optional, method_name = _MUTATIONS[action]
    document = _read_mutation(path)
    if not required.issubset(document) or set(document) - required - optional:
        raise ValueError(
            f"{action} JSON fields must include {sorted(required)} and only allow {sorted(optional)}"
        )
    _validate_mutation(action, document)
    case_id = str(document.pop("case_id"))
    case = _case_by_id(bundle, case_id)
    method = getattr(case, method_name)
    if method_name == "set_core_info":
        method(document.pop("name"), document.pop("value"), **document)
    elif method_name == "replace_structure":
        method(document.pop("structure_id"), document.pop("data"), **document)
    elif method_name == "add_observation":
        method(document.pop("data"))
    elif method_name == "add_evidence":
        method(document.pop("target"), document.pop("source"), **document)
    elif method_name == "verify_evidence":
        method(document.pop("evidence_id"), **document)
    else:
        method(document.pop("approved"), **document)
    revision = bundle.status["source_revision"]
    with bundle._state.connect() as connection:
        row = connection.execute(
            "SELECT event_id FROM audit_events WHERE case_id=? AND source_revision=? "
            "ORDER BY rowid DESC LIMIT 1",
            (case_id, revision),
        ).fetchone()
    if row is None:
        raise RuntimeError("Case mutation committed without an audit event")
    return case_id, str(row[0])


def _boolean_item(value: str) -> tuple[str, bool]:
    if "=" not in value:
        return value, True
    name, raw = value.rsplit("=", 1)
    lowered = raw.casefold()
    if not name or lowered not in {"true", "false"}:
        raise ValueError(f"expected NAME[=true|false], got {value!r}")
    return name, lowered == "true"


def _assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expected NAME=VALUE, got {value!r}")
    name, raw = value.split("=", 1)
    if not name:
        raise ValueError("assignment name cannot be empty")
    return name, raw


def _checker_definitions(values: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for value in values:
        name, locator = _assignment(value)
        if ":" not in locator:
            raise ValueError("--checker requires NAME=FILE:ENTRYPOINT")
        path_text, entrypoint = locator.rsplit(":", 1)
        path = Path(path_text).expanduser().resolve()
        source = path.read_text(encoding="utf-8")
        imports = _inspect_source(source, entrypoint)
        definition = _CheckerDefinition.from_info(
            {
                "kind": "custom",
                "source": source,
                "entrypoint": entrypoint,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "imports": list(imports),
                "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
            }
        )
        if name in result:
            raise ValueError(f"duplicate --checker name {name!r}")
        result[name] = dict(definition.info())
    return result


def _put_checks(
    target: dict[str, Any],
    additions: list[str],
    removals: list[str],
    checkers: dict[str, dict[str, Any]],
    referenced: set[str],
) -> None:
    for name in removals:
        target.pop(name, None)
    for raw in additions:
        name, enabled = _boolean_item(raw)
        target[name] = checkers[name] if name in checkers else {"kind": "builtin", "enabled": enabled}
        referenced.add(name)


def _edit_gate(bundle: Bundle, args: argparse.Namespace):
    current = bundle.gates[args.gate_name]
    info = current.info
    edits = any(
        getattr(args, name, None)
        for name in (
            "add_check", "rm_check", "set_level", "rm_level", "checker",
            "add_structure_check", "rm_structure_check", "add_direct", "rm_direct",
            "add_veto", "rm_veto", "set_tolerance", "rm_tolerance",
        )
    )
    if args.show:
        if edits:
            raise ValueError("--show cannot be combined with Gate edits")
        return current, "show"
    if not edits:
        raise ValueError("specify --show or at least one Gate edit")
    checkers = _checker_definitions(args.checker)
    referenced: set[str] = set()
    check_field = "checklist" if args.gate_name == "benchmark" else "checks"
    _put_checks(info[check_field], args.add_check, args.rm_check, checkers, referenced)
    if args.gate_name in {"quality", "evidence"}:
        levels = info["level_checks"]
        for level in args.rm_level:
            levels.pop(level, None)
        for value in args.set_level:
            level, raw = _assignment(value)
            levels[level] = [name for name in raw.split(",") if name]
    if args.gate_name == "quality":
        _put_checks(
            info["structure_checks"],
            args.add_structure_check,
            args.rm_structure_check,
            checkers,
            referenced,
        )
    if args.gate_name == "benchmark":
        _put_checks(
            info["direct_entry"],
            args.add_direct,
            args.rm_direct,
            checkers,
            referenced,
        )
        vetoes = set(info["vetoes"])
        vetoes.update(args.add_veto)
        vetoes.difference_update(args.rm_veto)
        info["vetoes"] = sorted(vetoes)
        tolerances = info["numeric_tolerances"]
        for name in args.rm_tolerance:
            tolerances.pop(name, None)
        for value in args.set_tolerance:
            property_name, raw = _assignment(value)
            numbers = raw.split(",")
            if len(numbers) not in {1, 2}:
                raise ValueError("--set-tolerance requires PROPERTY=ABS[,REL]")
            tolerances[property_name] = {
                "absolute": float(numbers[0]),
                "relative": float(numbers[1]) if len(numbers) == 2 else 0.0,
            }
    unused = sorted(set(checkers) - referenced)
    if unused:
        raise ValueError("unreferenced --checker definitions: " + ", ".join(unused))
    classes = {
        "evidence": EvidenceGate,
        "quality": QualityGate,
        "benchmark": BenchmarkGate,
    }
    replacement = classes[args.gate_name]._from_info(info)
    bundle.use_gate(replacement)
    return replacement, "edit"


def _store_counts(store: TargetStore) -> dict[str, int]:
    with store._connect() as connection:
        store._metadata(connection)
        return {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in sorted(store._tables(connection))
        }


def _store_audit(store: TargetStore) -> list[dict[str, Any]]:
    issues = []
    with store._connect() as connection:
        store._metadata(connection)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            issues.append({"code": "sqlite_integrity", "detail": integrity})
        for row in connection.execute("PRAGMA foreign_key_check"):
            issues.append({"code": "foreign_key", "detail": list(row)})
        assets = connection.execute(
            "SELECT content_sha256,size,relative_path FROM target_assets ORDER BY content_sha256"
        ).fetchall()
    for row in assets:
        path = store.asset_root / row["relative_path"]
        if not path.is_file():
            issues.append({"code": "missing_asset", "path": row["relative_path"]})
            continue
        content = path.read_bytes()
        if len(content) != row["size"] or hashlib.sha256(content).hexdigest() != row["content_sha256"]:
            issues.append({"code": "asset_integrity", "path": row["relative_path"]})
    return issues


def _claim_rows(store: TargetStore, dataset: Optional[str], evidence_level: Optional[str], limit: Optional[int]):
    conditions = []
    values: list[Any] = []
    if dataset is not None:
        conditions.append("c.dataset_id=?")
        values.append(dataset)
    if evidence_level is not None:
        conditions.append("a.evidence_level=?")
        values.append(evidence_level)
    query = (
        "SELECT c.dataset_id,c.release_id,c.case_id,c.source_payload_json,a.evidence_level "
        "FROM cases c JOIN case_assessments a USING(dataset_id,release_id,case_id)"
    )
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY c.dataset_id,c.release_id,c.public_index"
    rows = []
    with store._connect() as connection:
        store._metadata(connection)
        for row in connection.execute(query, values):
            facts = json.loads(row["source_payload_json"])
            for claim in facts.get("claims", []):
                rows.append(
                    {
                        "dataset": row["dataset_id"],
                        "release_id": row["release_id"],
                        "case_id": row["case_id"],
                        "evidence_level": row["evidence_level"],
                        "claim": claim,
                    }
                )
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def run(args: argparse.Namespace) -> int:
    """Execute one parsed command; every completed result emits one envelope."""

    try:
        if args.ds_action == "list":
            database = _active_database(args)
            emit_envelope(
                "hotpot.ds.list",
                {"database": str(database), "datasets": list(DatasetCatalog(database))},
            )
            return 0

        if args.ds_action == "info":
            dataset = _load_dataset(args)
            emit_envelope(
                "hotpot.ds.info",
                {"dataset": _dataset_locator(dataset), "info": dataset.info},
            )
            return 0

        if args.ds_action == "labels":
            dataset = _load_dataset(args)
            emit_envelope(
                "hotpot.ds.labels",
                {
                    "dataset": _dataset_locator(dataset),
                    "core_labels": dataset.core_labels,
                    "labels": dataset.labels,
                },
            )
            return 0

        if args.ds_action == "get":
            if args.selector == "-":
                if args.index or args.id or args.identifier is not None:
                    raise ValueError("SELECTOR '-' cannot be combined with a selector mode")
                dataset, records = _protocol_selection(args, "hotpot.ds.get")
            else:
                dataset = _load_dataset(args)
                data, _ = _select_data(
                    dataset,
                    args.selector,
                    identifier=args.identifier,
                    by_id=args.id,
                )
                records = [data]
            selected_structure = any(
                value is not None
                for value in (args.structure_index, args.structure_role, args.structure_id)
            )
            if selected_structure and not (args.smi or args.xyz):
                raise ValueError("structure selection requires --smi or --xyz")
            representations = []
            fmt = "smi" if args.smi else "xyz" if args.xyz else None
            if fmt is not None:
                representations = [
                    render_structure_record(
                        data,
                        fmt,
                        structure_index=args.structure_index,
                        structure_role=args.structure_role,
                        structure_id=args.structure_id,
                    )
                    for data in records
                ]
            emit_envelope(
                "hotpot.ds.get",
                {
                    "dataset": _dataset_locator(dataset),
                    "items": [data.info for data in records],
                    "representations": representations,
                    "count": len(records),
                },
                channel="structure" if fmt else "default",
            )
            _present(representations)
            return 0

        if args.ds_action == "subset":
            dataset = _load_dataset(args)
            result = dataset.subset(**_criteria(args.where))
            minimum_quality = args.quality
            minimum_evidence = args.evidence
            result = result._filtered(
                lambda item: int(item.quality_level[1:]) >= minimum_quality
                and (
                    minimum_evidence == 0
                    or (
                        item.evidence_level is not None
                        and int(item.evidence_level[1:]) >= minimum_evidence
                    )
                )
            )
            items = _summaries(result, args.limit)
            emit_envelope(
                "hotpot.ds.subset",
                {
                    "dataset": _dataset_locator(dataset),
                    "items": items,
                    "representations": [],
                    "count": len(items),
                },
            )
            return 0

        if args.ds_action == "search":
            criteria = _criteria(args.where)
            if args.scope == "dataset":
                result = DatasetSearcher(**criteria).search(DatasetCatalog(_active_database(args)))
                values = result if args.limit is None else result[: args.limit]
                items = [value.info for value in values]
                emit_envelope(
                    "hotpot.ds.search",
                    {"items": items, "count": len(items)},
                    channel="dataset",
                )
                return 0
            if args.dataset is None:
                raise ValueError("DATASET is required for data and SMARTS search")
            dataset = _load_dataset(args)
            if args.scope == "smarts":
                if args.smarts is None:
                    raise ValueError("--smarts is required for SMARTS search")
                result = SmartsSearcher(args.smarts).search(dataset)
            else:
                result = DataSearcher(**criteria).search(dataset)
            items = _summaries(result, args.limit)
            emit_envelope(
                "hotpot.ds.search",
                {
                    "dataset": _dataset_locator(dataset),
                    "items": items,
                    "representations": [],
                    "count": len(items),
                },
                channel=args.scope,
            )
            return 0

        if args.ds_action == "export":
            if args.item == "-":
                if args.index or args.id or args.identifier is not None:
                    raise ValueError("ITEM '-' cannot be combined with a selector mode")
                dataset, records = _protocol_selection(args, "hotpot.ds.export")
                selected = dataset._select(records)
                selection = {"kind": "protocol", "ids": [item.id for item in records]}
            else:
                dataset = _load_dataset(args)
                selected, selection = _export_selection(dataset, args)
            summary = export_records(
                selected,
                Path(args.output),
                fmt=args.format,
                selection=selection,
            )
            emit_envelope(
                "hotpot.ds.export",
                {
                    "output": _resolved(args.output),
                    "dataset": _dataset_locator(selected),
                    "record_count": summary["record_count"],
                    "structure_count": summary["structure_count"],
                    "requested_format": summary["requested_format"],
                },
            )
            return 0

        if args.ds_action == "detail":
            if args.detail_action is None:
                emit_envelope("hotpot.ds.detail", {"commands": ["status", "audit", "query claims"]})
                return 0
            store = TargetStore(_active_database(args))
            if args.detail_action == "status":
                emit_envelope(
                    "hotpot.ds.detail.status",
                    {"database": str(store.path), "counts": _store_counts(store)},
                )
                return 0
            if args.detail_action == "audit":
                issues = _store_audit(store)
                emit_envelope(
                    "hotpot.ds.detail.audit",
                    {
                        "database": str(store.path),
                        "passed": not issues,
                        "issues": issues,
                        "counts": _store_counts(store),
                    },
                )
                return 0 if not issues else 2
            rows = _claim_rows(store, args.dataset, args.evidence_level, args.limit)
            emit_envelope(
                "hotpot.ds.detail.query.claims",
                {"items": rows, "count": len(rows)},
            )
            return 0

        if args.ds_action == "bundle":
            if args.bundle_action == "build":
                if any(
                    hasattr(args, name)
                    for name in ("bundle_root_option", "bundle_dataset", "bundle_db")
                ):
                    raise ValueError(
                        "bundle build takes BUNDLE_ROOT, --dataset, and --db after build"
                    )
                root = Path(args.bundle_root).expanduser().resolve()
                name = _manifest_dataset(root, getattr(args, "dataset", None))
                bundle = Bundle(root, name, db=_db(args))
                status = bundle.status
                emit_envelope(
                    "hotpot.ds.bundle.build",
                    {
                        "bundle": _bundle_locator(bundle),
                        "mode": bundle.mode,
                        "source_revision": status["source_revision"],
                        "policy_revision": status["policy_revision"],
                        "policy_digest": status["policy_digest"],
                        "is_synchronized": False,
                    },
                )
                return 0
            bundle = _open_bundle(args)
            if args.bundle_action == "status":
                status = bundle.status
                fields = (
                    "source_revision", "source_digest", "policy_revision", "policy_digest",
                    "generation_id", "generation_digest", "is_synchronized", "certificate_state",
                    "runtime_policy_current",
                    "target_precondition_satisfied", "ingest_ready", "release_state", "next_action",
                )
                emit_envelope(
                    "hotpot.ds.bundle.status",
                    {"bundle": _bundle_locator(bundle), **{name: status[name] for name in fields}},
                )
                return 0
            if args.bundle_action == "synchronize":
                result = bundle.synchronize()
                counts = result["archive_counts"]
                result["archive_counts"] = {
                    name: int(counts.get(name, 0))
                    for name in ("incomplete", "ordinary", "benchmark")
                }
                emit_envelope(
                    "hotpot.ds.bundle.synchronize",
                    {"bundle": _bundle_locator(bundle), **result},
                )
                return 0
            if args.bundle_action == "get":
                locator = _bundle_locator(bundle)
                if args.view in {"case.info", "case.assessment"}:
                    if args.case_id is None:
                        raise ValueError(f"{args.view} requires CASE_ID")
                    case = _case_by_id(bundle, args.case_id)
                    if args.view == "case.info":
                        emit_envelope(
                            "hotpot.ds.bundle.get",
                            {"bundle": locator, "case_id": case.id, "info": case.info},
                            channel="case_info",
                        )
                    else:
                        emit_envelope(
                            "hotpot.ds.bundle.get",
                            {"bundle": locator, "case_id": case.id, "assessment": case.assessment},
                            channel="case_assessment",
                        )
                    return 0
                if args.case_id is not None:
                    raise ValueError("CASE_ID applies only to case.info and case.assessment")
                if args.view == "info":
                    emit_envelope("hotpot.ds.bundle.get", {"bundle": locator, "info": bundle.info}, channel="info")
                elif args.view == "summary":
                    emit_envelope("hotpot.ds.bundle.get", {"bundle": locator, "summary": bundle.summary}, channel="summary")
                else:
                    bundle._require_generation()
                    criteria = _case_criteria(args)
                    values = bundle.cases.select(**criteria) if criteria else bundle.cases
                    emit_envelope(
                        "hotpot.ds.bundle.get",
                        {"bundle": locator, "criteria": criteria, "items": [{"case_id": case.id} for case in values]},
                        channel="cases",
                    )
                return 0
            if args.bundle_action == "set":
                case_id, event_id = _mutate_case(bundle, args.method, args.input)
                emit_envelope(
                    "hotpot.ds.bundle.set",
                    {
                        "bundle": _bundle_locator(bundle),
                        "action": args.method,
                        "case_id": case_id,
                        "event_ids": [event_id],
                        "source_revision": bundle.status["source_revision"],
                        "is_synchronized": False,
                    },
                )
                return 0
            replacement, action = _edit_gate(bundle, args)
            status = bundle.status
            emit_envelope(
                "hotpot.ds.bundle.gate",
                {
                    "bundle": _bundle_locator(bundle),
                    "gate": args.gate_name,
                    "action": action,
                    "configuration": replacement.info,
                    "policy_revision": status["policy_revision"],
                    "policy_digest": status["policy_digest"],
                    "is_synchronized": bundle.is_synchronized,
                },
            )
            return 0

        if args.ds_action == "certify":
            root = Path(args.bundle_root).expanduser().resolve()
            name = _manifest_dataset(root, getattr(args, "dataset", None))
            bundle = load_bundle(root, name, db=_db(args))
            path = bundle.write_certificate()
            status = bundle.status
            emit_envelope(
                "hotpot.ds.certify",
                {
                    "bundle": _bundle_locator(bundle),
                    "certificate_path": str(path),
                    "certificate_digest": certificate_digest(path),
                    "source_digest": status["source_digest"],
                    "policy_digest": status["policy_digest"],
                    "generation_digest": status["generation_digest"],
                },
            )
            return 0

        if args.ds_action == "ingest":
            root = Path(args.bundle_root).expanduser().resolve()
            name = _manifest_dataset(root, getattr(args, "dataset", None))
            bundle = load_bundle(root, name, db=args.db)
            idempotent = bundle._authoritative_receipt() is not None
            snapshot = bundle._ingestion_snapshot(
                certificate_digest(bundle.certificate_path),
                args.include_incomplete,
            )
            result = bundle.ingest(
                include_incomplete=args.include_incomplete,
                reason=args.reason,
            )
            metadata = TargetStore(args.db).metadata()
            emit_envelope(
                "hotpot.ds.ingest",
                {
                    "bundle": _bundle_locator(bundle),
                    "store": {
                        "canonical_path": _resolved(args.db),
                        "store_uuid": metadata["store_uuid"],
                        "schema_id": metadata["schema_id"],
                    },
                    "receipt_digest": result["receipt_digest"],
                    "included": {
                        "count": result["included_count"],
                        "sha256": snapshot["digests"]["included_cases"],
                    },
                    "excluded": {
                        "count": result["excluded_count"],
                        "sha256": snapshot["digests"]["excluded_cases"],
                    },
                    "idempotent": idempotent,
                },
            )
            return 0
    except Exception as error:
        print(f"hotpot ds: {error}", file=sys.stderr)
        return 2
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HELP_FORMATTER)
    add_arguments(parser)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
