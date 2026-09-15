"""Run the optional Hotpot/RDKit/Open Babel SMARTS differential audit."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, Optional

import networkx
from openbabel import openbabel as ob
from rdkit import Chem, rdBase

from hotpot.cheminfo.core_utils import read_mol
from hotpot.cheminfo.search.search import Searcher, Substructure

from .cases import CASES, DifferentialCase
from .normalize import (
    embeddings_from_hotpot_hits,
    normalize_embeddings,
    target_atom_sets,
    unique_embeddings,
    unique_embeddings_from_hotpot_hits,
)


def _package_version(distribution: str) -> Optional[str]:
    try:
        value = metadata.distribution(distribution).metadata.get("Version")
    except metadata.PackageNotFoundError:
        return None
    return None if value is None else str(value)


def _git_revision() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def engine_metadata() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "hotpot": {
            "distribution": "hotpot-zzy",
            "package_version": _package_version("hotpot-zzy"),
            "git_revision": _git_revision(),
            "backend": f"NetworkX {networkx.__version__}",
            "options": {
                "aromaticity": "Open Babel target perception exposed through Hotpot",
                "chirality": "unsupported",
                "hydrogens": "Hotpot Atom implicit/explicit hydrogen properties",
                "uniquification": "all query mappings grouped by target atom set",
            },
        },
        "rdkit": {
            "runtime_version": rdBase.rdkitVersion,
            "package_version": _package_version("rdkit"),
            "options": {
                "sanitize_target": True,
                "useChirality": False,
                "uniquify_raw": False,
                "uniquify_secondary": True,
                "maxMatches": 1_000_000,
            },
        },
        "openbabel": {
            "runtime_version": ob.OBReleaseVersion(),
            "package_version": _package_version("openbabel-wheel"),
            "options": {
                "target_format": "smi",
                "single_match": False,
                "raw": "OBSmartsPattern.GetMapList",
                "unique": "OBSmartsPattern.GetUMapList",
            },
        },
    }


def _success_result(
    raw_embeddings: object,
    engine_unique_embeddings: object,
) -> Dict[str, object]:
    raw = normalize_embeddings(raw_embeddings)
    unique = unique_embeddings(engine_unique_embeddings)
    atom_sets = target_atom_sets(raw)
    return {
        "query_accepted": True,
        "target_accepted": True,
        "matched": bool(raw),
        "raw_embedding_count": len(raw),
        "engine_unique_embedding_count": len(unique),
        "unique_target_atom_set_count": len(atom_sets),
        "embeddings": [list(item) for item in raw],
        "engine_unique_embeddings": [list(item) for item in unique],
        "unique_target_atom_sets": [list(item) for item in atom_sets],
        "embedding_coordinate_system": "engine-local target atom indices",
        "error": None,
    }


def _failure_result(
    *,
    query_accepted: bool,
    target_accepted: Optional[bool],
    phase: str,
    error: object,
) -> Dict[str, object]:
    return {
        "query_accepted": query_accepted,
        "target_accepted": target_accepted,
        "matched": None,
        "raw_embedding_count": None,
        "engine_unique_embedding_count": None,
        "unique_target_atom_set_count": None,
        "embeddings": None,
        "engine_unique_embeddings": None,
        "unique_target_atom_sets": None,
        "embedding_coordinate_system": None,
        "error": {
            "phase": phase,
            "type": type(error).__name__,
            "diagnostic": str(error),
        },
    }


def run_hotpot(case: DifferentialCase) -> Dict[str, object]:
    try:
        query = Substructure.from_smarts(case.smarts)
    except Exception as exc:
        return _failure_result(
            query_accepted=False, target_accepted=None, phase="query_compile", error=exc
        )
    try:
        molecule = read_mol(case.smiles, fmt="smi")
    except Exception as exc:
        return _failure_result(
            query_accepted=True,
            target_accepted=False,
            phase="target_prepare",
            error=exc,
        )
    try:
        hits = Searcher(query).search(molecule)
        raw = embeddings_from_hotpot_hits(hits)
        unique = unique_embeddings_from_hotpot_hits(hits)
    except Exception as exc:
        return _failure_result(
            query_accepted=True, target_accepted=True, phase="match", error=exc
        )
    return _success_result(raw, unique)


def run_rdkit(case: DifferentialCase) -> Dict[str, object]:
    try:
        query = Chem.MolFromSmarts(case.smarts)
    except Exception as exc:
        return _failure_result(
            query_accepted=False, target_accepted=None, phase="query_compile", error=exc
        )
    if query is None:
        return _failure_result(
            query_accepted=False,
            target_accepted=None,
            phase="query_compile",
            error=ValueError("Chem.MolFromSmarts returned None"),
        )
    try:
        molecule = Chem.MolFromSmiles(case.smiles, sanitize=True)
    except Exception as exc:
        return _failure_result(
            query_accepted=True,
            target_accepted=False,
            phase="target_prepare",
            error=exc,
        )
    if molecule is None:
        return _failure_result(
            query_accepted=True,
            target_accepted=False,
            phase="target_prepare",
            error=ValueError("Chem.MolFromSmiles returned None"),
        )
    try:
        raw = molecule.GetSubstructMatches(
            query, uniquify=False, useChirality=False, maxMatches=1_000_000
        )
        unique = molecule.GetSubstructMatches(
            query, uniquify=True, useChirality=False, maxMatches=1_000_000
        )
    except Exception as exc:
        return _failure_result(
            query_accepted=True, target_accepted=True, phase="match", error=exc
        )
    return _success_result(raw, unique)


def run_openbabel(case: DifferentialCase) -> Dict[str, object]:
    try:
        pattern = ob.OBSmartsPattern()
        accepted = bool(pattern.Init(case.smarts))
    except Exception as exc:
        return _failure_result(
            query_accepted=False, target_accepted=None, phase="query_compile", error=exc
        )
    if not accepted:
        return _failure_result(
            query_accepted=False,
            target_accepted=None,
            phase="query_compile",
            error=ValueError("OBSmartsPattern.Init returned false"),
        )
    conversion = ob.OBConversion()
    conversion.SetInFormat("smi")
    molecule = ob.OBMol()
    try:
        target_accepted = bool(conversion.ReadString(molecule, case.smiles))
    except Exception as exc:
        return _failure_result(
            query_accepted=True,
            target_accepted=False,
            phase="target_prepare",
            error=exc,
        )
    if not target_accepted:
        return _failure_result(
            query_accepted=True,
            target_accepted=False,
            phase="target_prepare",
            error=ValueError("OBConversion.ReadString returned false"),
        )
    try:
        pattern.Match(molecule, False)
        raw = normalize_embeddings(pattern.GetMapList(), index_base=1)
        unique = normalize_embeddings(pattern.GetUMapList(), index_base=1)
    except Exception as exc:
        return _failure_result(
            query_accepted=True, target_accepted=True, phase="match", error=exc
        )
    return _success_result(raw, unique)


ENGINE_RUNNERS: Dict[str, Callable[[DifferentialCase], Dict[str, object]]] = {
    "hotpot": run_hotpot,
    "rdkit": run_rdkit,
    "openbabel": run_openbabel,
}


def classify(case: DifferentialCase, engines: Dict[str, Dict[str, object]]) -> str:
    if case.group == "hotpot_extension":
        return "extension_only"

    target_acceptance = {
        result["target_accepted"]
        for result in engines.values()
        if result["query_accepted"] and result["target_accepted"] is not None
    }
    if len(target_acceptance) > 1:
        return "preprocessing_disagreement"

    reference_signature = tuple(
        (engines[name]["query_accepted"], engines[name]["matched"])
        for name in ("rdkit", "openbabel")
    )
    if reference_signature[0] != reference_signature[1]:
        return "oracle_disagreement"

    project_signature = (
        engines["hotpot"]["query_accepted"],
        engines["hotpot"]["matched"],
    )
    contract_groups = {"safe_intersection", "element_equivalence", "invalid_syntax"}
    if case.group in contract_groups and project_signature != reference_signature[0]:
        expected_signature = (case.expected_query_accepted, case.expected_matched)
        if project_signature == expected_signature:
            return "oracle_disagreement"
        if reference_signature[0] == expected_signature:
            return "project_mismatch"
        return "unresolved"

    signatures = {
        (result["query_accepted"], result["target_accepted"], result["matched"])
        for result in engines.values()
    }
    if len(signatures) == 1:
        if case.compare_enumeration and engines["hotpot"]["matched"] is not None:
            enumeration = {
                (
                    result["raw_embedding_count"],
                    result["unique_target_atom_set_count"],
                )
                for result in engines.values()
            }
            if len(enumeration) > 1:
                return "enumeration_disagreement"
        return "unanimous"

    if case.group in {"unsupported_feature", "dialect_boundary"}:
        return "extension_only"
    return "unresolved"


def run_case(case: DifferentialCase) -> Dict[str, object]:
    engines = {name: runner(case) for name, runner in ENGINE_RUNNERS.items()}
    hotpot_result = engines["hotpot"]
    contract_checks = []
    if case.expected_query_accepted is not None:
        contract_checks.append(
            hotpot_result["query_accepted"] == case.expected_query_accepted
        )
    if case.expected_matched is not None:
        contract_checks.append(hotpot_result["matched"] == case.expected_matched)
    return {
        "id": case.case_id,
        "smarts": case.smarts,
        "target_smiles": case.smiles,
        "case_group": case.group,
        "features": list(case.features),
        "evidence_tier": case.evidence_tier,
        "manual_expected_matched": case.expected_matched,
        "manual_expected_query_accepted": case.expected_query_accepted,
        "compare_enumeration": case.compare_enumeration,
        "equivalence_key": case.equivalence_key,
        "equivalence_variant": case.equivalence_variant,
        "notes": case.notes,
        "classification": classify(case, engines),
        "manual_contract_agrees": all(contract_checks) if contract_checks else None,
        "engines": engines,
    }


def build_element_equivalence_matrix(results: object) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for result in results:
        key = result["equivalence_key"]
        variant = result["equivalence_variant"]
        if key is not None and variant is not None:
            grouped.setdefault(key, {})[variant] = result

    rows = []
    mismatch_counts = {engine: 0 for engine in ENGINE_RUNNERS}
    for key in sorted(grouped):
        variants = grouped[key]
        atomic_number = variants["atomic_number"]
        symbol = variants["symbol"]
        engines = {}
        for engine in ENGINE_RUNNERS:
            left = atomic_number["engines"][engine]
            right = symbol["engines"][engine]
            left_signature = (
                left["query_accepted"],
                left["target_accepted"],
                left["matched"],
                left["raw_embedding_count"],
                left["engine_unique_embedding_count"],
                left["unique_target_atom_set_count"],
                left["embeddings"],
            )
            right_signature = (
                right["query_accepted"],
                right["target_accepted"],
                right["matched"],
                right["raw_embedding_count"],
                right["engine_unique_embedding_count"],
                right["unique_target_atom_set_count"],
                right["embeddings"],
            )
            equivalent = left_signature == right_signature
            if not equivalent:
                mismatch_counts[engine] += 1
            engines[engine] = {
                "equivalent": equivalent,
                "atomic_number_query_accepted": left["query_accepted"],
                "symbol_query_accepted": right["query_accepted"],
                "atomic_number_matched": left["matched"],
                "symbol_matched": right["matched"],
            }
        rows.append({"key": key, "engines": engines})
    return {
        "element_count": len(rows),
        "queries_per_element": 2,
        "mismatch_counts": mismatch_counts,
        "rows": rows,
    }


def build_report(selected_ids: Optional[set] = None) -> Dict[str, object]:
    selected = [
        case for case in CASES if selected_ids is None or case.case_id in selected_ids
    ]
    results = [run_case(case) for case in selected]
    counts: Dict[str, int] = {}
    for result in results:
        label = str(result["classification"])
        counts[label] = counts.get(label, 0) + 1
    return {
        "schema_version": 1,
        "purpose": "optional multi-engine audit; no single engine is normative",
        "count_semantics": {
            "raw_embedding_count": "all query-order embeddings reported by the engine, including exact duplicates",
            "engine_unique_embedding_count": "engine-specific uniquified embeddings; diagnostic only in cross-engine comparisons",
            "unique_target_atom_set_count": "embeddings collapsed by unordered target atom set",
        },
        "identity_policy": {
            "coordinate_system": "Each engine reports its own zero-based target atom indices.",
            "cross_engine_embedding_identity_comparison": False,
            "reason": "SMILES readers are not assumed to preserve shared atom numbering. Cross-engine classification compares acceptance, existence, raw embedding count, and normalized target-set count; engine-specific unique modes and embedding tuples are diagnostic evidence only.",
        },
        "versions": engine_metadata(),
        "summary": {"case_count": len(results), "classifications": counts},
        "element_equivalence_matrix": build_element_equivalence_matrix(results),
        "cases": results,
    }


def _parse_args(argv: object = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write complete JSON report")
    parser.add_argument("--case-id", action="append", help="run only this case ID")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero for project/manual-contract mismatches",
    )
    return parser.parse_args(argv)


def main(argv: object = None) -> int:
    args = _parse_args(argv)
    selected = set(args.case_id) if args.case_id else None
    report = build_report(selected)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))

    if not args.strict:
        print("audit mode: findings are recorded but do not change the exit status")
        return 0
    failures = [
        case
        for case in report["cases"]
        if case["classification"] == "project_mismatch"
        or case["manual_contract_agrees"] is False
    ]
    if failures:
        print("strict differential failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure['id']}: {failure['classification']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
