"""Execute the full deterministic corpus and emit every mismatch as JSON.

Unlike the thin adapter, this report collector catches unexpected exceptions so
one parser crash does not prevent later cases from being audited. Every such
exception is recorded as a failure, and strict mode exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, Optional

from .adapter import match_smarts, parse_smarts
from .corpus import INVALID_PARSE_CASES, SEMANTIC_CASES, VALID_PARSE_CASES
from .corpus_validator import assert_valid_corpora


def _version(distribution: str) -> Optional[str]:
    try:
        value = metadata.distribution(distribution).metadata.get("Version")
    except metadata.PackageNotFoundError:
        return None
    return None if value is None else str(value)


def _crash(case: Mapping[str, object], exc: Exception) -> Mapping[str, object]:
    return {
        "case_id": case["id"],
        "classification": "unexpected_exception",
        "smarts": case["smarts"],
        "target_smiles": case.get("target", {}).get("text")
        if isinstance(case.get("target"), Mapping)
        else None,
        "expected": case.get("expected_outcome", case.get("expected")),
        "actual": {"exception_type": type(exc).__name__, "diagnostic": str(exc)},
        "failure_phase": "unknown",
        "features": list(case["features"]),
        "source": case["source"],
    }


def _collect(
    cases,
    evaluator: Callable[[Mapping[str, object]], Optional[Mapping[str, object]]],
):
    failures = []
    for case in cases:
        try:
            failure = evaluator(case)
        except Exception as exc:
            failure = _crash(case, exc)
        if failure is not None:
            failures.append(failure)
    return failures


def _audit_valid(case):
    result = parse_smarts(case["smarts"])
    if result.accepted:
        return None
    return {
        "case_id": case["id"],
        "classification": "parse_rejection",
        "smarts": case["smarts"],
        "target_smiles": None,
        "expected": "accept",
        "actual": result.error_code,
        "failure_phase": result.phase,
        "diagnostic": result.diagnostic,
        "features": list(case["features"]),
        "source": case["source"],
    }


def _audit_invalid(case):
    result = parse_smarts(case["smarts"])
    expected_phases = tuple(case["expected_phase"])
    if (
        not result.accepted
        and result.error_code == "invalid_syntax"
        and result.phase in expected_phases
    ):
        return None
    return {
        "case_id": case["id"],
        "classification": "invalid_query_accepted"
        if result.accepted
        else "wrong_rejection_class",
        "smarts": case["smarts"],
        "target_smiles": None,
        "expected": {
            "outcome": "reject",
            "phase": list(expected_phases),
            "error_code": "invalid_syntax",
        },
        "actual": {
            "accepted": result.accepted,
            "phase": result.phase,
            "error_code": result.error_code,
        },
        "failure_phase": result.phase,
        "diagnostic": result.diagnostic,
        "features": list(case["features"]),
        "source": case["source"],
    }


def _audit_semantic(case):
    expected = bool(case["expected"]["matched"])
    target = case["target"]["text"]
    result = match_smarts(case["smarts"], target)
    if result.query_accepted and result.target_accepted and result.matched is expected:
        return None
    return {
        "case_id": case["id"],
        "classification": "semantic_mismatch",
        "smarts": case["smarts"],
        "target_smiles": target,
        "expected": {"matched": expected},
        "actual": {
            "query_accepted": result.query_accepted,
            "target_accepted": result.target_accepted,
            "matched": result.matched,
            "raw_embedding_count": result.raw_embedding_count,
            "error_code": result.error_code,
        },
        "failure_phase": result.phase,
        "diagnostic": result.diagnostic,
        "features": list(case["features"]),
        "source": case["source"],
    }


def build_report():
    assert_valid_corpora()
    failures_by_corpus = {
        "valid_parse": _collect(VALID_PARSE_CASES, _audit_valid),
        "invalid_parse": _collect(INVALID_PARSE_CASES, _audit_invalid),
        "semantic": _collect(SEMANTIC_CASES, _audit_semantic),
    }
    total_cases = (
        len(VALID_PARSE_CASES) + len(INVALID_PARSE_CASES) + len(SEMANTIC_CASES)
    )
    failure_count = sum(len(items) for items in failures_by_corpus.values())
    return {
        "schema_version": 1,
        "mode": "strict corpus conformance audit",
        "environment": {
            "python": platform.python_version(),
            "hotpot_zzy": _version("hotpot-zzy"),
            "networkx": _version("networkx"),
            "openbabel_wheel": _version("openbabel-wheel"),
            "rdkit": _version("rdkit"),
        },
        "summary": {
            "total_cases": total_cases,
            "failure_count": failure_count,
            "passed_count": total_cases - failure_count,
            "corpora": {
                "valid_parse": {
                    "cases": len(VALID_PARSE_CASES),
                    "failures": len(failures_by_corpus["valid_parse"]),
                },
                "invalid_parse": {
                    "cases": len(INVALID_PARSE_CASES),
                    "failures": len(failures_by_corpus["invalid_parse"]),
                },
                "semantic": {
                    "cases": len(SEMANTIC_CASES),
                    "failures": len(failures_by_corpus["semantic"]),
                },
            },
        },
        "failures": failures_by_corpus,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="collect all failures but return zero; default strict mode returns one",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    report = build_report()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if report["summary"]["failure_count"] and not args.report_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
