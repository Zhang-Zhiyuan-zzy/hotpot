"""Schema and governance checks for the offline SMARTS corpora."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .corpus import INVALID_PARSE_CASES, SEMANTIC_CASES, VALID_PARSE_CASES


MANIFEST_PATH = Path(__file__).with_name("corpus") / "manifest.json"


ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "valid_core",
        "valid_extension",
        "invalid_syntax",
        "unsupported_feature",
        "dialect_disputed",
        "semantic_case",
        "regression",
        "robustness_only",
    }
)
ACCEPTING_CLASSIFICATIONS = frozenset({"valid_core", "valid_extension"})


@dataclass(frozen=True)
class ValidationIssue:
    corpus: str
    case_id: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.corpus}:{self.case_id}:{self.code}: {self.message}"


def bundled_corpora() -> Mapping[str, Sequence[Mapping[str, object]]]:
    return {
        "valid_parse": VALID_PARSE_CASES,
        "invalid_parse": INVALID_PARSE_CASES,
        "semantic": SEMANTIC_CASES,
    }


def load_manifest() -> Mapping[str, object]:
    return json.loads(MANIFEST_PATH.read_text())


def validate_manifest() -> Tuple[ValidationIssue, ...]:
    manifest = load_manifest()
    issues = []
    pseudo_case = {"id": "manifest"}
    if manifest.get("schema_version") != "1.0":
        issues.append(
            _issue("manifest", pseudo_case, "schema_version", "expected schema 1.0")
        )
    declared = manifest.get("corpora", {})
    actual = {name: len(cases) for name, cases in bundled_corpora().items()}
    for name, count in actual.items():
        entry = declared.get(name, {}) if isinstance(declared, Mapping) else {}
        if entry.get("expected_count") != count:
            issues.append(
                _issue(
                    "manifest",
                    pseudo_case,
                    "corpus_count",
                    f"{name} declares {entry.get('expected_count')!r}, actual {count}",
                )
            )
    provenance = manifest.get("provenance", {})
    for field in ("kind", "name", "version", "license", "reviewed"):
        if not isinstance(provenance, Mapping) or field not in provenance:
            issues.append(
                _issue("manifest", pseudo_case, "provenance", f"missing {field!r}")
            )
    return tuple(issues)


def _issue(
    corpus: str, case: Mapping[str, object], code: str, message: str
) -> ValidationIssue:
    return ValidationIssue(corpus, str(case.get("id", "<missing>")), code, message)


def _case_fingerprint(case: Mapping[str, object]) -> str:
    if case.get("classification") == "semantic_case":
        relevant = {
            "smarts": case.get("smarts"),
            "target": case.get("target"),
            "options": case.get("options"),
            "expected": case.get("expected"),
        }
    else:
        relevant = {
            "smarts": case.get("smarts"),
            "classification": case.get("classification"),
            "expected_outcome": case.get("expected_outcome"),
        }
    return json.dumps(relevant, sort_keys=True, ensure_ascii=True)


def _validate_source(
    corpus_name: str, case: Mapping[str, object]
) -> Iterable[ValidationIssue]:
    source = case.get("source")
    if not isinstance(source, Mapping):
        yield _issue(corpus_name, case, "source", "source must be a mapping")
        return
    for field in ("kind", "name", "version", "license", "reviewed"):
        if field not in source:
            yield _issue(corpus_name, case, "source", f"source is missing {field!r}")
    if source.get("kind") == "external":
        for field in ("origin", "import_method", "reference_engine"):
            if field not in source:
                yield _issue(
                    corpus_name,
                    case,
                    "external_provenance",
                    f"external source is missing {field!r}",
                )


def _validate_case(
    corpus_name: str, case: Mapping[str, object]
) -> Iterable[ValidationIssue]:
    for field in ("id", "smarts", "classification", "features", "dialect", "source"):
        if field not in case:
            yield _issue(corpus_name, case, "required_field", f"missing {field!r}")

    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        yield _issue(corpus_name, case, "id", "id must be a non-empty string")
    smarts = case.get("smarts")
    if not isinstance(smarts, str):
        yield _issue(corpus_name, case, "smarts", "smarts must be a string")
    features = case.get("features")
    if (
        not isinstance(features, (list, tuple))
        or not features
        or any(not isinstance(feature, str) or not feature for feature in features)
    ):
        yield _issue(
            corpus_name, case, "features", "features must contain non-empty tags"
        )

    classification = case.get("classification")
    if classification not in ALLOWED_CLASSIFICATIONS:
        yield _issue(
            corpus_name,
            case,
            "classification",
            f"unknown classification {classification!r}",
        )
    outcome = case.get("expected_outcome")
    if classification in ACCEPTING_CLASSIFICATIONS and outcome != "accept":
        yield _issue(
            corpus_name, case, "outcome", "legal parse case must expect accept"
        )
    if classification in {"invalid_syntax", "unsupported_feature"}:
        if outcome != "reject":
            yield _issue(corpus_name, case, "outcome", "reject case must expect reject")
        phases = case.get("expected_phase")
        if (
            not isinstance(phases, (list, tuple))
            or not phases
            or any(phase not in {"tokenize", "query_compile"} for phase in phases)
        ):
            yield _issue(
                corpus_name,
                case,
                "expected_phase",
                "reject case needs tokenize/query_compile expected phase",
            )

    if classification == "semantic_case":
        target = case.get("target")
        if (
            not isinstance(target, Mapping)
            or target.get("format") != "smiles"
            or not isinstance(target.get("text"), str)
        ):
            yield _issue(
                corpus_name, case, "target", "semantic case needs a SMILES target"
            )
        expected = case.get("expected")
        if not isinstance(expected, Mapping) or not isinstance(
            expected.get("matched"), bool
        ):
            yield _issue(
                corpus_name,
                case,
                "semantic_expected",
                "semantic expected.matched must be boolean",
            )
        oracle = case.get("oracle")
        if (
            not isinstance(oracle, Mapping)
            or "kind" not in oracle
            or oracle.get("reviewed") is not True
        ):
            yield _issue(
                corpus_name,
                case,
                "oracle",
                "semantic case needs an explicitly reviewed oracle",
            )

    if classification == "dialect_disputed" and case.get("enforce_core") is not False:
        yield _issue(
            corpus_name,
            case,
            "disputed_core",
            "dialect-disputed cases must explicitly set enforce_core=false",
        )

    if case.get("known_failure"):
        failure = case["known_failure"]
        if not isinstance(failure, Mapping):
            yield _issue(
                corpus_name, case, "known_failure", "known_failure must be a mapping"
            )
        else:
            for field in ("issue_id", "reason", "added", "exit_condition"):
                if not failure.get(field):
                    yield _issue(
                        corpus_name,
                        case,
                        "known_failure",
                        f"known_failure is missing {field!r}",
                    )

    yield from _validate_source(corpus_name, case)


def validate_corpora(
    corpora: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
) -> Tuple[ValidationIssue, ...]:
    """Return all schema/governance issues without hiding later failures."""
    selected = bundled_corpora() if corpora is None else corpora
    issues = []
    seen_ids = {}
    seen_fingerprints = {}
    for corpus_name, cases in selected.items():
        for case in cases:
            issues.extend(_validate_case(corpus_name, case))
            case_id = case.get("id")
            if case_id in seen_ids:
                issues.append(
                    _issue(
                        corpus_name,
                        case,
                        "duplicate_id",
                        f"also used in {seen_ids[case_id]}",
                    )
                )
            else:
                seen_ids[case_id] = corpus_name
            fingerprint = _case_fingerprint(case)
            if fingerprint in seen_fingerprints and not case.get("duplicate_reason"):
                issues.append(
                    _issue(
                        corpus_name,
                        case,
                        "duplicate_case",
                        f"duplicates {seen_fingerprints[fingerprint]} without a reason",
                    )
                )
            else:
                seen_fingerprints[fingerprint] = str(case_id)
    return tuple(issues)


def assert_valid_corpora(
    corpora: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
) -> None:
    issues = validate_corpora(corpora)
    if corpora is None:
        issues += validate_manifest()
    if issues:
        raise ValueError("\n".join(str(issue) for issue in issues))


def main() -> int:
    issues = validate_corpora() + validate_manifest()
    for issue in issues:
        print(issue)
    if issues:
        print(f"corpus validation failed: {len(issues)} issue(s)")
        return 1
    counts = {name: len(cases) for name, cases in bundled_corpora().items()}
    print("corpus validation passed:", json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_CLASSIFICATIONS",
    "ValidationIssue",
    "assert_valid_corpora",
    "bundled_corpora",
    "load_manifest",
    "validate_manifest",
    "validate_corpora",
]
