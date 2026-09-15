"""Deterministic short-run grammar, invalid-mutation, and robustness audit."""

from __future__ import annotations

import argparse
import json
import platform
import random
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from hotpot.cheminfo.core_utils import read_mol
from hotpot.cheminfo.search.search import Searcher, Substructure

from .generators import (
    generate_robustness_text,
    generate_structured_stress_smarts,
    generate_valid_smarts,
    mutate_valid_smarts,
)


DEFAULT_SEED = 20260915
EXPECTED_REJECTIONS = (ValueError, NotImplementedError)
TARGETS = ("C", "CCO", "c1ccccc1", "CC(=O)N", "C1CCCCC1")


class CaseTimeout(RuntimeError):
    pass


@contextmanager
def deadline(seconds: float):
    """Bound one in-process parser/matcher call on the supported Unix platform."""

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise CaseTimeout(f"case exceeded {seconds:.3f} s")

    previous = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def evaluate_query(
    text: str, molecules: Iterable[object], timeout: float
) -> Dict[str, object]:
    started = time.perf_counter()
    phase = "query_compile"
    try:
        with deadline(timeout):
            query = Substructure.from_smarts(text)
            phase = "match"
            match_counts = tuple(
                len(Searcher(query).search(molecule)) for molecule in molecules
            )
    except EXPECTED_REJECTIONS as exc:
        return {
            "outcome": "rejected",
            "phase": phase,
            "error_type": type(exc).__name__,
            "diagnostic": str(exc),
            "elapsed_s": time.perf_counter() - started,
        }
    except CaseTimeout as exc:
        return {
            "outcome": "timeout",
            "phase": phase,
            "error_type": type(exc).__name__,
            "diagnostic": str(exc),
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "outcome": "crash",
            "phase": phase,
            "error_type": type(exc).__name__,
            "diagnostic": str(exc),
            "elapsed_s": time.perf_counter() - started,
        }
    return {
        "outcome": "accepted",
        "phase": "complete",
        "query_atom_count": len(query.query_atoms),
        "query_bond_count": len(query.query_bonds),
        "match_counts": list(match_counts),
        "elapsed_s": time.perf_counter() - started,
    }


def outcome_signature(result: Dict[str, object]) -> Tuple[object, ...]:
    """Ignore timing and prose diagnostics when checking determinism."""

    return (
        result["outcome"],
        result.get("phase"),
        result.get("error_type"),
        result.get("query_atom_count"),
        result.get("query_bond_count"),
        tuple(result.get("match_counts", ())),
    )


def _failure(
    *,
    category: str,
    iteration: int,
    text: str,
    seed: int,
    result: Dict[str, object],
    mutation: Optional[str] = None,
    base_smarts: Optional[str] = None,
    edit_position: Optional[int] = None,
    edit_payload: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "category": category,
        "iteration": iteration,
        "seed": seed,
        "mutation": mutation,
        "base_smarts": base_smarts,
        "edit_position": edit_position,
        "edit_payload": edit_payload,
        "smarts": text,
        "result": result,
    }


def run_audit(
    *,
    seed: int,
    iterations: int,
    max_atoms: int,
    max_structured_depth: int,
    max_robustness_length: int,
    timeout: float,
) -> Dict[str, object]:
    stream_seeds = {
        "valid": seed * 10 + 1,
        "invalid_mutation": seed * 10 + 2,
        "robustness": seed * 10 + 3,
        "structured_stress": seed * 10 + 4,
    }
    valid_rng = random.Random(stream_seeds["valid"])
    mutation_rng = random.Random(stream_seeds["invalid_mutation"])
    robustness_rng = random.Random(stream_seeds["robustness"])
    stress_rng = random.Random(stream_seeds["structured_stress"])
    molecules = tuple(read_mol(smiles, fmt="smi") for smiles in TARGETS)
    failures = []
    outcome_counts: Dict[str, int] = {}
    observed_inputs = {
        "valid": set(),
        "invalid": set(),
        "robustness": set(),
        "structured_stress": set(),
    }
    mutation_counts: Dict[str, int] = {}
    stress_kind_counts: Dict[str, int] = {}

    for iteration in range(iterations):
        valid = generate_valid_smarts(valid_rng, max_atoms=max_atoms)
        observed_inputs["valid"].add(valid)
        first = evaluate_query(valid, molecules, timeout)
        second = evaluate_query(valid, molecules, timeout)
        outcome_counts["valid." + str(first["outcome"])] = (
            outcome_counts.get("valid." + str(first["outcome"]), 0) + 1
        )
        if first["outcome"] != "accepted":
            failures.append(
                _failure(
                    category="valid_query_not_accepted",
                    iteration=iteration,
                    text=valid,
                    seed=seed,
                    result=first,
                )
            )
        if outcome_signature(first) != outcome_signature(second):
            failures.append(
                _failure(
                    category="nondeterministic_valid_query",
                    iteration=iteration,
                    text=valid,
                    seed=seed,
                    result={"first": first, "second": second},
                )
            )

        mutation, invalid_base, invalid, edit_position, edit_payload = (
            mutate_valid_smarts(mutation_rng, valid)
        )
        observed_inputs["invalid"].add(invalid)
        mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1
        first = evaluate_query(invalid, molecules, timeout)
        second = evaluate_query(invalid, molecules, timeout)
        outcome_counts["invalid." + str(first["outcome"])] = (
            outcome_counts.get("invalid." + str(first["outcome"]), 0) + 1
        )
        if first["outcome"] != "rejected":
            failures.append(
                _failure(
                    category="invalid_query_not_rejected",
                    iteration=iteration,
                    text=invalid,
                    seed=seed,
                    result=first,
                    mutation=mutation,
                    base_smarts=invalid_base,
                    edit_position=edit_position,
                    edit_payload=edit_payload,
                )
            )
        if outcome_signature(first) != outcome_signature(second):
            failures.append(
                _failure(
                    category="nondeterministic_invalid_query",
                    iteration=iteration,
                    text=invalid,
                    seed=seed,
                    result={"first": first, "second": second},
                    mutation=mutation,
                    base_smarts=invalid_base,
                    edit_position=edit_position,
                    edit_payload=edit_payload,
                )
            )

        arbitrary = generate_robustness_text(
            robustness_rng, max_length=max_robustness_length
        )
        observed_inputs["robustness"].add(arbitrary)
        first = evaluate_query(arbitrary, molecules, timeout)
        second = evaluate_query(arbitrary, molecules, timeout)
        outcome_counts["robustness." + str(first["outcome"])] = (
            outcome_counts.get("robustness." + str(first["outcome"]), 0) + 1
        )
        if first["outcome"] in {"crash", "timeout"}:
            failures.append(
                _failure(
                    category="robustness_" + str(first["outcome"]),
                    iteration=iteration,
                    text=arbitrary,
                    seed=seed,
                    result=first,
                )
            )
        if outcome_signature(first) != outcome_signature(second):
            failures.append(
                _failure(
                    category="nondeterministic_robustness",
                    iteration=iteration,
                    text=arbitrary,
                    seed=seed,
                    result={"first": first, "second": second},
                )
            )

        stress_kind, stress = generate_structured_stress_smarts(
            stress_rng, max_structured_depth
        )
        observed_inputs["structured_stress"].add(stress)
        stress_kind_counts[stress_kind] = stress_kind_counts.get(stress_kind, 0) + 1
        first = evaluate_query(stress, molecules, timeout)
        second = evaluate_query(stress, molecules, timeout)
        outcome_counts["structured_stress." + str(first["outcome"])] = (
            outcome_counts.get("structured_stress." + str(first["outcome"]), 0) + 1
        )
        if first["outcome"] != "accepted":
            failures.append(
                _failure(
                    category="structured_stress_not_accepted",
                    iteration=iteration,
                    text=stress,
                    seed=seed,
                    result=first,
                    mutation=stress_kind,
                )
            )
        if outcome_signature(first) != outcome_signature(second):
            failures.append(
                _failure(
                    category="nondeterministic_structured_stress",
                    iteration=iteration,
                    text=stress,
                    seed=seed,
                    result={"first": first, "second": second},
                    mutation=stress_kind,
                )
            )

    failure_counts: Dict[str, int] = {}
    for failure in failures:
        category = str(failure["category"])
        failure_counts[category] = failure_counts.get(category, 0) + 1
    return {
        "schema_version": 1,
        "seed": seed,
        "python": platform.python_version(),
        "configuration": {
            "iterations_per_category": iterations,
            "max_atoms": max_atoms,
            "max_structured_depth": max_structured_depth,
            "max_robustness_length": max_robustness_length,
            "timeout_per_evaluation_s": timeout,
            "targets": list(TARGETS),
            "invalid_generation": "one named local corruption of the generated valid query from the same iteration",
            "derived_stream_seeds": stream_seeds,
        },
        "summary": {
            "evaluated": iterations * 4,
            "outcomes": outcome_counts,
            "failure_count": len(failures),
            "failure_categories": failure_counts,
            "unique_inputs": {
                category: len(inputs) for category, inputs in observed_inputs.items()
            },
            "invalid_mutations": dict(sorted(mutation_counts.items())),
            "structured_stress_kinds": dict(sorted(stress_kind_counts.items())),
        },
        "failures": failures,
    }


def _parse_args(argv: object = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "heavy"), default="quick")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: object = None) -> int:
    args = _parse_args(argv)
    heavy = args.profile == "heavy"
    report = run_audit(
        seed=args.seed,
        iterations=args.iterations or (2_000 if heavy else 100),
        max_atoms=24 if heavy else 8,
        max_structured_depth=24 if heavy else 8,
        max_robustness_length=1_024 if heavy else 96,
        timeout=args.timeout or (1.0 if heavy else 0.25),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if report["failures"]:
        print(
            f"SMARTS robustness audit failed; reproduce with --seed {args.seed}",
            file=sys.stderr,
        )
        for failure in report["failures"][:20]:
            print(
                f"  iteration={failure['iteration']} "
                f"category={failure['category']} smarts={failure['smarts']!r}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
