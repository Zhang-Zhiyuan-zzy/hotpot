"""Private integrity and summary helpers for Plan20 execution records."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "hotpot-plan20-scientific-report/v1"
OUTCOMES = {
    "completed", "no_output", "execution_error", "not_run_ineligible", "not_run_unknown"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def summarize(cases: list[dict], trials: list[dict]) -> dict:
    return {
        "case_count": len(cases),
        "eligibility_counts": dict(Counter(c["eligibility"]["state"] for c in cases)),
        "trial_count": len(trials),
        "trial_outcomes": dict(Counter(t["outcome"] for t in trials)),
        "attempted_trials": sum(t["attempted"] for t in trials),
        "unresolved_trials": sum(t["scientific_status"] == "unresolved" for t in trials),
        "execution_error_count": sum(bool(c["eligibility"].get("error")) for c in cases)
        + sum(len(t["errors"]) for t in trials),
        "trial_elapsed_seconds_sum": sum(
            t["elapsed_seconds"] for t in trials if t["elapsed_seconds"] is not None
        ),
        "trial_elapsed_unavailable_count": sum(t["elapsed_seconds"] is None for t in trials),
    }


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"nonfinite JSON literal: {value}")


def read_report(path: Path) -> tuple[dict, Path]:
    """Check report completeness and referenced bytes, not scientific values."""
    path = path / "report.json" if path.is_dir() else path
    report = json.loads(path.read_text(), parse_constant=_reject_nonfinite_constant)
    root = path.resolve().parent
    if report["schema"] != SCHEMA:
        raise ValueError("unsupported scientific report schema")
    cases = report["cases"]
    trials = report["trials"]
    case_ids = [case["case_id"] for case in cases]
    seeds = report["configuration"]["root_seeds"]
    if not cases or not seeds or len(set(case_ids)) != len(case_ids) or len(set(seeds)) != len(seeds):
        raise ValueError("empty or duplicate case/root-seed roster")
    expected = {(case_id, seed) for case_id in case_ids for seed in seeds}
    observed = [(t["case_id"], t["root_seed"]) for t in trials]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("missing, extra or duplicate trial records")
    by_id = {case["case_id"]: case for case in cases}
    if report["source"]["sha256"] != report["source"]["snapshot_artifact"]["sha256"]:
        raise ValueError("source digest differs from the frozen snapshot")
    if [case["index"] for case in cases] != report["configuration"]["selected_indices"]:
        raise ValueError("case roster differs from the requested selection")
    all_frames = []
    for trial in trials:
        case = by_id[trial["case_id"]]
        if trial["case_index"] != case["index"]:
            raise ValueError("trial source index differs from the case roster")
        if trial["effective_seed"] != trial["root_seed"] + case["index"]:
            raise ValueError("trial seed does not match its recorded derivation")
        if trial["outcome"] not in OUTCOMES:
            raise ValueError("unknown trial outcome")
        if trial["eligibility"] != case["eligibility"]["state"]:
            raise ValueError("trial eligibility differs from the case roster")
        if trial["outcome"] == "completed" and trial["selected_frame"] is None:
            raise ValueError("completed trial omitted its selected frame")
        for field in ("stages", "observations", "missing_evidence", "errors", "artifacts"):
            trial[field]
        frames = trial["last_finite_frames"] + ([trial["selected_frame"]] if trial["selected_frame"] else [])
        for frame in frames:
            if frame["coordinates"] not in trial["artifacts"]:
                raise ValueError("frame coordinates are not covered by artifact identities")
        all_frames += frames
    if report["summary"] != summarize(cases, trials):
        raise ValueError("report summary does not describe all recorded attempts")
    references = [report["source"]["snapshot_artifact"]]
    references += report["artifacts"]
    references += [a for trial in trials for a in trial["artifacts"]]
    for reference in references:
        target = (root / reference["path"]).resolve()
        if not target.is_relative_to(root) or sha256(target) != reference["sha256"]:
            raise ValueError(f"artifact identity mismatch: {reference['path']}")
    for frame in all_frames:
        coordinates = np.load(root / frame["coordinates"]["path"], allow_pickle=False)
        if list(coordinates.shape) != frame["shape"] or coordinates.shape != (len(frame["atom_identity"]), 3):
            raise ValueError("coordinate artifact shape differs from its frame/row identities")
        if bool(np.isfinite(coordinates).all()) != frame["coordinates_finite"]:
            raise ValueError("coordinate finite flag differs from its artifact")
    return report, root
