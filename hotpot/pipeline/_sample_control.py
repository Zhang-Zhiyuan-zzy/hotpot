"""Private positional sample reporting and explicit break helpers."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

from .artifacts import _write_payload
from .contracts import JSONValue, MolecularPayload, MolecularRecord, StageContext


def _sample_counts(rows: list[dict[str, JSONValue]]) -> dict[str, JSONValue]:
    return {
        "sample_outcomes_version": 1,
        "input_count": len(rows),
        "attempted_count": sum(row["outcome"] != "not-attempted" for row in rows),
        "output_count": sum(row["output_index"] is not None for row in rows),
        "unavailable_count": sum(row["outcome"] == "unavailable" for row in rows),
        "sample_break_count": sum(row["outcome"] == "sample-break" for row in rows),
        "not_attempted_count": sum(row["outcome"] == "not-attempted" for row in rows),
        "sample_outcomes": tuple(rows),
    }


def _check_findings(
    report: object,
    prefix: str,
    serializer: Callable[[object], JSONValue],
) -> tuple[dict[str, JSONValue], ...]:
    return tuple(
        {"code": f"{prefix}:{check.name}", **serializer(check)}
        for check in report.checks if not check.passed
    )


def _break_scope(
    codes: Iterable[str],
    sample_codes: Tuple[str, ...],
    batch_codes: Tuple[str, ...],
) -> Optional[str]:
    codes = tuple(codes)
    if any(code in batch_codes for code in codes):
        return "batch"
    if any(code in sample_codes for code in codes):
        return "sample"
    return None


def _save_observation(index: int, record: MolecularRecord, context: StageContext) -> str:
    path = context.stage_directory / "observations" / f"{index:04d}.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_payload(path, MolecularPayload((record,)))
    return path.relative_to(context.stage_directory).as_posix()
