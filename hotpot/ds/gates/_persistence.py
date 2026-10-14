"""Private persistence helpers for current Gate policy configurations."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Tuple

from ._gates import BenchmarkGate, EvidenceGate, QualityGate


_GATE_ORDER = {EvidenceGate: 0, QualityGate: 1, BenchmarkGate: 2}
_GATE_BY_NAME = {
    "EvidenceGate": EvidenceGate,
    "QualityGate": QualityGate,
    "BenchmarkGate": BenchmarkGate,
}


def default_gates() -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]:
    return EvidenceGate(), QualityGate(), BenchmarkGate()


def serialize_gate_configs(gates: Iterable[object]) -> list:
    normalized = _normalize_gate_set(gates)
    return [gate.info for gate in normalized]


def deserialize_gate_configs(
    configurations: Iterable[Mapping[str, object]],
) -> Tuple[EvidenceGate, QualityGate, BenchmarkGate]:
    gates = []
    for configuration in configurations:
        gate_name = configuration.get("gate")
        gate_type = _GATE_BY_NAME.get(gate_name)
        if gate_type is None:
            raise ValueError(f"unknown Gate configuration: {gate_name}")
        gates.append(gate_type._from_info(configuration))
    return _normalize_gate_set(gates)


def materialize_checker_sources(
    gates: Iterable[object], bundle_root: Path
) -> Tuple[Path, ...]:
    checkers_root = Path(bundle_root) / "policy" / "checkers"
    sources = {}
    for gate in _normalize_gate_set(gates):
        for definition in _checker_definitions(gate):
            if definition.kind == "custom":
                sources[definition.source_sha256] = definition.source
    if not sources:
        return ()
    checkers_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for digest, source in sorted(sources.items()):
        destination = checkers_root / f"{digest}.py"
        encoded = source.encode("utf-8")
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise ValueError("content-addressed checker path contains different bytes")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=checkers_root
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        paths.append(destination)
    return tuple(paths)


def _normalize_gate_set(gates: Iterable[object]) -> tuple:
    by_type = {}
    for gate in gates:
        gate_type = type(gate)
        if gate_type not in _GATE_ORDER:
            raise TypeError("a Gate set accepts only current Gate types")
        if gate_type in by_type:
            raise ValueError(f"duplicate Gate type: {gate_type.__name__}")
        by_type[gate_type] = gate
    if set(by_type) != set(_GATE_ORDER):
        raise ValueError("a persisted Gate set must contain all three Gate types")
    return tuple(by_type[gate_type] for gate_type in _GATE_ORDER)


def _checker_definitions(gate: object) -> tuple:
    if isinstance(gate, EvidenceGate):
        return tuple(definition for _, definition in gate._checks)
    if isinstance(gate, QualityGate):
        return tuple(
            definition
            for _, definition in gate._checks + gate._structure_checks
        )
    return tuple(
        definition
        for _, definition in gate._checklist + gate._direct_entry
    )
