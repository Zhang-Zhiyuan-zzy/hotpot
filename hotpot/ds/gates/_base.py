"""Shared immutable configuration machinery for Gates."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from ._checker import Checker, _CheckerDefinition


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class _ImmutableGate:
    __slots__ = ("_digest", "_info")

    def _finish(self, info: Mapping[str, Any]) -> None:
        encoded = _canonical_bytes(dict(info))
        object.__setattr__(self, "_info", encoded.decode("utf-8"))
        object.__setattr__(self, "_digest", hashlib.sha256(encoded).hexdigest())

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    @property
    def info(self) -> dict:
        return json.loads(self._info)

    @property
    def digest(self) -> str:
        return self._digest


def _normalize_checks(
    supplied: Optional[Mapping[str, Any]],
    defaults: Mapping[str, bool],
    builtins: Iterable[str],
    *,
    merge_defaults: bool,
) -> Tuple[Tuple[str, _CheckerDefinition], ...]:
    values = dict(defaults) if supplied is None or merge_defaults else {}
    if supplied is not None:
        values.update(dict(supplied))
    builtin_names = frozenset(builtins)
    normalized = []
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("checker names must be non-empty strings")
        if type(value) is bool:
            if name not in builtin_names:
                raise ValueError(f"unknown built-in checker: {name}")
            definition = _CheckerDefinition.builtin(value)
        elif callable(value):
            definition = _CheckerDefinition.custom(value)
        else:
            raise TypeError("checker mappings accept only bool or a custom checker")
        normalized.append((name, definition))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _checks_info(checks: Sequence[Tuple[str, _CheckerDefinition]]) -> dict:
    return {name: dict(definition.info()) for name, definition in checks}


def _normalize_levels(
    supplied: Optional[Mapping[str, Iterable[str]]],
    defaults: Mapping[str, Iterable[str]],
    levels: Sequence[str],
    available_checks: Iterable[str],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    values = {level: tuple(checks) for level, checks in defaults.items()}
    if supplied is not None:
        for level, checks in supplied.items():
            if level not in levels:
                raise ValueError(f"unknown assessment level: {level}")
            values[level] = tuple(checks)
    if set(values) != set(levels):
        raise ValueError("every assessment level must have a check definition")
    available = frozenset(available_checks)
    normalized = []
    previous = frozenset()
    for level in levels:
        checks = values[level]
        if any(not isinstance(check, str) or not check for check in checks):
            raise ValueError("level check names must be non-empty strings")
        if len(set(checks)) != len(checks):
            raise ValueError("level check names must be unique")
        unknown = set(checks) - available
        if unknown:
            raise ValueError(
                f"level {level} refers to unknown checks: {', '.join(sorted(unknown))}"
            )
        current = frozenset(checks)
        if not previous.issubset(current):
            raise ValueError("assessment level requirements must be cumulative")
        previous = current
        normalized.append((level, tuple(sorted(current))))
    return tuple(normalized)


def _levels_info(levels: Sequence[Tuple[str, Sequence[str]]]) -> dict:
    return {level: list(checks) for level, checks in levels}


def _normalize_tolerances(
    tolerances: Optional[Mapping[str, Mapping[str, float]]]
) -> Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...]:
    normalized = []
    for property_name, specification in (tolerances or {}).items():
        if not isinstance(property_name, str) or not property_name.strip():
            raise ValueError("numeric tolerance property names must be non-empty strings")
        if not isinstance(specification, Mapping):
            raise TypeError("a numeric tolerance must be a mapping")
        unknown = set(specification) - {"absolute", "relative"}
        if unknown or not specification:
            raise ValueError("numeric tolerances accept only absolute and relative")
        result = {}
        for name in ("absolute", "relative"):
            value = specification.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("numeric tolerances must be numbers")
            normalized_value = float(value)
            if normalized_value < 0.0 or not math.isfinite(normalized_value):
                raise ValueError("numeric tolerances must be finite and non-negative")
            result[name] = normalized_value
        normalized.append((property_name, tuple(result.items())))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _tolerances_info(
    tolerances: Sequence[Tuple[str, Sequence[Tuple[str, float]]]]
) -> dict:
    return {
        property_name: dict(specification)
        for property_name, specification in tolerances
    }
