"""Immutable Gate policy configurations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Union

from ._base import (
    _ImmutableGate,
    _checks_info,
    _levels_info,
    _normalize_checks,
    _normalize_levels,
    _normalize_tolerances,
    _tolerances_info,
)
from ._checker import CHECKER_PROTOCOL_DIGEST, Checker, _CheckerDefinition


_EVIDENCE_BUILTINS = frozenset(
    {
        "extracted_material",
        "explicit_report",
        "usable_locator",
        "primary_source_verified",
        "independent_corroboration",
    }
)
_EVIDENCE_DEFAULT_CHECKS = {
    "extracted_material": True,
    "explicit_report": True,
    "usable_locator": True,
    "primary_source_verified": True,
    "independent_corroboration": True,
}
_EVIDENCE_DEFAULT_LEVELS = {
    "E0": (),
    "E1": ("extracted_material",),
    "E2": ("extracted_material", "explicit_report", "usable_locator"),
    "E3": (
        "extracted_material",
        "explicit_report",
        "usable_locator",
        "primary_source_verified",
    ),
    "E4": (
        "extracted_material",
        "explicit_report",
        "usable_locator",
        "primary_source_verified",
        "independent_corroboration",
    ),
}

_QUALITY_BUILTINS = frozenset(
    {
        "governed_collection",
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
        "primary_source_supported",
        "validation_controls",
        "independent_corroboration",
        "reproducible_policy",
    }
)
_QUALITY_DEFAULT_CHECKS = {name: True for name in _QUALITY_BUILTINS}
_QUALITY_STRUCTURE_BUILTINS = frozenset(
    {"asset_available", "asset_hash_valid", "asset_parseable", "declared_role_valid"}
)
_QUALITY_DEFAULT_STRUCTURE_CHECKS = {
    name: True for name in _QUALITY_STRUCTURE_BUILTINS
}
_QUALITY_DEFAULT_LEVELS = {
    "B0": (),
    "B1": ("provenance_complete", "subject_mapping_audited"),
    "B2": (
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
    ),
    "B3": (
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
        "primary_source_supported",
        "validation_controls",
    ),
    "B4": (
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
        "primary_source_supported",
        "validation_controls",
        "independent_corroboration",
        "reproducible_policy",
    ),
}

_BENCHMARK_CHECKLIST_BUILTINS = frozenset(
    {"verified_primary_source", "conflict_free"}
)
_BENCHMARK_DIRECT_BUILTINS = frozenset({"hotpot_or_user_attestation"})
_BENCHMARK_VETO_BUILTINS = frozenset(
    {"conflict_free", "attestation_not_rejected"}
)


class EvidenceGate(_ImmutableGate):
    __slots__ = ("_checks", "_levels", "_require_locator")

    def __init__(
        self,
        checks: Optional[Mapping[str, Union[bool, Checker]]] = None,
        *,
        level_checks: Optional[Mapping[str, Iterable[str]]] = None,
        require_locator: bool = True,
    ):
        if type(require_locator) is not bool:
            raise TypeError("require_locator must be bool")
        check_values = dict(_EVIDENCE_DEFAULT_CHECKS)
        check_values["usable_locator"] = require_locator
        checks_normalized = _normalize_checks(
            checks,
            check_values,
            _EVIDENCE_BUILTINS,
            merge_defaults=True,
        )
        default_levels = dict(_EVIDENCE_DEFAULT_LEVELS)
        if not require_locator:
            default_levels = {
                level: tuple(check for check in names if check != "usable_locator")
                for level, names in default_levels.items()
            }
        levels = _normalize_levels(
            level_checks,
            default_levels,
            ("E0", "E1", "E2", "E3", "E4"),
            (name for name, _ in checks_normalized),
        )
        object.__setattr__(self, "_checks", checks_normalized)
        object.__setattr__(self, "_levels", levels)
        object.__setattr__(self, "_require_locator", require_locator)
        self._finish(
            {
                "gate": "EvidenceGate",
                "checks": _checks_info(checks_normalized),
                "level_checks": _levels_info(levels),
                "require_locator": require_locator,
                "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
            }
        )

    @classmethod
    def _from_info(cls, info: Mapping[str, Any]) -> "EvidenceGate":
        _require_gate_fields(
            info,
            {
                "gate",
                "checks",
                "level_checks",
                "require_locator",
                "checker_protocol_digest",
            },
            "EvidenceGate",
        )
        return _restore_gate(cls, info)


class QualityGate(_ImmutableGate):
    __slots__ = ("_checks", "_levels", "_structure_checks")

    def __init__(
        self,
        checks: Optional[Mapping[str, Union[bool, Checker]]] = None,
        *,
        level_checks: Optional[Mapping[str, Iterable[str]]] = None,
        structure_checks: Optional[Mapping[str, Union[bool, Checker]]] = None,
    ):
        checks_normalized = _normalize_checks(
            checks,
            _QUALITY_DEFAULT_CHECKS,
            _QUALITY_BUILTINS,
            merge_defaults=True,
        )
        structures_normalized = _normalize_checks(
            structure_checks,
            _QUALITY_DEFAULT_STRUCTURE_CHECKS,
            _QUALITY_STRUCTURE_BUILTINS,
            merge_defaults=True,
        )
        levels = _normalize_levels(
            level_checks,
            _QUALITY_DEFAULT_LEVELS,
            ("B0", "B1", "B2", "B3", "B4"),
            (name for name, _ in checks_normalized),
        )
        object.__setattr__(self, "_checks", checks_normalized)
        object.__setattr__(self, "_levels", levels)
        object.__setattr__(self, "_structure_checks", structures_normalized)
        self._finish(
            {
                "gate": "QualityGate",
                "checks": _checks_info(checks_normalized),
                "level_checks": _levels_info(levels),
                "structure_checks": _checks_info(structures_normalized),
                "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
            }
        )

    @classmethod
    def _from_info(cls, info: Mapping[str, Any]) -> "QualityGate":
        _require_gate_fields(
            info,
            {
                "gate",
                "checks",
                "level_checks",
                "structure_checks",
                "checker_protocol_digest",
            },
            "QualityGate",
        )
        return _restore_gate(cls, info)


class BenchmarkGate(_ImmutableGate):
    __slots__ = ("_checklist", "_direct_entry", "_vetoes", "_tolerances")

    def __init__(
        self,
        checklist: Optional[Mapping[str, Union[bool, Checker]]] = None,
        direct_entry: Optional[Mapping[str, Union[bool, Checker]]] = None,
        *,
        vetoes: Optional[Iterable[str]] = None,
        numeric_tolerances: Optional[Mapping[str, Mapping[str, float]]] = None,
    ):
        checklist_normalized = _normalize_checks(
            checklist,
            {"verified_primary_source": True, "conflict_free": True},
            _BENCHMARK_CHECKLIST_BUILTINS,
            merge_defaults=False,
        )
        direct_normalized = _normalize_checks(
            direct_entry,
            {"hotpot_or_user_attestation": True},
            _BENCHMARK_DIRECT_BUILTINS,
            merge_defaults=False,
        )
        veto_values = (
            ("conflict_free", "attestation_not_rejected")
            if vetoes is None
            else tuple(vetoes)
        )
        if any(not isinstance(name, str) or not name for name in veto_values):
            raise ValueError("veto names must be non-empty strings")
        if len(set(veto_values)) != len(veto_values):
            raise ValueError("veto names must be unique")
        configured = {
            name for name, _ in checklist_normalized + direct_normalized
        }
        unknown_vetoes = set(veto_values) - configured - _BENCHMARK_VETO_BUILTINS
        if unknown_vetoes:
            raise ValueError(
                f"unknown benchmark vetoes: {', '.join(sorted(unknown_vetoes))}"
            )
        tolerances = _normalize_tolerances(numeric_tolerances)
        object.__setattr__(self, "_checklist", checklist_normalized)
        object.__setattr__(self, "_direct_entry", direct_normalized)
        object.__setattr__(self, "_vetoes", tuple(sorted(veto_values)))
        object.__setattr__(self, "_tolerances", tolerances)
        self._finish(
            {
                "gate": "BenchmarkGate",
                "checklist": _checks_info(checklist_normalized),
                "direct_entry": _checks_info(direct_normalized),
                "vetoes": sorted(veto_values),
                "numeric_tolerances": _tolerances_info(tolerances),
                "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
            }
        )

    @classmethod
    def _from_info(cls, info: Mapping[str, Any]) -> "BenchmarkGate":
        _require_gate_fields(
            info,
            {
                "gate",
                "checklist",
                "direct_entry",
                "vetoes",
                "numeric_tolerances",
                "checker_protocol_digest",
            },
            "BenchmarkGate",
        )
        return _restore_gate(cls, info)


def _require_gate_fields(
    info: Mapping[str, Any], expected: set, gate_name: str
) -> None:
    if set(info) != expected or info.get("gate") != gate_name:
        raise ValueError(f"invalid {gate_name} configuration")
    if info.get("checker_protocol_digest") != CHECKER_PROTOCOL_DIGEST:
        raise ValueError("Gate checker protocol digest does not match")


def _restore_checks(value: Any) -> tuple:
    if not isinstance(value, Mapping):
        raise TypeError("serialized checks must be a mapping")
    restored = []
    for name, definition in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checker names must be non-empty strings")
        if not isinstance(definition, Mapping):
            raise TypeError("serialized checker definitions must be mappings")
        restored.append((name, _CheckerDefinition.from_info(definition)))
    return tuple(sorted(restored, key=lambda item: item[0]))


def _validate_restored_builtins(checks: tuple, allowed: frozenset) -> None:
    unknown = {
        name
        for name, definition in checks
        if definition.kind == "builtin" and name not in allowed
    }
    if unknown:
        raise ValueError(
            f"unknown built-in checker: {', '.join(sorted(unknown))}"
        )


def _restore_gate(cls: type, info: Mapping[str, Any]) -> _ImmutableGate:
    instance = cls.__new__(cls)
    if cls is EvidenceGate:
        checks = _restore_checks(info["checks"])
        _validate_restored_builtins(checks, _EVIDENCE_BUILTINS)
        if type(info["require_locator"]) is not bool:
            raise TypeError("require_locator must be bool")
        levels = _normalize_levels(
            info["level_checks"],
            {},
            ("E0", "E1", "E2", "E3", "E4"),
            (name for name, _ in checks),
        )
        object.__setattr__(instance, "_checks", checks)
        object.__setattr__(instance, "_levels", levels)
        object.__setattr__(instance, "_require_locator", info["require_locator"])
        canonical = {
            "gate": "EvidenceGate",
            "checks": _checks_info(checks),
            "level_checks": _levels_info(levels),
            "require_locator": info["require_locator"],
            "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
        }
    elif cls is QualityGate:
        checks = _restore_checks(info["checks"])
        structures = _restore_checks(info["structure_checks"])
        _validate_restored_builtins(checks, _QUALITY_BUILTINS)
        _validate_restored_builtins(structures, _QUALITY_STRUCTURE_BUILTINS)
        levels = _normalize_levels(
            info["level_checks"],
            {},
            ("B0", "B1", "B2", "B3", "B4"),
            (name for name, _ in checks),
        )
        object.__setattr__(instance, "_checks", checks)
        object.__setattr__(
            instance,
            "_levels",
            levels,
        )
        object.__setattr__(
            instance,
            "_structure_checks",
            structures,
        )
        canonical = {
            "gate": "QualityGate",
            "checks": _checks_info(checks),
            "level_checks": _levels_info(levels),
            "structure_checks": _checks_info(structures),
            "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
        }
    else:
        checklist = _restore_checks(info["checklist"])
        direct = _restore_checks(info["direct_entry"])
        _validate_restored_builtins(checklist, _BENCHMARK_CHECKLIST_BUILTINS)
        _validate_restored_builtins(direct, _BENCHMARK_DIRECT_BUILTINS)
        vetoes = info["vetoes"]
        if (
            not isinstance(vetoes, list)
            or any(not isinstance(name, str) or not name for name in vetoes)
            or len(set(vetoes)) != len(vetoes)
        ):
            raise ValueError("serialized vetoes must be unique non-empty strings")
        configured = {name for name, _ in checklist + direct}
        unknown_vetoes = set(vetoes) - configured - _BENCHMARK_VETO_BUILTINS
        if unknown_vetoes:
            raise ValueError(
                f"unknown benchmark vetoes: {', '.join(sorted(unknown_vetoes))}"
            )
        tolerances = _normalize_tolerances(info["numeric_tolerances"])
        object.__setattr__(instance, "_checklist", checklist)
        object.__setattr__(
            instance, "_direct_entry", direct
        )
        object.__setattr__(instance, "_vetoes", tuple(sorted(vetoes)))
        object.__setattr__(
            instance,
            "_tolerances",
            tolerances,
        )
        canonical = {
            "gate": "BenchmarkGate",
            "checklist": _checks_info(checklist),
            "direct_entry": _checks_info(direct),
            "vetoes": sorted(vetoes),
            "numeric_tolerances": _tolerances_info(tolerances),
            "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
        }
    instance._finish(canonical)
    return instance
