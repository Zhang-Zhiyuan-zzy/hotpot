"""Private tagged JSON codecs for scientific-state artifacts."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple, Type

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    CandidateProvenance,
    CoverageStatus,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    CoordinationFragmentAssignment,
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from .contracts import DerivedStateValue, StateCandidate

__all__ = []


_SUPPORTED_STATE_CANDIDATE_TYPES: Tuple[Type[object], ...] = (
    NetChargeCandidateEvidence,
    LewisFormalChargeCandidate,
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
    SpinContributionCandidate,
    ElectronicStateCandidate,
    DerivedStateValue,
)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError("Serialized value must be a string-keyed object")
    return value


def _require_keys(
    payload: Mapping[str, object],
    required_keys: Tuple[str, ...],
) -> None:
    actual_keys = frozenset(payload)
    expected_keys = frozenset(required_keys)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Serialized object schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _sequence(value: object) -> Tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError("Serialized tuple value must be a list")
    return tuple(value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Serialized value must be a string")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Serialized value must be a boolean")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Serialized value must be an integer")
    return value


def _optional_integer(value: object) -> Optional[int]:
    return None if value is None else _integer(value)


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("Serialized value must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Serialized numeric value must be finite")
    return number


def _strings(value: object) -> Tuple[str, ...]:
    return tuple(_string(item) for item in _sequence(value))


def _integers(value: object) -> Tuple[int, ...]:
    return tuple(_integer(item) for item in _sequence(value))


def _optional_integers(value: object) -> Tuple[Optional[int], ...]:
    return tuple(_optional_integer(item) for item in _sequence(value))


def _provenance_to_payload(
    provenance: CandidateProvenance,
) -> Dict[str, object]:
    return {
        "identity": provenance.identity,
        "authority": provenance.authority.value,
        "source_identity": provenance.source_identity,
        "evidence_ids": list(provenance.evidence_ids),
        "assumption_ids": list(provenance.assumption_ids),
    }


def _provenance_from_payload(value: object) -> CandidateProvenance:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "identity",
            "authority",
            "source_identity",
            "evidence_ids",
            "assumption_ids",
        ),
    )
    return CandidateProvenance(
        identity=_string(payload["identity"]),
        authority=StateValueAuthority(_string(payload["authority"])),
        source_identity=_string(payload["source_identity"]),
        evidence_ids=_strings(payload["evidence_ids"]),
        assumption_ids=_strings(payload["assumption_ids"]),
    )


def _component_charges_to_payload(
    component_charges: Tuple[Tuple[str, int], ...],
) -> list[object]:
    return [
        [component_identity, charge]
        for component_identity, charge in component_charges
    ]


def _component_charges_from_payload(
    value: object,
) -> Tuple[Tuple[str, int], ...]:
    charges = []
    for item in _sequence(value):
        pair = _sequence(item)
        if len(pair) != 2:
            raise ValueError("Component charge must contain identity and value")
        charges.append((_string(pair[0]), _integer(pair[1])))
    return tuple(charges)


def _coordination_fragment_from_payload(
    value: object,
) -> CoordinationFragmentAssignment:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "component_identity",
            "ionic_bookkeeping_charge",
            "lxz_class",
            "donated_electrons",
        ),
    )
    lxz_class = payload["lxz_class"]
    return CoordinationFragmentAssignment(
        component_identity=_string(payload["component_identity"]),
        ionic_bookkeeping_charge=_optional_integer(
            payload["ionic_bookkeeping_charge"]
        ),
        lxz_class=None if lxz_class is None else _string(lxz_class),
        donated_electrons=_optional_integer(payload["donated_electrons"]),
    )


def _state_candidate_to_payload(
    candidate: StateCandidate,
) -> Dict[str, object]:
    common = {"provenance": _provenance_to_payload(candidate.provenance)}
    if isinstance(candidate, NetChargeCandidateEvidence):
        return {"type": "net_charge", **common, "charge": candidate.charge}
    if isinstance(candidate, LewisFormalChargeCandidate):
        return {
            "type": "lewis_formal_charge",
            **common,
            "net_charge_candidate_identity": (
                candidate.net_charge_candidate_identity
            ),
            "atom_formal_charges": list(candidate.atom_formal_charges),
            "bond_orders": list(candidate.bond_orders),
            "nonbonding_electrons": list(candidate.nonbonding_electrons),
            "radical_electrons": list(candidate.radical_electrons),
            "component_charges": _component_charges_to_payload(
                candidate.component_charges
            ),
        }
    if isinstance(candidate, CoordinationCountingCandidate):
        return {
            "type": "coordination_counting",
            **common,
            "convention": candidate.convention.value,
            "fragments": [
                {
                    "component_identity": fragment.component_identity,
                    "ionic_bookkeeping_charge": (
                        fragment.ionic_bookkeeping_charge
                    ),
                    "lxz_class": fragment.lxz_class,
                    "donated_electrons": fragment.donated_electrons,
                }
                for fragment in candidate.fragments
            ],
        }
    if isinstance(candidate, OxidationStateCandidateEvidence):
        return {
            "type": "oxidation_state",
            **common,
            "convention": candidate.convention.value,
            "atom_states": list(candidate.atom_states),
            "coverage": candidate.coverage.value,
            "coordination_counting_candidate_identity": (
                candidate.coordination_counting_candidate_identity
            ),
        }
    if isinstance(candidate, SpinContributionCandidate):
        return {
            "type": "spin_contribution",
            **common,
            "scope": candidate.scope.value,
            "subject_identity": candidate.subject_identity,
            "parent_candidate_ids": list(candidate.parent_candidate_ids),
            "two_s": candidate.two_s,
            "pure_spin": candidate.pure_spin,
        }
    if isinstance(candidate, ElectronicStateCandidate):
        return {
            "type": "electronic_state",
            **common,
            "net_charge_candidate_identity": (
                candidate.net_charge_candidate_identity
            ),
            "oxidation_state_candidate_identity": (
                candidate.oxidation_state_candidate_identity
            ),
            "electron_count": candidate.electron_count,
            "two_s": candidate.two_s,
            "multiplicity": candidate.multiplicity,
            "pure_spin": candidate.pure_spin,
        }
    if isinstance(candidate, DerivedStateValue):
        return {
            "type": "derived_state_value",
            **common,
            "quantity": candidate.quantity.value,
            "value": candidate.value,
            "equation_id": candidate.equation_id,
            "parent_candidate_ids": list(candidate.parent_candidate_ids),
            "counting_convention": candidate.counting_convention.value,
        }
    raise TypeError("Unsupported scientific-state candidate type")


def _state_candidate_from_payload(value: object) -> StateCandidate:
    payload = _mapping(value)
    if "type" not in payload:
        raise ValueError("Serialized scientific-state candidate has no type")
    candidate_type = _string(payload["type"])
    if candidate_type == "net_charge":
        _require_keys(payload, ("type", "provenance", "charge"))
        return NetChargeCandidateEvidence(
            _provenance_from_payload(payload["provenance"]),
            _integer(payload["charge"]),
        )
    if candidate_type == "lewis_formal_charge":
        _require_keys(
            payload,
            (
                "type",
                "provenance",
                "net_charge_candidate_identity",
                "atom_formal_charges",
                "bond_orders",
                "nonbonding_electrons",
                "radical_electrons",
                "component_charges",
            ),
        )
        return LewisFormalChargeCandidate(
            provenance=_provenance_from_payload(payload["provenance"]),
            net_charge_candidate_identity=_string(
                payload["net_charge_candidate_identity"]
            ),
            atom_formal_charges=_integers(payload["atom_formal_charges"]),
            bond_orders=tuple(
                _number(item) for item in _sequence(payload["bond_orders"])
            ),
            nonbonding_electrons=_integers(
                payload["nonbonding_electrons"]
            ),
            radical_electrons=_integers(payload["radical_electrons"]),
            component_charges=_component_charges_from_payload(
                payload["component_charges"]
            ),
        )
    if candidate_type == "coordination_counting":
        _require_keys(
            payload,
            ("type", "provenance", "convention", "fragments"),
        )
        fragments = tuple(
            _coordination_fragment_from_payload(item)
            for item in _sequence(payload["fragments"])
        )
        return CoordinationCountingCandidate(
            _provenance_from_payload(payload["provenance"]),
            CountingConvention(_string(payload["convention"])),
            fragments,
        )
    if candidate_type == "oxidation_state":
        _require_keys(
            payload,
            (
                "type",
                "provenance",
                "convention",
                "atom_states",
                "coverage",
                "coordination_counting_candidate_identity",
            ),
        )
        counting_identity = payload[
            "coordination_counting_candidate_identity"
        ]
        return OxidationStateCandidateEvidence(
            provenance=_provenance_from_payload(payload["provenance"]),
            convention=CountingConvention(_string(payload["convention"])),
            atom_states=_optional_integers(payload["atom_states"]),
            coverage=CoverageStatus(_string(payload["coverage"])),
            coordination_counting_candidate_identity=(
                None
                if counting_identity is None
                else _string(counting_identity)
            ),
        )
    if candidate_type == "spin_contribution":
        _require_keys(
            payload,
            (
                "type",
                "provenance",
                "scope",
                "subject_identity",
                "parent_candidate_ids",
                "two_s",
                "pure_spin",
            ),
        )
        return SpinContributionCandidate(
            provenance=_provenance_from_payload(payload["provenance"]),
            scope=SpinContributionScope(_string(payload["scope"])),
            subject_identity=_string(payload["subject_identity"]),
            parent_candidate_ids=_strings(payload["parent_candidate_ids"]),
            two_s=_integer(payload["two_s"]),
            pure_spin=_boolean(payload["pure_spin"]),
        )
    if candidate_type == "electronic_state":
        _require_keys(
            payload,
            (
                "type",
                "provenance",
                "net_charge_candidate_identity",
                "oxidation_state_candidate_identity",
                "electron_count",
                "two_s",
                "multiplicity",
                "pure_spin",
            ),
        )
        oxidation_identity = payload["oxidation_state_candidate_identity"]
        return ElectronicStateCandidate(
            provenance=_provenance_from_payload(payload["provenance"]),
            net_charge_candidate_identity=_string(
                payload["net_charge_candidate_identity"]
            ),
            oxidation_state_candidate_identity=(
                None
                if oxidation_identity is None
                else _string(oxidation_identity)
            ),
            electron_count=_integer(payload["electron_count"]),
            two_s=_integer(payload["two_s"]),
            multiplicity=_optional_integer(payload["multiplicity"]),
            pure_spin=_boolean(payload["pure_spin"]),
        )
    if candidate_type == "derived_state_value":
        _require_keys(
            payload,
            (
                "type",
                "provenance",
                "quantity",
                "value",
                "equation_id",
                "parent_candidate_ids",
                "counting_convention",
            ),
        )
        return DerivedStateValue(
            provenance=_provenance_from_payload(payload["provenance"]),
            quantity=ScientificQuantity(_string(payload["quantity"])),
            value=_integer(payload["value"]),
            equation_id=_string(payload["equation_id"]),
            parent_candidate_ids=_strings(payload["parent_candidate_ids"]),
            counting_convention=CountingConvention(
                _string(payload["counting_convention"])
            ),
        )
    raise ValueError(f"Unknown scientific-state candidate type {candidate_type!r}")
