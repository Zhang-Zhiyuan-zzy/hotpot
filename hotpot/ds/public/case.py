"""Bundle-owned scientific cases and read-only case collections.

These types are intentionally returned by :class:`hotpot.ds.Bundle` rather
than exported from the first-level public namespace.  A Case never owns or
persists state: every read and mutation goes through its sole Bundle owner.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union


_EVIDENCE_DETAIL_FIELDS = {
    "doi",
    "url",
    "scientific_work_id",
    "source_lineage_id",
    "derived_from",
}
_VERIFICATION_DETAIL_FIELDS = {"procedure_version", "verified_at"}
_VERIFICATION_PROCEDURE = "hotpot.ds.evidence-verification/current"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _evidence_details(
    source: str, details: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    values = deepcopy(dict(details or {}))
    unsupported = values.keys() - _EVIDENCE_DETAIL_FIELDS
    if unsupported:
        raise ValueError(
            "unsupported evidence detail fields: " + ", ".join(sorted(unsupported))
        )
    for field in _EVIDENCE_DETAIL_FIELDS - {"derived_from"}:
        if field in values and (
            not isinstance(values[field], str) or not values[field].strip()
        ):
            raise ValueError(f"evidence details.{field} must be a non-empty string")
    if "derived_from" in values:
        parents = values["derived_from"]
        if isinstance(parents, str) or not isinstance(parents, Sequence):
            raise ValueError("evidence details.derived_from must be a sequence")
        if any(not isinstance(item, str) or not item.strip() for item in parents):
            raise ValueError(
                "evidence details.derived_from must contain non-empty strings"
            )
        if len(parents) != len(set(parents)):
            raise ValueError("evidence details.derived_from cannot contain duplicates")
        values["derived_from"] = list(parents)

    lowered = source.casefold()
    inferred: Dict[str, str] = {}
    if lowered.startswith("doi:"):
        doi = source[4:].strip()
        if not doi:
            raise ValueError("doi evidence source must include an identifier")
        inferred = {"doi": doi, "scientific_work_id": f"doi:{doi.casefold()}"}
    elif lowered.startswith("url:"):
        url = source[4:].strip()
        if not url:
            raise ValueError("url evidence source must include a URL")
        inferred = {"url": url}
    elif lowered.startswith(("http://", "https://")):
        inferred = {"url": source}
        doi_marker = "doi.org/"
        marker_index = lowered.find(doi_marker)
        if marker_index >= 0:
            doi = source[marker_index + len(doi_marker) :].strip()
            inferred.update(
                {"doi": doi, "scientific_work_id": f"doi:{doi.casefold()}"}
            )
    elif lowered.startswith("file:"):
        path = source[5:].strip()
        if not path:
            raise ValueError("file evidence source must include a Bundle-relative path")
        inferred = {"path": path}
    for field, value in inferred.items():
        supplied = values.get(field)
        if supplied is not None and supplied.casefold() != value.casefold():
            raise ValueError(
                f"evidence details.{field} conflicts with the source reference"
            )
        values.setdefault(field, value)
    return values


def _verification_details(
    details: Optional[Mapping[str, Any]],
) -> Dict[str, str]:
    values = deepcopy(dict(details or {}))
    unsupported = values.keys() - _VERIFICATION_DETAIL_FIELDS
    if unsupported:
        raise ValueError(
            "unsupported verification detail fields: "
            + ", ".join(sorted(unsupported))
        )
    for field, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"verification details.{field} must be a non-empty string"
            )
    if "verified_at" in values:
        timestamp = values["verified_at"].replace("Z", "+00:00")
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            raise ValueError("verification details.verified_at must include a timezone")
    values.setdefault("procedure_version", _VERIFICATION_PROCEDURE)
    values.setdefault("verified_at", _utc_now())
    return values


class CaseOwnershipError(RuntimeError):
    """A Case operation attempted to escape its owning Bundle."""


class UnsynchronizedBundleError(RuntimeError):
    """A derived result was requested before explicit synchronization."""


class Case:
    """One independently assessable, permanently Bundle-owned record."""

    __slots__ = ("__owner", "__case_id")

    def __init__(self, owner: Any, case_id: str) -> None:
        object.__setattr__(self, "_Case__owner", owner)
        object.__setattr__(self, "_Case__case_id", case_id)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_Case__owner", "_Case__case_id"} and hasattr(self, name):
            raise CaseOwnershipError("Case ownership and identity are immutable")
        object.__setattr__(self, name, value)

    @property
    def id(self) -> str:
        return self.__case_id

    @property
    def _owner(self) -> Any:
        return self.__owner

    def _facts(self) -> Dict[str, Any]:
        return deepcopy(self.__owner._case_payload(self.__case_id))

    def _derived(self) -> Dict[str, Any]:
        return deepcopy(self.__owner._case_assessment(self.__case_id))

    def _mutate(
        self,
        action: str,
        mutation: Callable[
            [Dict[str, Any]], tuple[Mapping[str, Any], Any]
        ],
    ) -> Any:
        return self.__owner._mutate_case(self.__case_id, action, mutation)

    @property
    def info(self) -> Dict[str, Any]:
        value = self._facts()
        value["id"] = self.id
        if self.__owner.is_synchronized:
            value["assessment"] = self._derived()
        return value

    @property
    def core_info(self) -> Dict[str, Any]:
        return deepcopy(self._facts().get("core_info", {}))

    @property
    def structures(self) -> tuple[Dict[str, Any], ...]:
        return tuple(deepcopy(self._facts().get("structures", [])))

    @property
    def labels(self) -> Dict[str, Any]:
        facts = self._facts()
        labels = facts.get("labels")
        if isinstance(labels, dict):
            return deepcopy(labels)
        return {
            item["property"]: deepcopy(item.get("value"))
            for item in facts.get("observations", [])
            if item.get("status", "active") == "active" and "property" in item
        }

    @property
    def observations(self) -> tuple[Dict[str, Any], ...]:
        return tuple(deepcopy(self._facts().get("observations", [])))

    @property
    def claims(self) -> tuple[Dict[str, Any], ...]:
        return tuple(deepcopy(self._facts().get("claims", [])))

    @property
    def evidence(self) -> tuple[Dict[str, Any], ...]:
        return tuple(deepcopy(self._facts().get("evidence", [])))

    @property
    def attestations(self) -> tuple[Dict[str, Any], ...]:
        return tuple(deepcopy(self._facts().get("attestations", [])))

    @property
    def evidence_levels(self) -> Dict[str, str]:
        return deepcopy(self._derived().get("evidence_levels", {}))

    @property
    def evidence_level(self) -> Optional[str]:
        return self._derived().get("evidence_level")

    @property
    def quality_level(self) -> str:
        return str(self._derived()["quality_level"])

    @property
    def archive_class(self) -> str:
        return str(self._derived()["archive_class"])

    @property
    def benchmark(self) -> bool:
        return bool(self._derived()["benchmark"])

    @property
    def benchmark_status(self) -> str:
        return str(self._derived()["benchmark_status"])

    @property
    def reasons(self) -> list[Dict[str, Any]]:
        return deepcopy(self._derived().get("reasons", []))

    @property
    def assessment(self) -> Dict[str, Any]:
        return self._derived()

    def set_core_info(
        self, name: str, value: Any, *, reason: Optional[str] = None
    ) -> None:
        replacement = deepcopy(value)

        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], None]:
            core_info = dict(facts.get("core_info", {}))
            before = deepcopy(core_info.get(name))
            core_info[name] = deepcopy(replacement)
            facts["core_info"] = core_info
            return {
                "name": name,
                "before": before,
                "after": replacement,
                "reason": reason,
            }, None

        self._mutate("core_info.set", mutate)

    def remove_core_info(self, name: str, *, reason: str) -> None:
        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], None]:
            core_info = dict(facts.get("core_info", {}))
            before = core_info.pop(name)
            facts["core_info"] = core_info
            return {"name": name, "before": before, "reason": reason}, None

        self._mutate("core_info.remove", mutate)

    def add_structure(self, data: Mapping[str, Any]) -> str:
        return self._add("structures", "structure_id", data, "structure.add")

    def replace_structure(
        self,
        structure_id: str,
        data: Mapping[str, Any],
        *,
        reason: str,
    ) -> str:
        return self._replace(
            "structures", "structure_id", structure_id, data, "structure.replace", reason
        )

    def withdraw_structure(self, structure_id: str, *, reason: str) -> None:
        self._withdraw(
            "structures", "structure_id", structure_id, "structure.withdraw", reason
        )

    def add_observation(self, data: Mapping[str, Any]) -> str:
        return self._add("observations", "observation_id", data, "observation.add")

    def replace_observation(
        self,
        observation_id: str,
        data: Mapping[str, Any],
        *,
        reason: str,
    ) -> str:
        return self._replace(
            "observations",
            "observation_id",
            observation_id,
            data,
            "observation.replace",
            reason,
        )

    def withdraw_observation(self, observation_id: str, *, reason: str) -> None:
        self._withdraw(
            "observations",
            "observation_id",
            observation_id,
            "observation.withdraw",
            reason,
        )

    def add_claim(self, data: Mapping[str, Any]) -> str:
        return self._add("claims", "claim_id", data, "claim.add")

    def replace_claim(
        self,
        claim_id: str,
        data: Mapping[str, Any],
        *,
        reason: str,
    ) -> str:
        return self._replace(
            "claims", "claim_id", claim_id, data, "claim.replace", reason
        )

    def withdraw_claim(self, claim_id: str, *, reason: str) -> None:
        self._withdraw("claims", "claim_id", claim_id, "claim.withdraw", reason)

    def add_evidence(
        self,
        target: str,
        source: str,
        *,
        relation: str = "supports",
        locator: Optional[str] = None,
        note: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if relation not in {"supports", "contradicts", "context"}:
            raise ValueError(
                "evidence relation must be 'supports', 'contradicts', or 'context'"
            )
        provenance = _evidence_details(source, details)
        return self._add(
            "evidence",
            "evidence_id",
            {
                "target": target,
                "source": source,
                "relation": relation,
                "locator": locator,
                "note": note,
                "status": "active",
                **provenance,
            },
            "evidence.add",
        )

    def verify_evidence(
        self,
        evidence_id: str,
        *,
        method: str,
        verifier: str,
        note: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if method not in {"AI", "Human"}:
            raise ValueError("evidence verification method must be 'AI' or 'Human'")
        if not isinstance(verifier, str) or not verifier.strip():
            raise ValueError("evidence verifier must be a non-empty string")
        verification_details = _verification_details(details)

        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
            self._active_item(facts, "evidence", "evidence_id", evidence_id)
            verification_id = self._new_identifier("verification")
            facts.setdefault("verifications", []).append(
                {
                    "verification_id": verification_id,
                    "evidence_id": evidence_id,
                    "method": method,
                    "verifier": verifier,
                    "note": note,
                    "status": "active",
                    **verification_details,
                }
            )
            return {"verification_id": verification_id}, verification_id

        return self._mutate("evidence.verify", mutate)

    def withdraw_verification(self, verification_id: str, *, reason: str) -> None:
        self._withdraw(
            "verifications",
            "verification_id",
            verification_id,
            "verification.withdraw",
            reason,
        )

    def replace_evidence(
        self,
        evidence_id: str,
        source: str,
        *,
        reason: str,
        locator: Optional[str] = None,
        note: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> str:
        provenance = _evidence_details(source, details)

        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
            previous = self._active_item(
                facts, "evidence", "evidence_id", evidence_id
            )
            replacement = {
                key: deepcopy(value)
                for key, value in previous.items()
                if key in {"target", "target_kind", "relation"}
            }
            replacement.update(
                {
                    "source": source,
                    "locator": locator,
                    "note": note,
                    **provenance,
                }
            )
            new_id = self._new_identifier("evidence")
            previous["status"] = "superseded"
            previous["superseded_by"] = new_id
            replacement.update(
                {
                    "evidence_id": new_id,
                    "status": "active",
                    "supersedes": evidence_id,
                }
            )
            facts.setdefault("evidence", []).append(replacement)
            return {
                "old_id": evidence_id,
                "new_id": new_id,
                "reason": reason,
            }, new_id

        return self._mutate("evidence.replace", mutate)

    def withdraw_evidence(self, evidence_id: str, *, reason: str) -> None:
        self._withdraw(
            "evidence", "evidence_id", evidence_id, "evidence.withdraw", reason
        )

    def attest_benchmark(
        self,
        approved: bool,
        *,
        authority: str,
        actor: str,
        reason: Optional[str] = None,
    ) -> str:
        if not isinstance(approved, bool):
            raise TypeError("approved must be a Boolean")
        if authority not in {"User", "Other"}:
            raise ValueError("public benchmark authority must be 'User' or 'Other'")
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("benchmark attestation actor must be a non-empty string")

        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
            new_id = self._new_identifier("attestation")
            superseded = []
            for item in facts.setdefault("attestations", []):
                if (
                    item.get("status", "active") == "active"
                    and item.get("authority") == authority
                    and item.get("actor") == actor
                ):
                    item["status"] = "superseded"
                    item["superseded_by"] = new_id
                    superseded.append(str(item.get("attestation_id")))
            attestation = {
                "attestation_id": new_id,
                "approved": approved,
                "authority": authority,
                "actor": actor,
                "reason": reason,
                "status": "active",
            }
            if superseded:
                attestation["supersedes"] = superseded
            facts["attestations"].append(attestation)
            return {
                "attestation_id": new_id,
                "superseded": superseded,
            }, new_id

        return self._mutate("benchmark.attest", mutate)

    def withdraw_attestation(self, attestation_id: str, *, reason: str) -> None:
        self._withdraw(
            "attestations",
            "attestation_id",
            attestation_id,
            "benchmark_attestation.withdraw",
            reason,
        )

    def _new_identifier(self, kind: str) -> str:
        return self.__owner._new_fact_identifier(self.id, kind)

    @staticmethod
    def _active_item(
        facts: Mapping[str, Any], collection: str, id_key: str, item_id: str
    ) -> Dict[str, Any]:
        for item in facts.get(collection, []):
            if item.get(id_key) == item_id and item.get("status", "active") == "active":
                return item
        raise KeyError(item_id)

    def _add(
        self,
        collection: str,
        id_key: str,
        data: Mapping[str, Any],
        action: str,
    ) -> str:
        item_template = deepcopy(dict(data))
        if any(
            key in item_template
            for key in (
                "quality_level",
                "evidence_level",
                "archive_class",
                "benchmark",
            )
        ):
            raise ValueError("derived assessment fields cannot be supplied as facts")

        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
            item = deepcopy(item_template)
            item_id = str(
                item.get(id_key)
                or self._new_identifier(id_key.removesuffix("_id"))
            )
            if any(
                existing.get(id_key) == item_id
                for existing in facts.get(collection, [])
            ):
                raise ValueError(f"duplicate {id_key} {item_id!r}")
            item[id_key] = item_id
            item.setdefault("status", "active")
            facts.setdefault(collection, []).append(item)
            return {id_key: item_id}, item_id

        return self._mutate(action, mutate)

    def _replace(
        self,
        collection: str,
        id_key: str,
        item_id: str,
        data: Mapping[str, Any],
        action: str,
        reason: str,
    ) -> str:
        replacement_template = deepcopy(dict(data))
        if any(
            key in replacement_template
            for key in (
                "quality_level",
                "evidence_level",
                "archive_class",
                "benchmark",
            )
        ):
            raise ValueError("derived assessment fields cannot be supplied as facts")

        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
            previous = self._active_item(facts, collection, id_key, item_id)
            replacement = deepcopy(replacement_template)
            new_id = str(
                replacement.get(id_key)
                or self._new_identifier(id_key.removesuffix("_id"))
            )
            if any(
                existing.get(id_key) == new_id
                for existing in facts.get(collection, [])
            ):
                raise ValueError(f"duplicate {id_key} {new_id!r}")
            previous["status"] = "superseded"
            previous["superseded_by"] = new_id
            replacement.update(
                {id_key: new_id, "status": "active", "supersedes": item_id}
            )
            facts.setdefault(collection, []).append(replacement)
            return {
                "old_id": item_id,
                "new_id": new_id,
                "reason": reason,
            }, new_id

        return self._mutate(action, mutate)

    def _withdraw(
        self,
        collection: str,
        id_key: str,
        item_id: str,
        action: str,
        reason: str,
    ) -> None:
        def mutate(facts: Dict[str, Any]) -> tuple[Dict[str, Any], None]:
            item = self._active_item(facts, collection, id_key, item_id)
            item["status"] = "withdrawn"
            item["withdrawal_reason"] = reason
            return {id_key: item_id, "reason": reason}, None

        self._mutate(action, mutate)

    def __repr__(self) -> str:
        return f"Case(id={self.id!r})"


class CaseView(Sequence[Case]):
    """A stable read-only view returning the owner's original Case objects."""

    __slots__ = ("_owner", "_ids")

    def __init__(self, owner: Any, case_ids: Sequence[str]) -> None:
        self._owner = owner
        self._ids = tuple(case_ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Case]:
        return (self._owner._case_object(case_id) for case_id in self._ids)

    def __getitem__(self, index: Union[int, slice]) -> Union[Case, "CaseView"]:
        if isinstance(index, slice):
            return CaseView(self._owner, self._ids[index])
        return self._owner._case_object(self._ids[index])


class CaseCollection(CaseView):
    """The one complete collection facade for all Bundle-owned Cases."""

    _ALLOWED = frozenset(
        {
            "archive_class",
            "quality_level",
            "min_quality",
            "evidence_level",
            "min_evidence",
            "benchmark_status",
            "reason",
            "verification",
            "attestation",
        }
    )
    _B_LEVELS = {f"B{index}": index for index in range(5)}
    _E_LEVELS = {f"E{index}": index for index in range(5)}

    def __init__(self, owner: Any) -> None:
        super().__init__(owner, owner._case_ids())

    def select(self, **criteria: Any) -> CaseView:
        unknown = sorted(set(criteria) - self._ALLOWED)
        if unknown:
            raise TypeError(f"unknown Case selection fields: {unknown}")
        selected = []
        for case_id in self._ids:
            case = self._owner._case_object(case_id)
            if self._matches(case, criteria):
                selected.append(case_id)
        return CaseView(self._owner, selected)

    @classmethod
    def _matches(cls, case: Case, criteria: Mapping[str, Any]) -> bool:
        assessment = case.assessment
        for name, expected in criteria.items():
            if name == "reason":
                actual: Any = [item.get("code") for item in assessment.get("reasons", [])]
            elif name == "verification":
                actual = [item.get("method") for item in case._facts().get("verifications", []) if item.get("status", "active") == "active"]
            elif name == "attestation":
                actual = [item.get("authority") for item in case._facts().get("attestations", []) if item.get("status", "active") == "active"]
            elif name == "min_quality":
                actual_level = str(assessment["quality_level"])
                wanted = str(expected)
                if cls._B_LEVELS[actual_level] < cls._B_LEVELS[wanted]:
                    return False
                continue
            elif name == "min_evidence":
                actual_value = assessment.get("evidence_level")
                if actual_value is None:
                    return False
                actual_level = str(actual_value)
                wanted = str(expected)
                if cls._E_LEVELS[actual_level] < cls._E_LEVELS[wanted]:
                    return False
                continue
            else:
                actual = assessment.get(name)
            if isinstance(expected, (list, set, tuple, frozenset)):
                options = set(expected)
                if isinstance(actual, list):
                    if not any(item in options for item in actual):
                        return False
                elif actual not in options:
                    return False
            elif isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
        return True


__all__ = [
    "Case",
    "CaseCollection",
    "CaseOwnershipError",
    "CaseView",
    "UnsynchronizedBundleError",
]
