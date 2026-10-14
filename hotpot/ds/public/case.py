"""Bundle-owned scientific cases and read-only case collections.

These types are intentionally returned by :class:`hotpot.ds.Bundle` rather
than exported from the first-level public namespace.  A Case never owns or
persists state: every read and mutation goes through its sole Bundle owner.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, Dict, Optional, Union


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

    def _commit(self, facts: Mapping[str, Any], action: str, details: Mapping[str, Any]) -> None:
        self.__owner._commit_case(
            self.__case_id,
            deepcopy(dict(facts)),
            action,
            deepcopy(dict(details)),
        )

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
        facts = self._facts()
        core_info = dict(facts.get("core_info", {}))
        before = deepcopy(core_info.get(name))
        core_info[name] = deepcopy(value)
        facts["core_info"] = core_info
        self._commit(
            facts,
            "core_info.set",
            {"name": name, "before": before, "after": value, "reason": reason},
        )

    def remove_core_info(self, name: str, *, reason: str) -> None:
        facts = self._facts()
        core_info = dict(facts.get("core_info", {}))
        before = core_info.pop(name)
        facts["core_info"] = core_info
        self._commit(
            facts,
            "core_info.remove",
            {"name": name, "before": before, "reason": reason},
        )

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
    ) -> str:
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
    ) -> str:
        if method not in {"AI", "Human"}:
            raise ValueError("evidence verification method must be 'AI' or 'Human'")
        facts = self._facts()
        self._active_item(facts, "evidence", "evidence_id", evidence_id)
        return self._add(
            "verifications",
            "verification_id",
            {
                "evidence_id": evidence_id,
                "method": method,
                "verifier": verifier,
                "note": note,
                "status": "active",
            },
            "evidence.verify",
        )

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
    ) -> str:
        facts = self._facts()
        previous = self._active_item(
            facts, "evidence", "evidence_id", evidence_id
        )
        replacement = {
            key: deepcopy(value)
            for key, value in previous.items()
            if key not in {"evidence_id", "status", "withdrawal_reason"}
        }
        replacement.update({"source": source, "locator": locator, "note": note})
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
        self._commit(
            facts,
            "evidence.replace",
            {"old_id": evidence_id, "new_id": new_id, "reason": reason},
        )
        return new_id

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
        if authority not in {"User", "Other"}:
            raise ValueError("public benchmark authority must be 'User' or 'Other'")
        return self._add(
            "attestations",
            "attestation_id",
            {
                "approved": bool(approved),
                "authority": authority,
                "actor": actor,
                "reason": reason,
                "status": "active",
            },
            "benchmark.attest",
        )

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
        facts = self._facts()
        item = deepcopy(dict(data))
        if any(key in item for key in ("quality_level", "evidence_level", "archive_class", "benchmark")):
            raise ValueError("derived assessment fields cannot be supplied as facts")
        item_id = str(item.get(id_key) or self._new_identifier(id_key.removesuffix("_id")))
        if any(existing.get(id_key) == item_id for existing in facts.get(collection, [])):
            raise ValueError(f"duplicate {id_key} {item_id!r}")
        item[id_key] = item_id
        item.setdefault("status", "active")
        facts.setdefault(collection, []).append(item)
        self._commit(facts, action, {id_key: item_id})
        return item_id

    def _replace(
        self,
        collection: str,
        id_key: str,
        item_id: str,
        data: Mapping[str, Any],
        action: str,
        reason: str,
    ) -> str:
        facts = self._facts()
        previous = self._active_item(facts, collection, id_key, item_id)
        replacement = deepcopy(dict(data))
        if any(key in replacement for key in ("quality_level", "evidence_level", "archive_class", "benchmark")):
            raise ValueError("derived assessment fields cannot be supplied as facts")
        new_id = str(replacement.get(id_key) or self._new_identifier(id_key.removesuffix("_id")))
        if any(existing.get(id_key) == new_id for existing in facts.get(collection, [])):
            raise ValueError(f"duplicate {id_key} {new_id!r}")
        previous["status"] = "superseded"
        previous["superseded_by"] = new_id
        replacement.update({id_key: new_id, "status": "active", "supersedes": item_id})
        facts.setdefault(collection, []).append(replacement)
        self._commit(
            facts,
            action,
            {"old_id": item_id, "new_id": new_id, "reason": reason},
        )
        return new_id

    def _withdraw(
        self,
        collection: str,
        id_key: str,
        item_id: str,
        action: str,
        reason: str,
    ) -> None:
        facts = self._facts()
        item = self._active_item(facts, collection, id_key, item_id)
        item["status"] = "withdrawn"
        item["withdrawal_reason"] = reason
        self._commit(facts, action, {id_key: item_id, "reason": reason})

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
                actual_level = str(assessment["quality_level"]).split("_", 1)[0]
                wanted = str(expected).split("_", 1)[0]
                if cls._B_LEVELS[actual_level] < cls._B_LEVELS[wanted]:
                    return False
                continue
            elif name == "min_evidence":
                actual_value = assessment.get("evidence_level")
                if actual_value is None:
                    return False
                actual_level = str(actual_value).split("_", 1)[0]
                wanted = str(expected).split("_", 1)[0]
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
