from copy import deepcopy

import pytest

from hotpot.ds.public.case import (
    CaseCollection,
    CaseOwnershipError,
    UnsynchronizedBundleError,
)


class Owner:
    def __init__(self):
        self.payloads = {
            "c1": {
                "id": "c1",
                "local_id": "one",
                "core_info": {"phase": "solid"},
                "structures": [],
                "observations": [],
                "claims": [],
                "evidence": [],
                "verifications": [],
                "attestations": [],
            },
            "c2": {
                "id": "c2",
                "local_id": "two",
                "core_info": {},
                "structures": [],
                "observations": [],
                "claims": [],
                "evidence": [],
                "verifications": [],
                "attestations": [],
            },
        }
        self.assessments = {
            "c1": {
                "quality_level": "B3",
                "evidence_level": "E3",
                "evidence_levels": {},
                "archive_class": "benchmark",
                "benchmark": True,
                "benchmark_status": "admitted",
                "reasons": [{"code": "primary_source_verified"}],
            },
            "c2": {
                "quality_level": "B1",
                "evidence_level": None,
                "evidence_levels": {},
                "archive_class": "incomplete",
                "benchmark": False,
                "benchmark_status": "ineligible",
                "reasons": [{"code": "missing_structure"}],
            },
        }
        self.objects = {}
        self.is_synchronized = True
        self.counter = 0

    def _case_ids(self):
        return tuple(self.payloads)

    def _case_object(self, case_id):
        from hotpot.ds.public.case import Case

        return self.objects.setdefault(case_id, Case(self, case_id))

    def _case_payload(self, case_id):
        return deepcopy(self.payloads[case_id])

    def _case_assessment(self, case_id):
        if not self.is_synchronized:
            raise UnsynchronizedBundleError("synchronize first")
        return deepcopy(self.assessments[case_id])

    def _mutate_case(self, case_id, action, mutation):
        payload = deepcopy(self.payloads[case_id])
        _, result = mutation(payload)
        self.payloads[case_id] = payload
        self.is_synchronized = False
        return result

    def _new_fact_identifier(self, case_id, kind):
        self.counter += 1
        return f"{case_id}:{kind}:{self.counter}"


def test_collection_owns_all_cases_and_views_return_same_objects():
    owner = Owner()
    cases = CaseCollection(owner)
    assert len(cases) == 2
    assert cases[0] is cases[:1][0]
    assert cases.select(archive_class="benchmark")[0] is cases[0]
    assert [case.id for case in cases.select(min_quality="B2")] == ["c1"]
    assert [case.id for case in cases.select(reason="missing_structure")] == ["c2"]


def test_reads_are_defensive_and_ownership_is_immutable():
    owner = Owner()
    case = CaseCollection(owner)[0]
    value = case.core_info
    value["phase"] = "gas"
    assert case.core_info["phase"] == "solid"
    with pytest.raises(CaseOwnershipError):
        case._Case__owner = Owner()


def test_mutation_only_marks_owner_unsynchronized():
    owner = Owner()
    case = CaseCollection(owner)[0]
    case.set_core_info("temperature", 298)
    assert owner.payloads["c1"]["core_info"]["temperature"] == 298
    assert not owner.is_synchronized
    with pytest.raises(UnsynchronizedBundleError):
        _ = case.quality_level


def test_fact_lifecycle_is_append_only_and_derived_fields_are_rejected():
    owner = Owner()
    case = CaseCollection(owner)[0]
    observation_id = case.add_observation({"property": "spin", "value": 1})
    owner.is_synchronized = True
    replacement_id = case.replace_observation(
        observation_id, {"property": "spin", "value": 2}, reason="correction"
    )
    rows = owner.payloads["c1"]["observations"]
    assert rows[0]["status"] == "superseded"
    assert rows[1]["observation_id"] == replacement_id
    with pytest.raises(ValueError, match="derived"):
        case.add_claim({"text": "claim", "evidence_level": "E4"})


def test_public_attestation_cannot_claim_hotpot_authority():
    owner = Owner()
    case = CaseCollection(owner)[0]
    with pytest.raises(ValueError, match="User.*Other"):
        case.attest_benchmark(True, authority="Hotpot", actor="caller")
    attestation_id = case.attest_benchmark(
        True, authority="User", actor="owner", reason="reviewed"
    )
    assert attestation_id.startswith("c1:attestation:")


def test_public_attestation_requires_exact_boolean_and_named_actor():
    owner = Owner()
    case = CaseCollection(owner)[0]
    with pytest.raises(TypeError, match="Boolean"):
        case.attest_benchmark(1, authority="User", actor="owner")
    with pytest.raises(ValueError, match="non-empty"):
        case.attest_benchmark(True, authority="User", actor="")


def test_new_attestation_supersedes_same_authority_and_actor_only():
    owner = Owner()
    case = CaseCollection(owner)[0]
    first = case.attest_benchmark(False, authority="User", actor="owner")
    owner.is_synchronized = True
    other = case.attest_benchmark(False, authority="User", actor="reviewer")
    owner.is_synchronized = True
    replacement = case.attest_benchmark(True, authority="User", actor="owner")

    rows = owner.payloads["c1"]["attestations"]
    assert rows[0]["attestation_id"] == first
    assert rows[0]["status"] == "superseded"
    assert rows[0]["superseded_by"] == replacement
    assert rows[1]["attestation_id"] == other
    assert rows[1]["status"] == "active"
    assert rows[2]["supersedes"] == [first]
    assert rows[2]["approved"] is True
