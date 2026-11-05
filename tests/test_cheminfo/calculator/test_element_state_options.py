"""Source common-state subsets and bounded modeled element defaults."""

from dataclasses import FrozenInstanceError
from importlib import resources
import json

import pytest

from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import (
    ElementOxidationStateDomain,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.registry import (
    element_oxidation_state_domain,
)


def _payload():
    return json.loads(resources.files(
        "hotpot.cheminfo.rules.chemistry.scientific_state"
    ).joinpath("data/element_oxidation_states.json").read_text())


def test_all_elements_have_attributable_options_and_explicit_default_basis():
    payload = _payload()
    sources = {source.get("source_identity"): source for source in payload["sources"]}
    for row in payload["elements"]:
        domain = element_oxidation_state_domain(row["atomic_number"])
        assert set(domain.important_states) <= set(domain.common_states) <= set(domain.states)
        assert domain.common_states == tuple(row["common_states"])
        assert domain.important_states == tuple(row["important_states"])
        assert domain.default_state == row["default_state"]
        assert domain.default_basis == row["default_basis"]
        source = row["options_source"]
        assert source["source_identity"] in sources
        assert source["url"].startswith("https://periodic-table.rsc.org/element/")
        assert len(source["sha256"]) == 64
        assert source["access_date"] == "2026-11-05"
        assert domain.registry_version == "2026-11-05.common-default-v1"
        assert domain.default_state is not None
        if domain.default_state not in domain.states:
            assert domain.default_state == 0
            assert domain.default_basis == "elemental-reference-no-compound-preference"


def test_registry_preserves_source_missingness_and_ambiguity():
    rows = _payload()["elements"]
    assert sum(bool(row["common_states"]) for row in rows) == 100
    assert sum(bool(row["important_states"]) for row in rows) == 99
    assert [row["symbol"] for row in rows if len(row["important_states"]) > 1] == [
        "Sn", "Ir", "Pt",
    ]
    assert {row["options_source"]["common_status"] for row in rows} == {
        "reported", "blank", "unknown",
    }


@pytest.mark.parametrize(
    ("z", "common", "important", "default"),
    (
        (6, (-4, -3, -2, -1, 0, 1, 2, 3, 4), (-4,), -4),
        (9, (-1,), (-1,), -1),
        (26, (-2, 0, 2, 3, 6), (3,), 3),
        (63, (2, 3), (3,), 3),
        (92, (3, 4, 5, 6), (6,), 6),
        (102, (2, 3), (3,), 3),
    ),
)
def test_source_emphasis_is_distinct_from_display_order_and_valence(
    z, common, important, default,
):
    domain = element_oxidation_state_domain(z)
    assert domain.common_states == common
    assert domain.important_states == important
    assert domain.default_state == default
    assert domain.default_basis == "rsc-unique-important"


@pytest.mark.parametrize(
    ("z", "important", "default", "basis"),
    (
        (50, (2, 4), 2, "modeled-least-absolute-nonzero-important"),
        (77, (1, 4), 1, "modeled-least-absolute-nonzero-important"),
        (78, (2, 4), 2, "modeled-least-absolute-nonzero-important"),
        (54, (), 2, "modeled-least-absolute-nonzero-common"),
    ),
)
def test_ambiguous_source_subsets_keep_explicit_modeled_tie_breaks(
    z, important, default, basis,
):
    domain = element_oxidation_state_domain(z)
    assert domain.important_states == important
    assert domain.default_state == default
    assert domain.default_basis == basis


@pytest.mark.parametrize("z", (2, 10, 18, 104, 113, 116, 118))
def test_missing_source_common_states_do_not_become_observed_zero(z):
    domain = element_oxidation_state_domain(z)
    assert domain.common_states == ()
    assert domain.important_states == ()
    assert domain.default_state == 0
    assert domain.default_basis == "elemental-reference-no-compound-preference"
    if z == 113:
        assert domain.states == ()


@pytest.mark.parametrize(("z", "added_state"), ((44, 0), (45, 0), (46, 0), (77, 0), (79, 4)))
def test_source_published_domain_additions_retain_attributable_evidence(z, added_state):
    row = _payload()["elements"][z - 1]
    added = next(state for state in row["states"] if state["value"] == added_state)
    assert added_state in row["common_states"]
    assert added["source_class"] == "rsc_visual_elements_compiled"
    assert added["source_identity"] == row["options_source"]["source_identity"]


def test_old_domain_constructor_keeps_defaulted_option_fields():
    domain = ElementOxidationStateDomain(26, (2, 3), "source", "version", "digest")
    assert domain.common_states == ()
    assert domain.important_states == ()
    assert domain.default_state is None
    assert domain.default_basis == ""
    with pytest.raises(FrozenInstanceError):
        domain.default_state = 3
