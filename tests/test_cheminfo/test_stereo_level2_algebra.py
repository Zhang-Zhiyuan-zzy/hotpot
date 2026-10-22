import json
from itertools import permutations
from pathlib import Path

import pytest

from hotpot.cheminfo import stereo


_GOLDEN = json.loads(
    (
        Path(__file__).parents[1]
        / "data"
        / "stereo"
        / "level2_algebra_golden.json"
    ).read_text()
)


def _carrier(token):
    if token.startswith("a:"):
        return stereo.StereoCarrier(
            stereo.StereoCarrierKind.ATOM,
            int(token.removeprefix("a:")),
        )
    return stereo.StereoCarrier(stereo.StereoCarrierKind(token))


def _parsed(text):
    result = stereo._parse_level2_descriptor(text)
    assert result.issues == ()
    assert result.descriptor is not None
    return result.descriptor


def _anchor(token):
    kind, atom_text = token.split(":")
    atom_indices = tuple(map(int, atom_text.split(",")))
    return stereo.StereoAnchor(
        stereo.StereoAnchorKind.ATOM if kind == "a" else stereo.StereoAnchorKind.BOND,
        atom_indices,
    )


def test_golden_corpus_has_stable_schema():
    assert _GOLDEN["schema"] == "hotpot.stereo.level2-algebra-golden/1"


@pytest.mark.parametrize("case", _GOLDEN["permutations"])
def test_permutation_golden_facts(case):
    items = tuple(case["items"])
    assert stereo._permutation_parity(items) == case["parity"]
    assert stereo._permutation_rank(items) == case["rank"]
    assert stereo._permutation_unrank(len(items), case["rank"]) == items


@pytest.mark.parametrize("size", range(7))
def test_rank_and_unrank_are_exact_for_small_finite_families(size):
    for expected_rank, permutation in enumerate(permutations(range(size))):
        assert stereo._permutation_rank(permutation) == expected_rank
        assert stereo._permutation_unrank(size, expected_rank) == permutation


@pytest.mark.parametrize(
    ("family", "expected_count"),
    (("SP", 3), ("TBP", 20), ("OH", 30)),
)
def test_all_supported_coordination_indices_parse_and_render(family, expected_count):
    indices = _GOLDEN["descriptor_ranges"][family]
    assert indices == list(range(1, expected_count + 1))
    for index in indices:
        text = f"hotpot.stereo/2/{family}/{index}"
        assert stereo._render_level2_descriptor(_parsed(text)) == text


@pytest.mark.parametrize("family", ("SP", "TBP", "OH"))
def test_coordination_swap_transform_covers_entire_golden_family(family):
    indices = _GOLDEN["descriptor_ranges"][family]
    expected = _GOLDEN["coordination_swap_01"][family]
    carrier_count = {"SP": 4, "TBP": 5, "OH": 6}[family]
    source = tuple(
        _carrier(f"a:{atom}") for atom in range(carrier_count)
    )
    reordered = (source[1], source[0], *source[2:])
    actual = []
    for index in indices:
        result = stereo._transform_level2_descriptor(
            _parsed(f"hotpot.stereo/2/{family}/{index}"),
            source,
            reordered,
        )
        assert result.issues == ()
        actual.append(result.descriptor.value)
    assert actual == expected


@pytest.mark.parametrize("case", _GOLDEN["transforms"], ids=lambda case: case["name"])
def test_descriptor_transform_golden_facts(case):
    source = tuple(map(_carrier, case["source"]))
    reordered = tuple(map(_carrier, case["reordered"]))
    result = stereo._transform_level2_descriptor(
        _parsed(case["descriptor"]),
        source,
        reordered,
        source_anchor=(
            _anchor(case["source_anchor"])
            if "source_anchor" in case
            else None
        ),
        reordered_anchor=(
            _anchor(case["reordered_anchor"])
            if "reordered_anchor" in case
            else None
        ),
    )
    assert result.issues == ()
    assert stereo._render_level2_descriptor(result.descriptor) == case["expected"]


@pytest.mark.parametrize(
    "text",
    (
        "hotpot.stereo/2/TH/even",
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/DB/together",
        "hotpot.stereo/2/DB/opposite",
        "hotpot.stereo/2/AX/even",
        "hotpot.stereo/2/AX/odd",
    ),
)
def test_reference_relative_descriptors_parse_and_render(text):
    assert stereo._render_level2_descriptor(_parsed(text)) == text


@pytest.mark.parametrize(
    ("text", "code"),
    (
        ("hotpot.stereo/1/TH/even", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/TH/R", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/DB/E", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/SP/0", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/TBP/21", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/OH/31", stereo.StereoIssueCode.INVALID_DESCRIPTOR),
        ("hotpot.stereo/2/NEW/1", stereo.StereoIssueCode.UNSUPPORTED),
    ),
)
def test_invalid_or_unknown_descriptor_has_stable_issue(text, code):
    result = stereo._parse_level2_descriptor(text)
    assert result.descriptor is None
    assert tuple(issue.code for issue in result.issues) == (code,)


def test_duplicate_and_malformed_carriers_are_rejected():
    carriers = (
        stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, 1),
        stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H),
        stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H),
        stereo.StereoCarrier(stereo.StereoCarrierKind.LONE_PAIR, 3),
    )
    issues = stereo._validate_stereo_carriers(carriers, expected_count=4)
    assert tuple(issue.code for issue in issues) == (
        stereo.StereoIssueCode.INVALID_CARRIER,
        stereo.StereoIssueCode.INVALID_CARRIER,
    )
    assert [issue.message for issue in issues] == [
        "non-atom carrier at position 3 cannot carry an atom row",
        "duplicate carrier identity at position 2",
    ]


def test_alkene_reordering_cannot_mix_endpoint_carrier_pairs():
    source = tuple(_carrier(token) for token in ("a:1", "a:2", "a:3", "a:4"))
    reordered = (source[0], source[2], source[1], source[3])
    result = stereo._transform_level2_descriptor(
        _parsed("hotpot.stereo/2/DB/together"),
        source,
        reordered,
    )
    assert result.descriptor is None
    assert tuple(issue.code for issue in result.issues) == (
        stereo.StereoIssueCode.INVALID_CARRIER,
    )


def test_reverse_bond_anchor_requires_reverse_endpoint_carrier_pairs():
    source = tuple(_carrier(token) for token in ("a:1", "a:2", "a:3", "a:4"))
    result = stereo._transform_level2_descriptor(
        _parsed("hotpot.stereo/2/DB/together"),
        source,
        source,
        source_anchor=_anchor("b:10,20"),
        reordered_anchor=_anchor("b:20,10"),
    )
    assert result.descriptor is None
    assert tuple(issue.code for issue in result.issues) == (
        stereo.StereoIssueCode.INVALID_CARRIER,
    )


@pytest.mark.parametrize(
    "anchor",
    (
        stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (1, 2)),
        stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (1, 1)),
        stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (-1, 2)),
    ),
)
def test_invalid_anchor_shape_or_identity_is_reported(anchor):
    issues = stereo._validate_stereo_anchor(
        anchor,
        family=stereo._Level2DescriptorFamily.DOUBLE_BOND,
    )
    assert issues
    assert all(issue.code is stereo.StereoIssueCode.INVALID_ANCHOR for issue in issues)


def test_reordering_requires_exactly_the_same_carrier_identities():
    source = tuple(_carrier(token) for token in ("a:1", "a:2", "a:3", "a:4"))
    reordered = (*source[:3], _carrier("a:9"))
    result = stereo._transform_level2_descriptor(
        _parsed("hotpot.stereo/2/TH/even"),
        source,
        reordered,
    )
    assert result.descriptor is None
    assert tuple(issue.code for issue in result.issues) == (
        stereo.StereoIssueCode.INVALID_CARRIER,
    )


@pytest.mark.parametrize("permutation", ((0, 0), (0, 2), (1, -1)))
def test_invalid_permutation_is_rejected(permutation):
    with pytest.raises(ValueError, match="each index exactly once"):
        stereo._permutation_rank(permutation)


@pytest.mark.parametrize(("size", "rank"), ((-1, 0), (3, -1), (3, 6)))
def test_invalid_permutation_rank_is_rejected(size, rank):
    with pytest.raises(ValueError, match="outside the permutation family"):
        stereo._permutation_unrank(size, rank)
