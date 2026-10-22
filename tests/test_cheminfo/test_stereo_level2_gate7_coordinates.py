import hashlib
import json
from itertools import permutations
from math import factorial, radians, sqrt
from pathlib import Path

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


_DATA_DIRECTORY = Path(__file__).parents[1] / "data" / "stereo"
_GOLDEN = json.loads(
    (
        _DATA_DIRECTORY
        / "level2_gate7_coordination_geometry_golden.json"
    ).read_text()
)
_FAMILIES = {
    "SP": stereo._Level2DescriptorFamily.SQUARE_PLANAR,
    "TBP": stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL,
    "SPY": stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL,
    "OH": stereo._Level2DescriptorFamily.OCTAHEDRAL,
}
_REFLECT_Y = np.diag((1.0, -1.0, 1.0))
_TEST_ROTATION = np.array(
    (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    )
)


def _family_golden(family):
    return _GOLDEN["families"][family.value]


def _descriptor(family, value):
    return stereo._Level2Descriptor(family, value)


def _independent_representatives(family):
    frozen = _family_golden(family)
    rotations = tuple(
        tuple(rotation) for rotation in frozen["proper_site_rotations"]
    )
    size = len(frozen["ideal_sites"])
    unseen = set(permutations(range(size)))
    representatives = []
    while unseen:
        seed = min(unseen)
        orbit = {
            tuple(seed[position] for position in rotation)
            for rotation in rotations
        }
        representatives.append(min(orbit))
        unseen.difference_update(orbit)
    return tuple(sorted(representatives))


def _independent_directions(family, value):
    sites = np.asarray(_family_golden(family)["ideal_sites"])
    representative = _independent_representatives(family)[value - 1]
    directions = np.empty_like(sites)
    for site, carrier_position in enumerate(representative):
        directions[carrier_position] = sites[site]
    return directions


def _independent_fit(ideal, observed):
    left, _, right_transpose = np.linalg.svd(ideal.T @ observed)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_transpose)
    rotation = left @ correction @ right_transpose
    fitted = ideal @ rotation
    angles = np.arctan2(
        np.linalg.norm(np.cross(fitted, observed), axis=1),
        np.clip(np.sum(fitted * observed, axis=1), -1.0, 1.0),
    )
    return rotation, float(np.sqrt(np.mean(np.square(angles))))


def _candidate(family, value, score):
    return stereo._Gate7CoordinateCandidate(
        _descriptor(family, value),
        score,
    )


def _candidate_pair(family, best, second):
    return (
        _candidate(family, 1, best),
        _candidate(family, 2, second),
    )


def _select(family, candidates):
    return stereo._select_gate7_coordination_descriptor(
        family,
        candidates,
        record_index=7,
        atom_indices=(4, 8, 15, 16, 23, 42),
    )


def _site(family, value, carrier_count=None):
    count = (
        len(_family_golden(family)["ideal_sites"])
        if carrier_count is None
        else carrier_count
    )
    return stereo._Gate7CoordinationSite(
        record_index=7,
        descriptor=_descriptor(family, value),
        center=0,
        carrier_rows=tuple(range(1, count + 1)),
    )


def _coordinates(directions, *, radii=None):
    center = np.array((13.0, -7.0, 5.0))
    count = len(directions)
    if radii is None:
        radii = np.linspace(0.75, 3.25, count)
    carriers = center + np.asarray(radii)[:, np.newaxis] * directions
    return np.vstack((center, carriers))


def _coordination_molecule(family, coordinates):
    carrier_count = len(coordinates) - 1
    molecule = Molecule()
    for atomic_number in (26, *((7,) * carrier_count)):
        molecule._create_atom(atomic_number=atomic_number)
    for carrier in range(1, carrier_count + 1):
        molecule._add_bond(0, carrier, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    descriptor = f"hotpot.stereo/2/{family.value}/1"
    record = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        carriers=tuple(
            stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)
            for row in range(1, carrier_count + 1)
        ),
        descriptor=descriptor,
        group=None,
        source_namespace="gate7-coordinate-test",
        source_code=descriptor,
    )
    payload = stereo.StereoPayload(1, (record,), (), ())
    stereo.set_stereo_payload(molecule, payload)
    return molecule, payload


def test_coordination_geometry_golden_is_independent_and_preserves_old_bytes():
    assert _GOLDEN["schema"] == (
        "hotpot.stereo.level2-gate7-coordination-geometry-golden/1"
    )
    for filename, expected in _GOLDEN["preserved_golden_sha256"].items():
        content = (_DATA_DIRECTORY / filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected

    policy = _GOLDEN["policy_radians"]
    assert policy == {
        "acceptance": radians(12.0),
        "same_family_margin": radians(2.0),
        "five_coordinate_margin": radians(3.0),
    }
    assert stereo._GATE7_ACCEPTANCE_RADIANS == policy["acceptance"]
    assert stereo._GATE7_SAME_FAMILY_MARGIN_RADIANS == policy[
        "same_family_margin"
    ]
    assert stereo._GATE7_FIVE_COORDINATE_MARGIN_RADIANS == policy[
        "five_coordinate_margin"
    ]


@pytest.mark.parametrize("family", tuple(_FAMILIES.values()))
def test_frozen_templates_have_complete_proper_site_rotation_groups(family):
    frozen = _family_golden(family)
    sites = np.asarray(frozen["ideal_sites"])
    expected = {
        tuple(rotation) for rotation in frozen["proper_site_rotations"]
    }
    actual = set()
    for reordering in permutations(range(len(sites))):
        rotation, score = _independent_fit(sites, sites[list(reordering)])
        if score < 1.0e-12 and np.linalg.det(rotation) > 0.0:
            actual.add(reordering)

    np.testing.assert_allclose(np.linalg.norm(sites, axis=1), 1.0)
    assert actual == expected
    assert set(stereo._COORDINATION_ROTATIONS[family]) == expected

    representatives = _independent_representatives(family)
    assert len(representatives) == frozen["cardinality"]
    assert len(representatives) * len(expected) == factorial(len(sites))
    assert representatives == stereo._COORDINATION_REPRESENTATIVES[family]


def test_all_83_descriptor_directions_follow_the_independent_site_oracle():
    descriptor_count = 0
    for family in _FAMILIES.values():
        frozen = _family_golden(family)
        for value in range(1, frozen["cardinality"] + 1):
            expected = _independent_directions(family, value)
            actual = stereo._gate7_ideal_carrier_directions(
                _descriptor(family, value)
            )
            np.testing.assert_array_equal(actual, expected)
            descriptor_count += 1
    assert descriptor_count == 83


@pytest.mark.parametrize("family", tuple(_FAMILIES.values()))
def test_proper_rotation_fitter_matches_an_independent_angular_oracle(family):
    assert np.linalg.det(_TEST_ROTATION) == pytest.approx(1.0)
    for value in range(1, _family_golden(family)["cardinality"] + 1):
        ideal = _independent_directions(family, value)
        observed = ideal @ _TEST_ROTATION
        expected_rotation, expected_score = _independent_fit(ideal, observed)
        rotation, score = stereo._gate7_proper_rotation_fit(ideal, observed)

        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-12)
        np.testing.assert_allclose(ideal @ rotation, observed, atol=1.0e-12)
        np.testing.assert_allclose(rotation, expected_rotation, atol=1.0e-12)
        assert score == pytest.approx(expected_score, abs=1.0e-14)
        assert score == pytest.approx(0.0, abs=1.0e-14)


@pytest.mark.parametrize("family", tuple(_FAMILIES.values()))
def test_every_ideal_descriptor_ranks_itself_and_matches_oracle_scores(family):
    cardinality = _family_golden(family)["cardinality"]
    for observed_value in range(1, cardinality + 1):
        observed = _independent_directions(family, observed_value)
        actual = stereo._gate7_coordination_candidates(family, observed)
        assert len(actual) == cardinality
        assert actual[0].descriptor == _descriptor(family, observed_value)
        assert actual[0].score_radians == pytest.approx(0.0, abs=1.0e-14)
        assert tuple(
            (candidate.score_radians, int(candidate.descriptor.value))
            for candidate in actual
        ) == tuple(
            sorted(
                (
                    candidate.score_radians,
                    int(candidate.descriptor.value),
                )
                for candidate in actual
            )
        )

        for candidate in actual:
            ideal = _independent_directions(
                family,
                int(candidate.descriptor.value),
            )
            _, expected = _independent_fit(ideal, observed)
            assert candidate.score_radians == pytest.approx(
                expected,
                abs=1.0e-13,
            )


@pytest.mark.parametrize("family", tuple(_FAMILIES.values()))
def test_nearest_wrong_descriptor_score_ties_and_witness_are_frozen(family):
    frozen = _family_golden(family)["nearest_wrong"]
    cardinality = _family_golden(family)["cardinality"]
    wrong_scores = []
    score_by_pair = {}
    for observed_value in range(1, cardinality + 1):
        observed = _independent_directions(family, observed_value)
        for candidate in stereo._gate7_coordination_candidates(family, observed):
            candidate_value = int(candidate.descriptor.value)
            if candidate_value == observed_value:
                continue
            pair = (observed_value, candidate_value)
            score_by_pair[pair] = candidate.score_radians
            wrong_scores.append(candidate.score_radians)

    minimum = min(wrong_scores)
    ties = sum(abs(score - minimum) < 1.0e-12 for score in wrong_scores)
    assert minimum == pytest.approx(frozen["score_radians"], abs=1.0e-13)
    assert ties == frozen["ordered_tie_count"]
    assert score_by_pair[tuple(frozen["witness"])] == pytest.approx(
        minimum,
        abs=1.0e-13,
    )


@pytest.mark.parametrize("family", tuple(_FAMILIES.values()))
def test_literal_y_reflection_follows_every_frozen_mirror_map(family):
    assert np.linalg.det(_REFLECT_Y) == pytest.approx(-1.0)
    frozen = _family_golden(family)
    mirror = tuple(frozen["mirror_y"])
    assert tuple(mirror[value - 1] for value in mirror) == tuple(
        range(1, frozen["cardinality"] + 1)
    )

    for source_value, expected_value in enumerate(mirror, start=1):
        source = _independent_directions(family, source_value)
        reflected = source @ _REFLECT_Y
        candidates = stereo._gate7_coordination_candidates(family, reflected)
        assert candidates[0].descriptor == _descriptor(family, expected_value)
        assert candidates[0].score_radians == pytest.approx(0.0, abs=1.0e-14)


def test_tbp_spy_cross_family_minimum_has_60_frozen_ties():
    tbp = stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    spy = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    frozen = _GOLDEN["five_coordinate_cross"]
    score_by_pair = {}
    for tbp_value in range(1, _family_golden(tbp)["cardinality"] + 1):
        observed = _independent_directions(tbp, tbp_value)
        for candidate in stereo._gate7_coordination_candidates(spy, observed):
            pair = (tbp_value, int(candidate.descriptor.value))
            score_by_pair[pair] = candidate.score_radians

    minimum = min(score_by_pair.values())
    ties = sum(
        abs(score - minimum) < 1.0e-12
        for score in score_by_pair.values()
    )
    witness = (
        frozen["witness"]["TBP"],
        frozen["witness"]["SPY"],
    )
    assert minimum == pytest.approx(frozen["score_radians"], abs=1.0e-13)
    assert ties == frozen["ordered_tie_count"] == 60
    assert score_by_pair[witness] == pytest.approx(minimum, abs=1.0e-13)


def test_no_centroid_probe_retains_center_rooted_independent_radii():
    probe = _GOLDEN["no_centroid_probe"]
    family = _FAMILIES[probe["family"]]
    site = _site(family, probe["descriptor"])
    coordinates = np.asarray(probe["coordinates"])

    directions, issue = stereo._gate7_center_rooted_directions(site, coordinates)
    assert issue is None
    np.testing.assert_array_equal(directions, probe["expected_directions"])
    np.testing.assert_array_equal(
        _independent_directions(family, probe["descriptor"])
        @ np.asarray(probe["proper_rotation"]),
        directions,
    )
    descriptor, issue = stereo._gate7_coordination_coordinate_descriptor(
        site,
        coordinates,
    )
    assert issue is None
    assert descriptor == site.descriptor

    donor_centroid = np.mean(coordinates[1:], axis=0)
    center = coordinates[probe["center_row"]]
    np.testing.assert_allclose(
        donor_centroid - center,
        probe["donor_centroid_offset"],
        atol=1.0e-15,
    )
    donor_centered = coordinates[1:] - donor_centroid
    donor_centered /= np.linalg.norm(donor_centered, axis=1)[:, np.newaxis]
    _, donor_centered_score = _independent_fit(
        _independent_directions(family, probe["descriptor"]),
        donor_centered,
    )
    assert donor_centered_score == pytest.approx(
        probe["donor_centered_score_radians"],
        abs=1.0e-13,
    )
    assert donor_centered_score > _GOLDEN["policy_radians"]["acceptance"]


def test_all_83_descriptors_survive_translation_rotation_scale_and_radii():
    descriptor_count = 0
    for family in _FAMILIES.values():
        cardinality = _family_golden(family)["cardinality"]
        for value in range(1, cardinality + 1):
            directions = _independent_directions(family, value) @ _TEST_ROTATION
            coordinates = _coordinates(
                directions,
                radii=np.geomspace(0.125, 32.0, len(directions)),
            )
            coordinates *= 17.0
            descriptor, issue = (
                stereo._gate7_coordination_coordinate_descriptor(
                    _site(family, value),
                    coordinates,
                )
            )
            assert issue is None
            assert descriptor == _descriptor(family, value)
            descriptor_count += 1
    assert descriptor_count == 83


def test_finite_extreme_coordinates_normalize_without_overflow():
    family = stereo._Level2DescriptorFamily.OCTAHEDRAL
    scale = 1.1e308
    coordinates = np.asarray(
        (
            (-scale, -scale, -scale),
            (scale, scale, scale),
            (scale, scale, -scale),
            (scale, -scale, scale),
            (-scale, scale, scale),
            (scale, -scale, -scale),
            (-scale, scale, -scale),
        )
    )

    directions, issue = stereo._gate7_center_rooted_directions(
        _site(family, 1),
        coordinates,
    )

    assert issue is None
    assert np.all(np.isfinite(directions))
    np.testing.assert_allclose(
        np.linalg.norm(directions, axis=1),
        np.ones(6),
        atol=1.0e-15,
    )


def test_huge_common_translation_does_not_underflow_local_directions():
    family = stereo._Level2DescriptorFamily.OCTAHEDRAL
    translation = 1.0e308
    coordinates = np.asarray(
        (
            (translation, 0.0, 0.0),
            (translation, 1.0, 0.0),
            (translation, -1.0, 0.0),
            (translation, 0.0, 1.0),
            (translation, 0.0, -1.0),
            (translation, 1.0, 1.0),
            (translation, -1.0, -1.0),
        )
    )

    directions, issue = stereo._gate7_center_rooted_directions(
        _site(family, 1),
        coordinates,
    )

    assert issue is None
    expected = np.asarray(
        (
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0 / sqrt(2.0), 1.0 / sqrt(2.0)),
            (0.0, -1.0 / sqrt(2.0), -1.0 / sqrt(2.0)),
        )
    )
    np.testing.assert_allclose(directions, expected, atol=1.0e-15)


def test_center_coincident_carrier_has_one_stable_relative_tolerance_issue():
    family = stereo._Level2DescriptorFamily.OCTAHEDRAL
    directions = _independent_directions(family, 1)
    coordinates = _coordinates(directions)
    coordinates[3] = coordinates[0]

    actual, issue = stereo._gate7_center_rooted_directions(
        _site(family, 1),
        coordinates,
    )

    assert actual is None
    assert issue == stereo.StereoValidationIssue(
        code=stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
        message=(
            "OH coordinate geometry has a carrier coincident with its center "
            "within relative tolerance 1e-08"
        ),
        record_index=7,
        atom_indices=(0, 1, 2, 3, 4, 5, 6),
    )


@pytest.mark.parametrize("relation", ("inside", "equal", "outside"))
def test_exact_12_degree_acceptance_boundary(relation):
    family = stereo._Level2DescriptorFamily.SQUARE_PLANAR
    cutoff = _GOLDEN["policy_radians"]["acceptance"]
    margin = _GOLDEN["policy_radians"]["same_family_margin"]
    best = {
        "inside": np.nextafter(cutoff, 0.0),
        "equal": cutoff,
        "outside": np.nextafter(cutoff, np.inf),
    }[relation]
    candidates = {
        family: _candidate_pair(family, best, best + margin * 2.0),
    }

    descriptor, issue = _select(family, candidates)

    if relation == "outside":
        assert descriptor is None
        assert issue == stereo.StereoValidationIssue(
            code=stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
            message=(
                "SP coordinate geometry exceeds the 12-degree angular RMS "
                "acceptance cutoff"
            ),
            record_index=7,
            atom_indices=(4, 8, 15, 16, 23, 42),
        )
    else:
        assert descriptor == _descriptor(family, 1)
        assert issue is None


@pytest.mark.parametrize("relation", ("inside", "equal", "outside"))
def test_exact_2_degree_same_family_boundary(relation):
    family = stereo._Level2DescriptorFamily.OCTAHEDRAL
    margin = _GOLDEN["policy_radians"]["same_family_margin"]
    gap = {
        "inside": np.nextafter(margin, 0.0),
        "equal": margin,
        "outside": np.nextafter(margin, np.inf),
    }[relation]
    candidates = {family: _candidate_pair(family, 0.0, gap)}

    descriptor, issue = _select(family, candidates)

    if relation == "outside":
        assert descriptor == _descriptor(family, 1)
        assert issue is None
    else:
        assert descriptor is None
        assert issue == stereo.StereoValidationIssue(
            code=stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
            message=(
                "OH coordinate geometry is ambiguous within the 2-degree "
                "same-family margin"
            ),
            record_index=7,
            atom_indices=(4, 8, 15, 16, 23, 42),
        )


def test_same_family_margin_uses_raw_second_score_above_acceptance():
    family = stereo._Level2DescriptorFamily.SQUARE_PLANAR
    cutoff = _GOLDEN["policy_radians"]["acceptance"]
    candidates = {
        family: _candidate_pair(
            family,
            cutoff - radians(1.0),
            cutoff + radians(0.5),
        )
    }

    descriptor, issue = _select(family, candidates)

    assert descriptor is None
    assert issue is not None
    assert issue.code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
    assert issue.message == (
        "SP coordinate geometry is ambiguous within the 2-degree "
        "same-family margin"
    )


@pytest.mark.parametrize("relation", ("inside", "equal", "outside"))
def test_exact_3_degree_five_coordinate_boundary(relation):
    tbp = stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    spy = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    family_margin = _GOLDEN["policy_radians"]["five_coordinate_margin"]
    same_margin = _GOLDEN["policy_radians"]["same_family_margin"]
    gap = {
        "inside": np.nextafter(family_margin, 0.0),
        "equal": family_margin,
        "outside": np.nextafter(family_margin, np.inf),
    }[relation]
    candidates = {
        tbp: _candidate_pair(tbp, 0.0, same_margin * 2.0),
        spy: _candidate_pair(spy, gap, gap + same_margin * 2.0),
    }

    descriptor, issue = _select(tbp, candidates)

    if relation == "outside":
        assert descriptor == _descriptor(tbp, 1)
        assert issue is None
    else:
        assert descriptor is None
        assert issue == stereo.StereoValidationIssue(
            code=stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
            message=(
                "TBP/SPY coordinate geometry is ambiguous within the "
                "3-degree family margin"
            ),
            record_index=7,
            atom_indices=(4, 8, 15, 16, 23, 42),
        )


def test_both_five_coordinate_families_outside_acceptance_are_degenerate():
    tbp = stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    spy = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    cutoff = _GOLDEN["policy_radians"]["acceptance"]
    same_margin = _GOLDEN["policy_radians"]["same_family_margin"]
    above = np.nextafter(cutoff, np.inf)
    candidates = {
        tbp: _candidate_pair(tbp, above, above + same_margin * 2.0),
        spy: _candidate_pair(
            spy,
            above + radians(4.0),
            above + radians(8.0),
        ),
    }

    descriptor, issue = _select(tbp, candidates)

    assert descriptor is None
    assert issue == stereo.StereoValidationIssue(
        code=stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
        message=(
            "TBP/SPY coordinate geometry exceeds the 12-degree angular RMS "
            "acceptance cutoff in both families"
        ),
        record_index=7,
        atom_indices=(4, 8, 15, 16, 23, 42),
    )


@pytest.mark.parametrize("relation", ("equal", "outside"))
def test_alternate_better_side_of_3_degree_boundary_never_migrates(relation):
    tbp = stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    spy = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    family_margin = _GOLDEN["policy_radians"]["five_coordinate_margin"]
    same_margin = _GOLDEN["policy_radians"]["same_family_margin"]
    gap = {
        "equal": family_margin,
        "outside": np.nextafter(family_margin, np.inf),
    }[relation]
    candidates = {
        tbp: _candidate_pair(tbp, gap, gap + same_margin * 2.0),
        spy: _candidate_pair(spy, 0.0, same_margin * 2.0),
    }

    descriptor, issue = _select(tbp, candidates)

    assert descriptor is None
    assert issue is not None
    if relation == "equal":
        assert issue.code is stereo.StereoIssueCode.DEGENERATE_GEOMETRY
        assert issue.message == (
            "TBP/SPY coordinate geometry is ambiguous within the "
            "3-degree family margin"
        )
    else:
        assert issue.code is stereo.StereoIssueCode.COORDINATE_CONTRADICTION
        assert issue.message == (
            "declared TBP family contradicts coordinate-preferred SPY family"
        )


def test_strong_alternate_family_is_a_contradiction_without_migration():
    tbp = stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    spy = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    same_margin = _GOLDEN["policy_radians"]["same_family_margin"]
    candidates = {
        tbp: _candidate_pair(tbp, radians(7.0), radians(11.0)),
        spy: _candidate_pair(spy, radians(1.0), radians(5.0)),
    }

    descriptor, issue = _select(tbp, candidates)

    assert same_margin == radians(2.0)
    assert descriptor is None
    assert issue == stereo.StereoValidationIssue(
        code=stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
        message="declared TBP family contradicts coordinate-preferred SPY family",
        record_index=7,
        atom_indices=(4, 8, 15, 16, 23, 42),
    )


def test_alternate_family_precedes_declared_numeric_index_ambiguity():
    tbp = stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    spy = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    candidates = {
        tbp: _candidate_pair(tbp, radians(8.0), radians(9.0)),
        spy: _candidate_pair(spy, 0.0, radians(4.0)),
    }

    descriptor, issue = _select(tbp, candidates)

    assert descriptor is None
    assert issue == stereo.StereoValidationIssue(
        code=stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
        message="declared TBP family contradicts coordinate-preferred SPY family",
        record_index=7,
        atom_indices=(4, 8, 15, 16, 23, 42),
    )


@pytest.mark.parametrize(
    ("declared", "observed", "message"),
    (
        (
            stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL,
            stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL,
            "declared TBP family contradicts coordinate-preferred SPY family",
        ),
        (
            stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL,
            stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL,
            "declared SPY family contradicts coordinate-preferred TBP family",
        ),
    ),
)
def test_real_ideal_tbp_spy_geometries_contradict_the_other_family(
    declared,
    observed,
    message,
):
    observed_value = (
        _GOLDEN["five_coordinate_cross"]["witness"][observed.value]
    )
    directions = _independent_directions(observed, observed_value)
    coordinates = _coordinates(directions)
    site = _site(declared, 1)

    descriptor, issue = stereo._gate7_coordination_coordinate_descriptor(
        site,
        coordinates,
    )

    assert descriptor is None
    assert issue == stereo.StereoValidationIssue(
        code=stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
        message=message,
        record_index=7,
        atom_indices=(0, 1, 2, 3, 4, 5),
    )


def test_private_coordinate_checkpoint_does_not_cut_over_public_gate7_semantics():
    family = stereo._Level2DescriptorFamily.SQUARE_PLANAR
    coordinates = _coordinates(_independent_directions(family, 1))
    molecule, payload = _coordination_molecule(family, coordinates)

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    assert not validation.is_valid
    assert tuple(issue.code for issue in validation.issues) == (
        stereo.StereoIssueCode.UNSUPPORTED,
    )
    assert assignment.payload == payload
    assert assignment.unresolved == validation.issues
    assert assignment.derived == ()
    assert assignment.payload.records[0].descriptor == "hotpot.stereo/2/SP/1"
    assert all(
        issue.code
        not in (
            stereo.StereoIssueCode.DEGENERATE_GEOMETRY,
            stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
        )
        for issue in (*validation.issues, *assignment.unresolved)
    )
    assert stereo.get_stereo_payload(molecule) is payload
