import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


_DATA_DIRECTORY = Path(__file__).parents[1] / "data" / "stereo"
_GOLDEN = json.loads(
    (_DATA_DIRECTORY / "level2_gate7_spy_algebra_golden.json").read_text()
)
_FAMILY = stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL
_IDEAL_SITES = np.array(
    (
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
        (-1, 0, 0),
        (0, -1, 0),
    ),
    dtype=np.int64,
)


def _normalize(configuration):
    apex, *base = configuration
    return min(
        (apex, *base[offset:], *base[:offset])
        for offset in range(4)
    )


def _independent_representatives():
    return tuple(
        sorted(
            {
                _normalize(configuration)
                for configuration in permutations(range(5))
            }
        )
    )


def _carriers():
    return tuple(
        stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)
        for row in (11, 3, 17, 5, 23)
    )


def _parsed(index):
    result = stereo._parse_level2_descriptor(
        f"hotpot.stereo/2/SPY/{index}"
    )
    assert result.issues == ()
    assert result.descriptor is not None
    return result.descriptor


def _independent_transform(index, source, reordered, representatives):
    site_atoms = tuple(
        source[position].atom for position in representatives[index - 1]
    )
    transformed = tuple(
        next(
            new_position
            for new_position, carrier in enumerate(reordered)
            if carrier.atom == atom
        )
        for atom in site_atoms
    )
    return representatives.index(_normalize(transformed)) + 1


def _induced_site_permutation(matrix):
    transformed = _IDEAL_SITES @ matrix.T
    return tuple(
        int(np.flatnonzero(np.all(_IDEAL_SITES == vector, axis=1))[0])
        for vector in transformed
    )


def _spy_molecule():
    molecule = Molecule()
    for atomic_number in (26, 7, 8, 9, 15, 16):
        molecule._create_atom(atomic_number=atomic_number)
    for donor in range(1, 6):
        molecule._add_bond(0, donor, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)
    record = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        carriers=tuple(
            stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)
            for row in range(1, 6)
        ),
        descriptor="hotpot.stereo/2/SPY/1",
        group=None,
        source_namespace="gate7-spy-algebra-test",
        source_code="hotpot.stereo/2/SPY/1",
    )
    payload = stereo.StereoPayload(1, (record,), (), ())
    stereo.set_stereo_payload(molecule, payload)
    return molecule, payload


def test_gate7_has_an_independent_schema_and_preserves_gate1_golden_bytes():
    assert _GOLDEN["schema"] == (
        "hotpot.stereo.level2-gate7-spy-algebra-golden/1"
    )
    gate1_bytes = (_DATA_DIRECTORY / "level2_algebra_golden.json").read_bytes()
    assert hashlib.sha256(gate1_bytes).hexdigest() == _GOLDEN[
        "gate1_golden_sha256"
    ]


def test_frozen_site_order_and_c4_permutations_follow_proper_rotations():
    assert _GOLDEN["ideal_site_order"] == ["+z", "+x", "+y", "-x", "-y"]
    rotations = (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    )
    induced = []
    for values in rotations:
        rotation = np.array(values, dtype=np.int64)
        assert round(float(np.linalg.det(rotation))) == 1
        induced.append(_induced_site_permutation(rotation))

    expected = tuple(map(tuple, _GOLDEN["proper_c4_rotations"]))
    # Active and pullback permutations are inverse conventions. C4 contains
    # both directions, so the frozen orbit is identical under either one.
    assert set(induced) == set(expected)
    assert stereo._SQUARE_PYRAMIDAL_ROTATIONS == expected


def test_all_120_arrangements_form_30_disjoint_size_four_orbits():
    representatives = _independent_representatives()
    assert representatives == tuple(map(tuple, _GOLDEN["representatives"]))
    assert representatives == stereo._COORDINATION_REPRESENTATIVES[_FAMILY]
    assert len(representatives) == 30

    orbit_union = set()
    for representative in representatives:
        apex, *base = representative
        orbit = {
            (apex, *base[offset:], *base[:offset])
            for offset in range(4)
        }
        assert len(orbit) == 4
        assert not orbit_union.intersection(orbit)
        orbit_union.update(orbit)
    assert orbit_union == set(permutations(range(5)))


def test_all_30_spy_indices_parse_and_render():
    assert _GOLDEN["descriptor_range"] == list(range(1, 31))
    for index in _GOLDEN["descriptor_range"]:
        descriptor = _parsed(index)
        assert stereo._render_level2_descriptor(descriptor) == (
            f"hotpot.stereo/2/SPY/{index}"
        )


@pytest.mark.parametrize("token", ("0", "31", "01", "unsupported", "-1"))
def test_invalid_spy_indices_are_rejected(token):
    result = stereo._parse_level2_descriptor(
        f"hotpot.stereo/2/SPY/{token}"
    )
    assert result.descriptor is None
    assert tuple(issue.code for issue in result.issues) == (
        stereo.StereoIssueCode.INVALID_DESCRIPTOR,
    )


def test_carrier_swap_golden_covers_every_spy_descriptor():
    source = _carriers()
    reordered = (source[1], source[0], *source[2:])
    actual = []
    for index in _GOLDEN["descriptor_range"]:
        result = stereo._transform_level2_descriptor(
            _parsed(index),
            source,
            reordered,
        )
        assert result.issues == ()
        actual.append(result.descriptor.value)
    assert actual == _GOLDEN["carrier_swap_01"]


def test_all_3600_carrier_reorderings_match_the_independent_oracle():
    representatives = _independent_representatives()
    source = _carriers()
    for index in _GOLDEN["descriptor_range"]:
        descriptor = _parsed(index)
        for reordering in permutations(range(5)):
            reordered = tuple(source[position] for position in reordering)
            result = stereo._transform_level2_descriptor(
                descriptor,
                source,
                reordered,
            )
            assert result.issues == ()
            assert result.descriptor.value == _independent_transform(
                index,
                source,
                reordered,
                representatives,
            )
            restored = stereo._transform_level2_descriptor(
                result.descriptor,
                reordered,
                source,
            )
            assert restored.issues == ()
            assert restored.descriptor.value == index


def test_yz_reflection_pairs_all_spy_descriptors_without_collapsing_them():
    representatives = _independent_representatives()
    reflection_matrix = np.diag((-1, 1, 1))
    assert round(float(np.linalg.det(reflection_matrix))) == -1
    reflection = (0, 3, 2, 1, 4)
    assert _induced_site_permutation(reflection_matrix) == reflection
    mirror = tuple(
        representatives.index(
            _normalize(tuple(representative[position] for position in reflection))
        )
        + 1
        for representative in representatives
    )

    assert mirror == tuple(_GOLDEN["mirror_yz"])
    assert all(mapped != index for index, mapped in enumerate(mirror, 1))
    assert all(mirror[mapped - 1] == index for index, mapped in enumerate(mirror, 1))
    assert reflection not in stereo._SQUARE_PYRAMIDAL_ROTATIONS


def test_spy_algebra_is_connected_to_public_graph_semantics():
    molecule, payload = _spy_molecule()
    target = molecule.copy()

    validation = stereo.validate_stereochemistry(molecule)
    assignment = stereo.assign_stereochemistry(molecule)
    alignment = stereo.align_stereochemistry(
        molecule,
        target,
        dict(enumerate(range(len(molecule.atoms)))),
    )

    assert validation == stereo.StereoValidationReport(True, ())
    assert assignment.payload is payload
    assert assignment.unresolved == ()
    assert assignment.derived == ()
    assert alignment.is_equivalent
    assert alignment.aligned_payload == payload
    assert alignment.issues == ()
    assert stereo.get_stereo_payload(molecule) is payload
