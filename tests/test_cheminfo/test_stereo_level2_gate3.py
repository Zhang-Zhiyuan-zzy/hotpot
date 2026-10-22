from __future__ import annotations

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


def _molecule(atomic_numbers, bonds, *, implicit_hydrogens=None):
    molecule = Molecule()
    hydrogen_counts = (
        (0,) * len(atomic_numbers)
        if implicit_hydrogens is None
        else implicit_hydrogens
    )
    for atomic_number, hydrogen_count in zip(
        atomic_numbers,
        hydrogen_counts,
    ):
        molecule.create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=hydrogen_count,
        )
    for begin, end, order in bonds:
        molecule.add_bond(begin, end, order)
    return molecule


def _atom(atom):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, atom)


_H = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)


def _record(anchor, carriers, descriptor):
    return stereo.StereoRecord(
        anchor=anchor,
        carriers=tuple(carriers),
        descriptor=descriptor,
        group=None,
        source_namespace="hotpot.stereo/2",
        source_code=descriptor,
    )


def _tetrahedral(center, carriers, value="even"):
    return _record(
        stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (center,)),
        carriers,
        f"hotpot.stereo/2/TH/{value}",
    )


def _double_bond(begin, end, carriers, value="together"):
    return _record(
        stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (begin, end)),
        carriers,
        f"hotpot.stereo/2/DB/{value}",
    )


def _set_records(molecule, *records):
    payload = stereo.StereoPayload(1, tuple(records), (), ())
    stereo.set_stereo_payload(molecule, payload)
    return payload


def _tetrahedral_molecule():
    return _molecule(
        (14, 6, 7, 8),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
        implicit_hydrogens=(1, 0, 0, 0),
    )


def _alkene_molecule():
    return _molecule(
        (6, 6, 9, 17, 7, 8),
        (
            (0, 1, 2.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 4, 1.0),
            (1, 5, 1.0),
        ),
    )


def test_identity_mapping_is_equivalent_and_does_not_mutate_inputs():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    record = _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H))
    reference_payload = _set_records(reference, record)
    target_payload = _set_records(target, record)
    atom_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    mapping_snapshot = atom_mapping.copy()

    report = stereo.align_stereochemistry(reference, target, atom_mapping)

    assert report.is_equivalent
    assert report.issues == ()
    assert report.aligned_payload.records == (record,)
    assert stereo.get_stereo_payload(reference) is reference_payload
    assert stereo.get_stereo_payload(target) is target_payload
    assert atom_mapping == mapping_snapshot


def test_tetrahedral_odd_carrier_permutation_transforms_parity():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    _set_records(
        reference,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H), "even"),
    )
    _set_records(
        target,
        _tetrahedral(0, (_atom(2), _atom(1), _atom(3), _H), "odd"),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        {0: 0, 1: 1, 2: 2, 3: 3},
    )

    assert report.is_equivalent
    assert report.aligned_payload.records[0].descriptor == "hotpot.stereo/2/TH/odd"
    assert report.aligned_payload.records[0].carriers == (
        _atom(2),
        _atom(1),
        _atom(3),
        _H,
    )


def test_nonidentity_atom_mapping_relabels_anchor_and_carriers():
    reference = _tetrahedral_molecule()
    target = _molecule(
        (8, 7, 6, 14),
        ((3, 0, 1.0), (3, 1, 1.0), (3, 2, 1.0)),
        implicit_hydrogens=(0, 0, 0, 1),
    )
    _set_records(
        reference,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H)),
    )
    target_record = _tetrahedral(3, (_atom(2), _atom(1), _atom(0), _H))
    _set_records(target, target_record)

    report = stereo.align_stereochemistry(
        reference,
        target,
        {0: 3, 1: 2, 2: 1, 3: 0},
    )

    assert report.is_equivalent
    assert report.aligned_payload.records[0].anchor == target_record.anchor
    assert report.aligned_payload.records[0].carriers == target_record.carriers


def test_double_bond_single_endpoint_swap_transforms_relation():
    reference = _alkene_molecule()
    target = _alkene_molecule()
    _set_records(
        reference,
        _double_bond(0, 1, tuple(map(_atom, (2, 3, 4, 5))), "together"),
    )
    _set_records(
        target,
        _double_bond(0, 1, tuple(map(_atom, (3, 2, 4, 5))), "opposite"),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        {atom: atom for atom in range(6)},
    )

    assert report.is_equivalent
    assert report.aligned_payload.records[0].descriptor == (
        "hotpot.stereo/2/DB/opposite"
    )


def test_double_bond_anchor_reversal_requires_and_accepts_pair_reversal():
    reference = _alkene_molecule()
    target = _alkene_molecule()
    _set_records(
        reference,
        _double_bond(0, 1, tuple(map(_atom, (2, 3, 4, 5)))),
    )
    target_record = _double_bond(
        1,
        0,
        tuple(map(_atom, (4, 5, 2, 3))),
    )
    _set_records(target, target_record)

    report = stereo.align_stereochemistry(
        reference,
        target,
        {atom: atom for atom in range(6)},
    )

    assert report.is_equivalent
    assert report.aligned_payload.records[0].anchor == target_record.anchor
    assert report.aligned_payload.records[0].carriers == target_record.carriers


def test_double_bond_endpoint_scope_allows_virtual_h_and_shared_ring_atom():
    reference = _molecule(
        (6, 6, 7),
        ((0, 1, 2.0), (0, 2, 1.0), (1, 2, 1.0)),
        implicit_hydrogens=(1, 1, 0),
    )
    target = _molecule(
        (6, 6, 7),
        ((0, 1, 2.0), (0, 2, 1.0), (1, 2, 1.0)),
        implicit_hydrogens=(1, 1, 0),
    )
    carriers = (_atom(2), _H, _atom(2), _H)
    _set_records(reference, _double_bond(0, 1, carriers))
    _set_records(target, _double_bond(1, 0, carriers))

    report = stereo.align_stereochemistry(
        reference,
        target,
        {0: 0, 1: 1, 2: 2},
    )

    assert report.is_equivalent
    assert report.issues == ()


def test_incomplete_mapping_reports_uncovered_stereo_rows():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    record = _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H))
    _set_records(reference, record)
    _set_records(target, record)

    report = stereo.align_stereochemistry(
        reference,
        target,
        {0: 0, 1: 1, 2: 2},
    )

    assert not report.is_equivalent
    assert [issue.message for issue in report.issues] == [
        "atom mapping does not cover reference stereo rows (3,)",
        "atom mapping does not cover target stereo rows (3,)",
    ]
    assert all(
        issue.code is stereo.StereoIssueCode.INCOMPLETE_MAPPING
        for issue in report.issues
    )


def test_many_to_one_and_out_of_range_mapping_are_explicit():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    record = _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H))
    _set_records(reference, record)
    _set_records(target, record)

    many_to_one = stereo.align_stereochemistry(
        reference,
        target,
        {0: 0, 1: 1, 2: 1, 3: 3},
    )
    out_of_range = stereo.align_stereochemistry(
        reference,
        target,
        {0: 0, 1: 1, 2: 2, 3: 9},
    )

    assert any("many-to-one" in issue.message for issue in many_to_one.issues)
    assert any(
        "target stereo rows (2,)" in issue.message
        for issue in many_to_one.issues
    )
    assert out_of_range.issues[0].message == (
        "target atom row 9 is outside molecule range 0..3"
    )
    assert not many_to_one.is_equivalent
    assert not out_of_range.is_equivalent


def test_missing_and_extra_stereo_records_are_reported_separately():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    record = _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H))
    _set_records(reference, record)
    _set_records(target)
    mapping = {atom: atom for atom in range(4)}

    missing = stereo.align_stereochemistry(reference, target, mapping)

    _set_records(reference)
    _set_records(target, record)
    extra = stereo.align_stereochemistry(reference, target, mapping)

    assert [issue.message for issue in missing.issues] == [
        "mapped reference stereo record has no target record",
    ]
    assert [issue.message for issue in extra.issues] == [
        "target stereo record has no mapped reference record",
    ]
    assert not missing.is_equivalent
    assert not extra.is_equivalent


def test_duplicate_target_record_and_invalid_target_graph_are_identified():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    record = _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H))
    _set_records(reference, record)
    _set_records(target, record, record)

    duplicate = stereo.align_stereochemistry(
        reference,
        target,
        {atom: atom for atom in range(4)},
    )

    invalid_target = _molecule(
        (14, 6, 7, 8, 9),
        ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)),
    )
    _set_records(
        invalid_target,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(4), _H)),
    )
    invalid_graph = stereo.align_stereochemistry(
        reference,
        invalid_target,
        {0: 0, 1: 1, 2: 2, 3: 4},
    )

    assert any(
        "multiple Level-2 geometry records" in issue.message
        for issue in duplicate.issues
    )
    assert any(
        issue.message.startswith("target: carrier atom row 4 is not adjacent")
        for issue in invalid_graph.issues
    )
    assert not duplicate.is_equivalent
    assert not invalid_graph.is_equivalent


def test_descriptor_mismatch_is_non_equivalent_and_aligned_value_is_stable():
    reference = _tetrahedral_molecule()
    target = _tetrahedral_molecule()
    _set_records(
        reference,
        _tetrahedral(0, (_atom(1), _atom(2), _atom(3), _H), "even"),
    )
    _set_records(
        target,
        _tetrahedral(0, (_atom(2), _atom(1), _atom(3), _H), "even"),
    )

    report = stereo.align_stereochemistry(
        reference,
        target,
        {atom: atom for atom in range(4)},
    )

    assert not report.is_equivalent
    assert report.aligned_payload.records[0].descriptor == "hotpot.stereo/2/TH/odd"
    assert report.issues == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.INVALID_DESCRIPTOR,
            "mapped descriptor hotpot.stereo/2/TH/odd does not match target "
            "descriptor hotpot.stereo/2/TH/even",
            record_index=0,
            atom_indices=(0, 2, 1, 3),
        ),
    )
