from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Bond, Molecule

from hotpot.cheminfo.operations import stereochemistry as stereo_operations
from hotpot.cheminfo.rules.chemistry.stereochemistry import algebra as stereo_algebra
from hotpot.cheminfo.rules.chemistry.stereochemistry import coordinates as stereo_coordinates


_FAMILIES = (
    (stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR, 4, 3),
    (stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL, 5, 20),
    (stereo_algebra._Level2DescriptorFamily.SQUARE_PYRAMIDAL, 5, 30),
    (stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL, 6, 30),
)
_ALL_DESCRIPTORS = tuple(
    (family, carrier_count, value)
    for family, carrier_count, cardinality in _FAMILIES
    for value in range(1, cardinality + 1)
)


def _molecule(
    atom_count,
    edges,
    *,
    implicit_hydrogens=None,
    atomic_numbers=None,
):
    molecule = Molecule()
    implicit_hydrogens = implicit_hydrogens or (0,) * atom_count
    for row, hydrogen_count in enumerate(implicit_hydrogens):
        molecule._create_atom(
            atomic_number=(
                atomic_numbers[row]
                if atomic_numbers is not None
                else (26 if row == 0 else 7)
            ),
            implicit_hydrogens=hydrogen_count,
        )
    for first, second in edges:
        molecule._add_bond(first, second, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _record(family, carriers, *, value=1, center=0):
    descriptor = f"hotpot.stereo/2/{family.value}/{value}"
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(
            stereo.StereoAnchorKind.ATOM,
            (center,),
        ),
        carriers=tuple(
            stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)
            for row in carriers
        ),
        descriptor=descriptor,
        group=None,
        source_namespace="gate7-public-test",
        source_code=descriptor,
    )


def _set_record(molecule, record, *, atom_facts=()):
    return _set_records(molecule, (record,), atom_facts=atom_facts)


def _set_records(molecule, records, *, atom_facts=()):
    payload = stereo.StereoPayload(1, tuple(records), tuple(atom_facts), ())
    stereo.set_stereo_payload(molecule, payload)
    return payload


def _coordination_molecule(family, carrier_count, *, value=1):
    carriers = tuple(range(1, carrier_count + 1))
    molecule = _molecule(
        carrier_count + 1,
        tuple((0, carrier) for carrier in carriers),
    )
    payload = _set_record(molecule, _record(family, carriers, value=value))
    return molecule, payload


def _ideal_coordinates(family, value):
    descriptor = stereo_algebra._Level2Descriptor(family, value)
    directions = stereo_coordinates._gate7_ideal_carrier_directions(descriptor)
    center = np.array((4.0, -3.0, 7.0))
    radii = np.linspace(0.8, 2.4, len(directions))
    return np.vstack((center, center + radii[:, None] * directions))


def _gate6_coordination_descendant_molecule(*, declared):
    molecule = _molecule(
        11,
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 8),
            (2, 9),
            (2, 10),
        ),
        atomic_numbers=(14, 26, 26, 9, 17, 7, 8, 16, 7, 8, 16),
    )
    records = [
        _record(
            stereo_algebra._Level2DescriptorFamily.TETRAHEDRAL,
            (1, 2, 3, 4),
            value="even",
        )
    ]
    if declared:
        records.append(
            _record(
                stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR,
                (0, 5, 6, 7),
                center=1,
            )
        )
    _set_records(molecule, records)
    return molecule


@pytest.mark.parametrize(
    ("family", "carrier_count", "value"),
    _ALL_DESCRIPTORS,
    ids=lambda value: value.value if hasattr(value, "value") else str(value),
)
def test_public_gate7_accepts_and_assigns_each_declared_family(
    family,
    carrier_count,
    value,
):
    molecule, payload = _coordination_molecule(
        family,
        carrier_count,
        value=value,
    )
    coordinates = _ideal_coordinates(family, value)

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    assert validation == stereo.StereoValidationReport(True, ())
    assert assignment.payload == payload
    assert assignment.unresolved == ()
    assert assignment.derived == ()
    assert stereo.get_stereo_payload(molecule) is payload


def test_public_assignment_replaces_only_the_declared_numeric_index():
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, payload = _coordination_molecule(family, 4, value=1)
    coordinates = _ideal_coordinates(family, 3)

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    assert len(validation.issues) == 1
    assert validation.issues[0].code is (
        stereo.StereoIssueCode.COORDINATE_CONTRADICTION
    )
    assert assignment.unresolved == validation.issues
    assert assignment.payload.records[0].descriptor == "hotpot.stereo/2/SP/3"
    assert assignment.derived == ()
    assert stereo.get_stereo_payload(molecule) is payload


def test_five_coordinate_contradiction_never_migrates_descriptor_family():
    family = stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    alternate = stereo_algebra._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    molecule, payload = _coordination_molecule(family, 5, value=1)
    coordinates = _ideal_coordinates(alternate, 1)

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )

    assert validation.issues == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.COORDINATE_CONTRADICTION,
            "declared TBP family contradicts coordinate-preferred SPY family",
            record_index=0,
            atom_indices=(0, 1, 2, 3, 4, 5),
        ),
    )
    assert assignment.payload == payload
    assert assignment.unresolved == validation.issues
    assert assignment.derived == ()


@pytest.mark.parametrize(("family", "carrier_count", "_"), _FAMILIES)
def test_generic_transform_and_public_alignment_cover_each_family(
    family,
    carrier_count,
    _,
):
    reference, _ = _coordination_molecule(family, carrier_count)
    source = stereo.get_stereo_payload(reference).records[0]
    reordered = replace(
        source,
        carriers=(source.carriers[1], source.carriers[0], *source.carriers[2:]),
    )
    descriptor = stereo_algebra._parse_level2_descriptor(source.descriptor).descriptor
    expected = stereo_algebra._transform_level2_descriptor(
        descriptor,
        source.carriers,
        reordered.carriers,
        source_anchor=source.anchor,
        reordered_anchor=reordered.anchor,
    )
    assert expected.issues == ()
    transformed = stereo_algebra._transform_supported_record(
        descriptor,
        source,
        reordered,
    )
    assert transformed == expected

    target = reference.copy()
    target_record = replace(
        reordered,
        descriptor=stereo_algebra._render_level2_descriptor(expected.descriptor),
    )
    _set_record(target, target_record)
    alignment = stereo.align_stereochemistry(
        reference,
        target,
        dict(enumerate(range(len(reference.atoms)))),
    )

    assert alignment.is_equivalent
    assert alignment.issues == ()
    assert alignment.aligned_payload.records[0].descriptor == (
        target_record.descriptor
    )


@pytest.mark.parametrize("source_has_bridge", (False, True))
def test_public_alignment_rejects_coordination_chelate_split_or_merge(
    source_has_bridge,
):
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    spokes = tuple((0, carrier) for carrier in range(1, 5))
    bridge = ((1, 5), (5, 2))
    source_edges = spokes + (bridge if source_has_bridge else ())
    target_edges = spokes + (bridge if not source_has_bridge else ())
    source = _molecule(6, source_edges)
    target = _molecule(6, target_edges)
    record = _record(family, (1, 2, 3, 4))
    _set_record(source, record)
    _set_record(target, record)

    alignment = stereo.align_stereochemistry(
        source,
        target,
        dict(enumerate(range(6))),
    )

    assert not alignment.is_equivalent
    assert alignment.issues == (
        stereo.StereoValidationIssue(
            stereo.StereoIssueCode.INVALID_DESCRIPTOR,
            "mapped coordination chelate partition does not match target partition",
            record_index=0,
            atom_indices=(0, 1, 2, 3, 4),
        ),
    )


def test_public_alignment_maps_coordination_chelate_partition_before_comparison():
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    source_edges = (
        *((0, carrier) for carrier in range(1, 5)),
        (1, 5),
        (5, 2),
    )
    target_edges = (
        *((0, carrier) for carrier in range(1, 5)),
        (3, 5),
        (5, 4),
    )
    source = _molecule(6, source_edges)
    target = _molecule(6, target_edges)
    _set_record(source, _record(family, (1, 2, 3, 4)))
    _set_record(target, _record(family, (3, 4, 1, 2)))
    mapping = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 5}

    alignment = stereo.align_stereochemistry(
        source,
        target,
        mapping,
    )

    assert alignment.is_equivalent
    assert alignment.issues == ()


def test_public_entry_points_project_each_coordination_graph_once(monkeypatch):
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    reference, _ = _coordination_molecule(family, 4)
    target = reference.copy()
    original = stereo_operations._gate6_projection
    calls = []

    def tracked_projection(molecule, **context):
        calls.append(molecule)
        return original(molecule, **context)

    monkeypatch.setattr(stereo_operations, "_gate6_projection", tracked_projection)

    validation = stereo.validate_stereochemistry(reference)
    assert validation.is_valid
    assert calls == [reference]

    calls.clear()
    assignment = stereo.assign_stereochemistry(reference)
    assert assignment.unresolved == ()
    assert calls == [reference]

    calls.clear()
    alignment = stereo.align_stereochemistry(
        reference,
        target,
        dict(enumerate(range(5))),
    )
    assert alignment.is_equivalent
    assert calls == [reference, target]


def test_public_alignment_never_coerces_five_coordinate_families():
    reference_family = stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL
    target_family = stereo_algebra._Level2DescriptorFamily.SQUARE_PYRAMIDAL
    reference, _ = _coordination_molecule(reference_family, 5)
    target, _ = _coordination_molecule(target_family, 5)

    alignment = stereo.align_stereochemistry(
        reference,
        target,
        dict(enumerate(range(6))),
    )

    assert not alignment.is_equivalent
    assert tuple(issue.code for issue in alignment.issues) == (
        stereo.StereoIssueCode.INVALID_ANCHOR,
        stereo.StereoIssueCode.INVALID_ANCHOR,
    )
    assert all("stereo record" in issue.message for issue in alignment.issues)


@pytest.mark.parametrize("cross_family", (False, True))
def test_same_center_coordination_records_reject_every_participant(cross_family):
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, _ = _coordination_molecule(family, 4)
    first = stereo.get_stereo_payload(molecule).records[0]
    second = (
        _record(
            stereo_algebra._Level2DescriptorFamily.TETRAHEDRAL,
            (1, 2, 3, 4),
            value="even",
        )
        if cross_family
        else replace(first, descriptor="hotpot.stereo/2/SP/2")
    )
    _set_records(molecule, (first, second))

    report = stereo.validate_stereochemistry(molecule)

    assert tuple(
        (issue.record_index, issue.code, issue.message)
        for issue in report.issues
    ) == (
        (
            0,
            stereo.StereoIssueCode.INVALID_ANCHOR,
            "atom center row 0 has multiple Level-2 geometry records",
        ),
        (
            1,
            stereo.StereoIssueCode.INVALID_ANCHOR,
            "atom center row 0 has multiple Level-2 geometry records",
        ),
    )


def _two_center_molecule(*, shared_donor):
    first = (1, 2, 3, 4)
    second = (1, 6, 7, 8) if shared_donor else (6, 7, 8, 9)
    atom_count = 9 if shared_donor else 10
    edges = tuple((0, row) for row in first) + tuple(
        (5, row) for row in second
    )
    molecule = _molecule(atom_count, edges)
    records = (
        _record(stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR, first),
        _record(
            stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR,
            second,
            center=5,
        ),
    )
    _set_records(molecule, records)
    return molecule


def test_public_gate7_accepts_independent_coordination_centers():
    report = stereo.validate_stereochemistry(
        _two_center_molecule(shared_donor=False)
    )

    assert report == stereo.StereoValidationReport(True, ())


def test_public_gate7_diagnoses_every_shared_donor_center():
    report = stereo.validate_stereochemistry(
        _two_center_molecule(shared_donor=True)
    )

    assert tuple(issue.record_index for issue in report.issues) == (0, 1)
    assert all(
        issue.code is stereo.StereoIssueCode.UNSUPPORTED
        and "participates in coupled multicenter coordination" in issue.message
        for issue in report.issues
    )


def test_public_gate7_rejects_current_implicit_hydrogen_in_first_shell():
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule = _molecule(
        5,
        tuple((0, carrier) for carrier in range(1, 5)),
        implicit_hydrogens=(1, 0, 0, 0, 0),
    )
    _set_record(molecule, _record(family, (1, 2, 3, 4)))

    report = stereo.validate_stereochemistry(molecule)

    assert tuple((issue.code, issue.message) for issue in report.issues) == (
        (
            stereo.StereoIssueCode.INVALID_CARRIER,
            "SP carriers must exactly cover the represented first shell at "
            "center row 0",
        ),
    )


def test_public_gate7_ignores_source_only_hydrogen_ledger():
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, _ = _coordination_molecule(family, 4)
    record = stereo.get_stereo_payload(molecule).records[0]
    fact = stereo.AtomTransportRecord(
        atom=0,
        isotope=None,
        radical_electrons=None,
        current_hydrogens=stereo.HydrogenTransportState(0, 0, True),
        source_hydrogens=stereo.HydrogenTransportState(1, 0, False),
        source_namespace="gate7-public-test",
    )
    _set_record(molecule, record, atom_facts=(fact,))

    assert stereo.validate_stereochemistry(molecule) == (
        stereo.StereoValidationReport(True, ())
    )


def test_public_gate7_preserves_non_atom_and_group_boundaries():
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, _ = _coordination_molecule(family, 4)
    record = stereo.get_stereo_payload(molecule).records[0]
    group = stereo.StereoGroupRef(
        "gate7-public-test",
        stereo.StereoGroupKind.ABS,
        1,
    )
    non_atom = replace(
        record,
        carriers=(
            *record.carriers[:2],
            stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H),
            record.carriers[3],
        ),
    )

    _set_record(molecule, replace(record, group=group))
    grouped = stereo.validate_stereochemistry(molecule)
    _set_record(molecule, non_atom)
    implicit = stereo.validate_stereochemistry(molecule)

    assert tuple((issue.code, issue.message) for issue in grouped.issues) == (
        (
            stereo.StereoIssueCode.UNSUPPORTED,
            "SP enhanced-stereo groups are unsupported for coordination records",
        ),
    )
    assert tuple((issue.code, issue.message) for issue in implicit.issues) == (
        (
            stereo.StereoIssueCode.UNSUPPORTED,
            "SP carrier position 2 uses unsupported implicit_h; Gate 7 supports "
            "atom-point carriers only",
        ),
    )


def test_public_gate7_never_discovers_undeclared_coordination_centers():
    molecule = _molecule(5, tuple((0, carrier) for carrier in range(1, 5)))

    validation = stereo.validate_stereochemistry(molecule)
    assignment = stereo.assign_stereochemistry(molecule)

    assert validation == stereo.StereoValidationReport(True, ())
    assert assignment.payload == stereo.StereoPayload(1, (), (), ())
    assert assignment.unresolved == ()
    assert assignment.derived == ()


@pytest.mark.parametrize("malformation", ("parallel", "dative", "atom_fact"))
def test_coordination_only_payload_triggers_complete_gate6_projection(
    malformation,
):
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, _ = _coordination_molecule(family, 4)
    record = stereo.get_stereo_payload(molecule).records[0]
    if malformation == "parallel":
        molecule._hided_covalent_bonds.append(
            Bond(molecule.atoms[0], molecule.atoms[1], bond_order=1.0)
        )
    elif malformation == "dative":
        molecule = _molecule(5, ((0, 2), (0, 3), (0, 4)))
        molecule._add_bond(
            0,
            1,
            bond_order=1.0,
            bond_kind="dative",
            bond_direction="sideways",
        )
        molecule._update_topology(clear_conformers=False)
        _set_record(molecule, record)
    else:
        hydrogen = stereo.HydrogenTransportState(0, 0, True)
        fact = stereo.AtomTransportRecord(
            atom=0,
            isotope=None,
            radical_electrons=None,
            current_hydrogens=hydrogen,
            source_hydrogens=hydrogen,
            source_namespace="gate7-public-test",
        )
        _set_record(molecule, record, atom_facts=(fact, fact))

    report = stereo.validate_stereochemistry(molecule)

    assert not report.is_valid
    global_issues = tuple(
        issue for issue in report.issues if issue.record_index is None
    )
    assert len(global_issues) == 1
    expected_text = {
        "parallel": "parallel represented bonds",
        "dative": "unknown direction",
        "atom_fact": "duplicate atom transport facts",
    }
    assert expected_text[malformation] in global_issues[0].message


@pytest.mark.parametrize(
    ("descriptor", "local_code", "triggers_projection"),
    (
        (
            "hotpot.stereo/2/SP/999",
            stereo.StereoIssueCode.INVALID_DESCRIPTOR,
            True,
        ),
        (
            "hotpot.stereo/2/NEW/999",
            stereo.StereoIssueCode.UNSUPPORTED,
            False,
        ),
    ),
)
def test_declared_coordination_family_projects_even_with_invalid_numeric_value(
    descriptor,
    local_code,
    triggers_projection,
):
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, _ = _coordination_molecule(family, 4)
    record = replace(
        stereo.get_stereo_payload(molecule).records[0],
        descriptor=descriptor,
    )
    _set_record(molecule, record)
    molecule._hided_covalent_bonds.append(
        Bond(molecule.atoms[0], molecule.atoms[1], bond_order=1.0)
    )

    report = stereo.validate_stereochemistry(molecule)

    local_issues = tuple(
        issue for issue in report.issues if issue.record_index == 0
    )
    global_issues = tuple(
        issue for issue in report.issues if issue.record_index is None
    )
    assert tuple(issue.code for issue in local_issues) == (local_code,)
    assert len(global_issues) == int(triggers_projection)
    if triggers_projection:
        assert "parallel represented bonds" in global_issues[0].message


def test_gate7_public_cutover_is_atomic_across_main_entry_points():
    family = stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR
    molecule, payload = _coordination_molecule(family, 4, value=3)
    coordinates = _ideal_coordinates(family, 3)
    target = molecule.copy()

    validation = stereo.validate_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    assignment = stereo.assign_stereochemistry(
        molecule,
        coordinates=coordinates,
    )
    alignment = stereo.align_stereochemistry(
        molecule,
        target,
        dict(enumerate(range(5))),
    )
    source_record = payload.records[0]
    transformed = stereo_algebra._transform_supported_record(
        stereo_algebra._parse_level2_descriptor(source_record.descriptor).descriptor,
        source_record,
        source_record,
    )
    parent_without_coordination = stereo.validate_stereochemistry(
        _gate6_coordination_descendant_molecule(declared=False)
    )
    parent_with_coordination = stereo.validate_stereochemistry(
        _gate6_coordination_descendant_molecule(declared=True)
    )

    assert validation.is_valid
    assert assignment.payload == payload
    assert assignment.unresolved == ()
    assert assignment.derived == ()
    assert alignment.is_equivalent
    assert alignment.issues == ()
    assert transformed.descriptor == stereo_algebra._parse_level2_descriptor(
        source_record.descriptor
    ).descriptor
    assert transformed.issues == ()
    assert any(
        issue.code is stereo.StereoIssueCode.NON_STEREOGENIC
        and issue.record_index == 0
        for issue in parent_without_coordination.issues
    )
    assert parent_with_coordination == stereo.StereoValidationReport(True, ())
