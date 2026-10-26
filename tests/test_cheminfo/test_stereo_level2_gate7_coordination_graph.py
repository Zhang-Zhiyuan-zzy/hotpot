from dataclasses import replace

import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Bond, Molecule


_FAMILY_CASES = (
    ("SP", stereo._Level2DescriptorFamily.SQUARE_PLANAR, 4),
    ("TBP", stereo._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL, 5),
    ("SPY", stereo._Level2DescriptorFamily.SQUARE_PYRAMIDAL, 5),
    ("OH", stereo._Level2DescriptorFamily.OCTAHEDRAL, 6),
)


def _molecule(atomic_numbers, edges=(), *, implicit_hydrogens=None):
    molecule = Molecule()
    implicit_hydrogens = implicit_hydrogens or (0,) * len(atomic_numbers)
    for atomic_number, hydrogen_count in zip(
        atomic_numbers,
        implicit_hydrogens,
    ):
        molecule._create_atom(
            atomic_number=atomic_number,
            implicit_hydrogens=hydrogen_count,
        )
    for first, second in edges:
        molecule._add_bond(first, second, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _carrier(row, kind=stereo.StereoCarrierKind.ATOM):
    return stereo.StereoCarrier(
        kind,
        row if kind is stereo.StereoCarrierKind.ATOM else None,
    )


def _record(
    family,
    center,
    carriers,
    *,
    value=1,
    group=None,
    anchor_kind=stereo.StereoAnchorKind.ATOM,
):
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(anchor_kind, (center,)),
        carriers=tuple(
            carrier
            if isinstance(carrier, stereo.StereoCarrier)
            else _carrier(carrier)
            for carrier in carriers
        ),
        descriptor=f"hotpot.stereo/2/{family}/{value}",
        group=group,
        source_namespace="gate7-coordination-graph-test",
        source_code=f"hotpot.stereo/2/{family}/{value}",
    )


def _opaque_record(center):
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (center,)),
        carriers=(),
        descriptor="external.coordination/haptic/centroid",
        group=None,
        source_namespace="external-coordination-test",
        source_code="haptic-centroid",
    )


def _descriptor(record):
    result = stereo._parse_level2_descriptor(record.descriptor)
    assert result.issues == ()
    assert result.descriptor is not None
    return result.descriptor


def _payload(*records):
    return stereo.StereoPayload(1, tuple(records), (), ())


def _projected_adjacency(molecule):
    graph, issues = stereo._gate6_projection(molecule)
    assert issues == ()
    assert graph is not None
    return graph, stereo._gate7_represented_adjacency(graph)


def _validated_site(molecule, record, *, record_index=0):
    _, adjacency = _projected_adjacency(molecule)
    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=record_index,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )
    assert issues == ()
    assert site is not None
    return site, adjacency


def _issue_facts(issues):
    return tuple((issue.record_index, issue.code, issue.message) for issue in issues)


def test_represented_shell_unifies_active_and_both_hidden_bond_classes():
    molecule = _molecule((26, 7, 8, 9, 15), ((0, 1), (0, 2)))
    molecule._hided_metal_bonds.append(
        Bond(
            molecule.atoms[0],
            molecule.atoms[3],
            bond_order=1.0,
            bond_kind="dative",
            bond_direction="atom2_to_atom1",
        )
    )
    molecule._hided_covalent_bonds.append(
        Bond(molecule.atoms[0], molecule.atoms[4], bond_order=1.0)
    )
    record = _record("SP", 0, (1, 2, 3, 4))

    graph, adjacency = _projected_adjacency(molecule)
    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=7,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert graph.target_spec["edges"] == ((0, 1), (0, 2), (0, 3), (0, 4))
    assert adjacency == (
        frozenset((1, 2, 3, 4)),
        frozenset((0,)),
        frozenset((0,)),
        frozenset((0,)),
        frozenset((0,)),
    )
    assert issues == ()
    assert site is not None
    assert site.record_index == 7
    assert site.descriptor == _descriptor(record)
    assert site.center == 0
    assert site.carrier_rows == (1, 2, 3, 4)


@pytest.mark.parametrize(
    ("family", "expected_family", "carrier_count"),
    _FAMILY_CASES,
)
def test_each_coordination_family_accepts_one_exact_atom_shell(
    family,
    expected_family,
    carrier_count,
):
    carriers = tuple(range(1, carrier_count + 1))
    molecule = _molecule(
        (26, *((7,) * carrier_count)),
        tuple((0, carrier) for carrier in carriers),
    )
    record = _record(family, 0, carriers)

    site, _ = _validated_site(molecule, record)

    assert site.descriptor.family is expected_family
    assert site.center == 0
    assert site.carrier_rows == carriers


def test_coordination_carrier_count_has_one_exact_non_cascading_issue():
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, (1, 2, 3))
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=2,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert _issue_facts(issues) == (
        (
            2,
            stereo.StereoIssueCode.INVALID_CARRIER,
            "expected 4 ordered carriers, got 3",
        ),
    )


@pytest.mark.parametrize(
    "kind",
    (
        stereo.StereoCarrierKind.IMPLICIT_H,
        stereo.StereoCarrierKind.LONE_PAIR,
        stereo.StereoCarrierKind.VACANCY,
    ),
)
def test_coordination_non_atom_carrier_has_one_exact_issue(kind):
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    carriers = (_carrier(1), _carrier(2), _carrier(None, kind), _carrier(4))
    record = _record("SP", 0, carriers)
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=3,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert _issue_facts(issues) == (
        (
            3,
            stereo.StereoIssueCode.UNSUPPORTED,
            f"SP carrier position 2 uses unsupported {kind.value}; "
            "Gate 7 supports atom-point carriers only",
        ),
    )


@pytest.mark.parametrize(
    ("carriers", "message"),
    (
        (
            (
                _carrier(1),
                _carrier(2),
                stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, None),
                _carrier(4),
            ),
            "atom carrier at position 2 requires a non-negative atom row",
        ),
        (
            (1, 2, 2, 4),
            "duplicate carrier identity at position 2",
        ),
    ),
)
def test_malformed_atom_carriers_reuse_exact_level2_diagnostics(carriers, message):
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, carriers)
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=4,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert _issue_facts(issues) == (
        (4, stereo.StereoIssueCode.INVALID_CARRIER, message),
    )


@pytest.mark.parametrize(
    ("center", "carriers", "code", "message", "atom_indices"),
    (
        (
            9,
            (1, 2, 3, 4),
            stereo.StereoIssueCode.INVALID_ANCHOR,
            "anchor atom row 9 is outside molecule range 0..4",
            (9,),
        ),
        (
            0,
            (1, 2, 3, 9),
            stereo.StereoIssueCode.INVALID_CARRIER,
            "carrier atom row 9 is outside molecule range 0..4",
            (9,),
        ),
    ),
)
def test_coordination_atom_ranges_have_exact_non_cascading_issues(
    center,
    carriers,
    code,
    message,
    atom_indices,
):
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", center, carriers)
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=5,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert _issue_facts(issues) == ((5, code, message),)
    assert issues[0].atom_indices == atom_indices


@pytest.mark.parametrize(
    "kind",
    (
        stereo.StereoGroupKind.ABS,
        stereo.StereoGroupKind.AND,
        stereo.StereoGroupKind.OR,
    ),
)
def test_all_enhanced_group_kinds_have_the_exact_coordination_boundary(kind):
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    group = stereo.StereoGroupRef("gate7-test", kind, 11)
    record = _record("SP", 0, (1, 2, 3, 4), group=group)
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=6,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert _issue_facts(issues) == (
        (
            6,
            stereo.StereoIssueCode.UNSUPPORTED,
            "SP enhanced-stereo groups are unsupported for coordination records",
        ),
    )


def test_coordination_carriers_must_equal_the_complete_represented_shell():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 16),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, (1, 2, 3, 5))
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=8,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert _issue_facts(issues) == (
        (
            8,
            stereo.StereoIssueCode.INVALID_CARRIER,
            "SP carriers must exactly cover the represented first shell at "
            "center row 0",
        ),
    )


def test_current_implicit_hydrogen_is_part_of_the_represented_shell():
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
        implicit_hydrogens=(1, 0, 0, 0, 0),
    )
    record = _record("SP", 0, (1, 2, 3, 4))
    graph, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=8,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    virtual_hydrogen = graph.virtual_hydrogens[0][0]
    assert virtual_hydrogen == len(molecule.atoms)
    assert adjacency[0] == frozenset((1, 2, 3, 4, virtual_hydrogen))
    assert site is None
    assert _issue_facts(issues) == (
        (
            8,
            stereo.StereoIssueCode.INVALID_CARRIER,
            "SP carriers must exactly cover the represented first shell at "
            "center row 0",
        ),
    )
    assert issues[0].atom_indices == (0, 1, 2, 3, 4)
    assert all(row < len(molecule.atoms) for row in issues[0].atom_indices)


def test_source_hydrogen_ledger_does_not_enter_the_current_shell():
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, (1, 2, 3, 4))
    current_hydrogens = stereo.HydrogenTransportState(0, 0, True)
    source_hydrogens = stereo.HydrogenTransportState(1, 0, False)
    stereo.set_stereo_payload(
        molecule,
        stereo.StereoPayload(
            1,
            (record,),
            (
                stereo.AtomTransportRecord(
                    atom=0,
                    isotope=None,
                    radical_electrons=None,
                    current_hydrogens=current_hydrogens,
                    source_hydrogens=source_hydrogens,
                    source_namespace="source-ledger-only-test",
                ),
            ),
            (),
        ),
    )

    graph, adjacency = _projected_adjacency(molecule)
    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=0,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert graph.virtual_hydrogens[0] == ()
    assert adjacency[0] == frozenset((1, 2, 3, 4))
    assert issues == ()
    assert site is not None


def test_anchor_cannot_be_coerced_into_an_atom_carrier():
    molecule = _molecule(
        (26, 7, 8, 9, 15),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, (0, 2, 3, 4))
    _, adjacency = _projected_adjacency(molecule)

    site, issues = stereo._validate_gate7_coordination_record(
        record,
        _descriptor(record),
        record_index=9,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )

    assert site is None
    assert len(issues) == 1
    assert issues[0].record_index == 9
    assert issues[0].code is stereo.StereoIssueCode.INVALID_CARRIER


def test_every_same_center_geometry_record_is_rejected_without_a_winner():
    records = (
        _record("TH", 0, (1, 2, 3, 4), value="even"),
        _record("SP", 0, (1, 2, 3, 4), value=1),
        _record("SP", 0, (1, 2, 3, 4), value=2),
        _record("TBP", 0, (1, 2, 3, 4, 5), value=1),
    )
    payload = _payload(*records)
    descriptors = {
        index: _descriptor(record)
        for index, record in enumerate(records)
    }

    issues = stereo._gate7_same_center_issues(
        payload,
        descriptors,
        atom_count=6,
    )

    assert tuple(issue.record_index for issue in issues) == (0, 1, 2, 3)
    assert all(
        issue.code is stereo.StereoIssueCode.INVALID_ANCHOR
        for issue in issues
    )
    assert all(
        issue.message == "atom center row 0 has multiple Level-2 geometry records"
        for issue in issues
    )
    assert all(issue.atom_indices == (0,) for issue in issues)


def test_chelate_pairs_are_order_independent_and_remove_only_own_spokes():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 6),
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 5),
            (5, 2),
        ),
    )
    source = _record("SP", 0, (1, 2, 3, 4))
    reordered = replace(source, carriers=(
        source.carriers[3],
        source.carriers[1],
        source.carriers[0],
        source.carriers[2],
    ))

    source_site, adjacency = _validated_site(molecule, source)
    reordered_site, _ = _validated_site(molecule, reordered)

    # Row 5 can represent another center.  Its two spokes remain available
    # after only row 0's center-to-carrier edges are removed.
    assert adjacency[5] == frozenset((1, 2))
    assert stereo._gate7_chelate_pairs(source_site, adjacency) == frozenset(
        ((1, 2),)
    )
    assert stereo._gate7_chelate_pairs(reordered_site, adjacency) == frozenset(
        ((1, 2),)
    )


def _coordination_sites(molecule, records):
    _, adjacency = _projected_adjacency(molecule)
    sites = []
    for record_index, record in enumerate(records):
        site, issues = stereo._validate_gate7_coordination_record(
            record,
            _descriptor(record),
            record_index=record_index,
            atom_count=len(molecule.atoms),
            adjacency=adjacency,
        )
        assert issues == ()
        assert site is not None
        sites.append(site)
    return tuple(sites), adjacency


def _two_center_sites(molecule, first_carriers, second_carriers):
    return _coordination_sites(
        molecule,
        (
            _record("SP", 0, first_carriers),
            _record("SP", 5, second_carriers),
        ),
    )


def test_disconnected_coordination_centers_remain_independent():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 28, 7, 8, 9, 15),
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
        ),
    )
    sites, adjacency = _two_center_sites(
        molecule,
        (1, 2, 3, 4),
        (6, 7, 8, 9),
    )

    assert stereo._gate7_coupled_multicenter_issues(sites, adjacency) == ()


def _assert_two_sp_centers_are_coupled(sites, adjacency):
    issues = stereo._gate7_coupled_multicenter_issues(sites, adjacency)
    assert _issue_facts(issues) == (
        (
            0,
            stereo.StereoIssueCode.UNSUPPORTED,
            "SP chelate partition at center row 0 participates in coupled "
            "multicenter coordination, which is unsupported",
        ),
        (
            1,
            stereo.StereoIssueCode.UNSUPPORTED,
            "SP chelate partition at center row 5 participates in coupled "
            "multicenter coordination, which is unsupported",
        ),
    )


def test_shared_donor_diagnoses_every_coupled_coordination_record():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 28, 8, 9, 15),
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 1),
            (5, 6),
            (5, 7),
            (5, 8),
        ),
    )
    sites, adjacency = _two_center_sites(
        molecule,
        (1, 2, 3, 4),
        (1, 6, 7, 8),
    )

    _assert_two_sp_centers_are_coupled(sites, adjacency)


def test_shared_donor_chain_diagnoses_every_incident_coordination_record():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 28, 7, 8, 9, 15, 27, 7, 8, 9),
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 1),
            (5, 6),
            (5, 7),
            (5, 8),
            (10, 6),
            (10, 11),
            (10, 12),
            (10, 13),
        ),
    )
    sites, adjacency = _coordination_sites(
        molecule,
        (
            _record("SP", 0, (1, 2, 3, 4)),
            _record("SP", 5, (1, 6, 7, 8)),
            _record("SP", 10, (6, 11, 12, 13)),
        ),
    )

    issues = stereo._gate7_coupled_multicenter_issues(sites, adjacency)

    assert tuple(issue.record_index for issue in issues) == (0, 1, 2)
    assert all(issue.code is stereo.StereoIssueCode.UNSUPPORTED for issue in issues)


def test_declared_center_as_another_center_carrier_diagnoses_both_records():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 28, 7, 8, 9),
        (
            (0, 5),
            (0, 1),
            (0, 2),
            (0, 3),
            (5, 6),
            (5, 7),
            (5, 8),
        ),
    )
    sites, adjacency = _two_center_sites(
        molecule,
        (5, 1, 2, 3),
        (0, 6, 7, 8),
    )

    _assert_two_sp_centers_are_coupled(sites, adjacency)


def test_carrier_component_reaching_another_center_diagnoses_both_records():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 28, 7, 8, 9, 15, 6),
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (1, 10),
            (10, 6),
        ),
    )
    sites, adjacency = _two_center_sites(
        molecule,
        (1, 2, 3, 4),
        (6, 7, 8, 9),
    )

    _assert_two_sp_centers_are_coupled(sites, adjacency)


def test_synthetic_alternate_path_back_to_own_center_is_diagnosed():
    descriptor = _descriptor(_record("SP", 0, (1, 2, 3, 4)))
    site = stereo._Gate7CoordinationSite(
        record_index=12,
        descriptor=descriptor,
        center=0,
        carrier_rows=(1, 2, 3, 4),
    )
    adjacency = (
        frozenset((1, 2, 3, 4, 5)),
        frozenset((0, 5)),
        frozenset((0,)),
        frozenset((0,)),
        frozenset((0,)),
        frozenset((0, 1)),
    )

    issues = stereo._gate7_coupled_multicenter_issues((site,), adjacency)

    assert _issue_facts(issues) == (
        (
            12,
            stereo.StereoIssueCode.UNSUPPORTED,
            "SP chelate partition at center row 0 participates in coupled "
            "multicenter coordination, which is unsupported",
        ),
    )


def test_malformed_and_opaque_records_do_not_enter_the_base_valid_site_set():
    molecule = _molecule(
        (26, 7, 8, 9, 15, 28, 7, 8, 9),
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 1),
            (5, 6),
            (5, 7),
            (5, 8),
        ),
    )
    valid = _record("SP", 0, (1, 2, 3, 4))
    malformed = _record("SP", 5, (1, 6, 7))
    opaque = _opaque_record(0)
    payload = _payload(valid, malformed, opaque)
    stereo.set_stereo_payload(molecule, payload)
    descriptors = {0: _descriptor(valid), 1: _descriptor(malformed)}
    _, adjacency = _projected_adjacency(molecule)

    assert stereo._gate7_same_center_issues(
        payload,
        descriptors,
        atom_count=len(molecule.atoms),
    ) == ()
    valid_site, valid_issues = stereo._validate_gate7_coordination_record(
        valid,
        descriptors[0],
        record_index=0,
        atom_count=len(molecule.atoms),
        adjacency=adjacency,
    )
    malformed_site, malformed_issues = (
        stereo._validate_gate7_coordination_record(
            malformed,
            descriptors[1],
            record_index=1,
            atom_count=len(molecule.atoms),
            adjacency=adjacency,
        )
    )

    assert valid_issues == ()
    assert valid_site is not None
    assert malformed_site is None
    assert _issue_facts(malformed_issues) == (
        (
            1,
            stereo.StereoIssueCode.INVALID_CARRIER,
            "expected 4 ordered carriers, got 3",
        ),
    )
    assert stereo.get_stereo_payload(molecule) is payload
    assert stereo.get_stereo_payload(molecule).records[2] is opaque


def test_graph_equivalent_coordination_ligands_are_allowed():
    molecule = _molecule(
        (26, 7, 7, 7, 7),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, (1, 2, 3, 4))

    site, adjacency = _validated_site(molecule, record)

    assert stereo._gate7_chelate_pairs(site, adjacency) == frozenset()
    assert stereo._gate7_coupled_multicenter_issues((site,), adjacency) == ()


def test_public_gate7_graph_semantics_allow_graph_equivalent_ligands():
    molecule = _molecule(
        (26, 7, 7, 7, 7),
        ((0, 1), (0, 2), (0, 3), (0, 4)),
    )
    record = _record("SP", 0, (1, 2, 3, 4))
    payload = _payload(record)
    stereo.set_stereo_payload(molecule, payload)
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
    assert all(
        issue.code is not stereo.StereoIssueCode.NON_STEREOGENIC
        for issue in (
            *validation.issues,
            *assignment.unresolved,
            *alignment.issues,
        )
    )
    assert stereo.get_stereo_payload(molecule) is payload
