from hotpot.cheminfo import stereo

from hotpot.cheminfo.rules.chemistry.stereochemistry import algebra as stereo_algebra


def _atom_carrier(atom):
    return stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, atom)


def test_remap_preserves_record_and_carrier_order_group_and_opaque_data():
    group = stereo.StereoGroupRef(
        scope="import-7f2d",
        kind=stereo.StereoGroupKind.OR,
        id=91,
    )
    implicit_h = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)
    lone_pair = stereo.StereoCarrier(stereo.StereoCarrierKind.LONE_PAIR)
    atom_record = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (4,)),
        carriers=(_atom_carrier(3), implicit_h, _atom_carrier(1), lone_pair),
        descriptor="opaque::descriptor/alpha",
        group=group,
        source_namespace="toolkit.namespace/vNext",
        source_code="raw-code-17",
    )
    bond_record = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (8, 4)),
        carriers=(_atom_carrier(3),),
        descriptor="opaque::descriptor/beta",
        group=group,
        source_namespace="toolkit.namespace/vNext",
        source_code="raw-code-23",
    )
    payload = stereo.StereoPayload(
        schema_version=12,
        records=(atom_record, bond_record),
        atom_facts=(),
        bond_facts=(),
    )

    remapped = stereo_algebra._remap_stereo_payload(
        payload,
        {4: 10, 3: 40, 1: 30, 8: 20},
        allowed_bonds=((10, 20),),
    )

    assert remapped.schema_version == 12
    assert tuple(record.source_code for record in remapped.records) == (
        "raw-code-17",
        "raw-code-23",
    )
    assert remapped.records[0].anchor.atom_indices == (10,)
    assert remapped.records[1].anchor.atom_indices == (20, 10)
    assert remapped.records[0].carriers == (
        _atom_carrier(40),
        implicit_h,
        _atom_carrier(30),
        lone_pair,
    )
    assert remapped.records[0].carriers[1] is implicit_h
    assert remapped.records[0].carriers[3] is lone_pair
    assert all(record.group is group for record in remapped.records)
    assert remapped.records[0].descriptor == atom_record.descriptor
    assert remapped.records[0].source_namespace == atom_record.source_namespace


def test_remap_drops_whole_records_for_missing_atoms_or_missing_bonds():
    retained = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        carriers=(_atom_carrier(1),),
        descriptor="retained",
        group=None,
        source_namespace="test",
        source_code="0",
    )
    missing_anchor = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (9,)),
        carriers=(),
        descriptor="missing-anchor",
        group=None,
        source_namespace="test",
        source_code="1",
    )
    missing_carrier = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        carriers=(
            _atom_carrier(1),
            _atom_carrier(9),
            stereo.StereoCarrier(stereo.StereoCarrierKind.VACANCY),
        ),
        descriptor="missing-carrier",
        group=None,
        source_namespace="test",
        source_code="2",
    )
    missing_bond = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (0, 2)),
        carriers=(),
        descriptor="missing-bond",
        group=None,
        source_namespace="test",
        source_code="3",
    )
    payload = stereo.StereoPayload(
        schema_version=1,
        records=(retained, missing_anchor, missing_carrier, missing_bond),
        atom_facts=(),
        bond_facts=(),
    )

    remapped = stereo_algebra._remap_stereo_payload(
        payload,
        {0: 10, 1: 11, 2: 12},
        allowed_bonds=((10, 11),),
    )

    assert tuple(record.descriptor for record in remapped.records) == ("retained",)
    assert remapped.records[0].carriers == (_atom_carrier(11),)


def test_axis_remap_preserves_full_order_and_requires_consecutive_edges():
    axis = stereo.StereoRecord(
        anchor=stereo.StereoAnchor(
            stereo.StereoAnchorKind.AXIS,
            (4, 8, 2, 6),
        ),
        carriers=(_atom_carrier(1),),
        descriptor="opaque-axis",
        group=None,
        source_namespace="test",
        source_code="axis",
    )
    payload = stereo.StereoPayload(1, (axis,), (), ())
    mapping = {1: 11, 2: 12, 4: 14, 6: 16, 8: 18}

    remapped = stereo_algebra._remap_stereo_payload(
        payload,
        mapping,
        allowed_bonds=((14, 18), (18, 12), (12, 16)),
    )
    assert remapped.records[0].anchor.atom_indices == (14, 18, 12, 16)
    assert remapped.records[0].carriers == (_atom_carrier(11),)

    missing_atom = stereo_algebra._remap_stereo_payload(
        payload,
        {1: 11, 2: 12, 4: 14, 8: 18},
    )
    assert missing_atom.records == ()

    missing_constituent = stereo_algebra._remap_stereo_payload(
        payload,
        mapping,
        allowed_bonds=((14, 18), (12, 16), (14, 12)),
    )
    assert missing_constituent.records == ()


def test_axis_level_one_does_not_interpret_path_length():
    records = tuple(
        stereo.StereoRecord(
            anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.AXIS, path),
            carriers=(),
            descriptor=f"axis-path-{len(path)}",
            group=None,
            source_namespace="test",
            source_code=str(len(path)),
        )
        for path in ((), (4,))
    )
    payload = stereo.StereoPayload(1, records, (), ())

    remapped = stereo_algebra._remap_stereo_payload(
        payload,
        {4: 14},
        allowed_bonds=(),
    )

    assert tuple(record.anchor.atom_indices for record in remapped.records) == (
        (),
        (14,),
    )


def test_remap_filters_facts_in_order_and_preserves_bond_endpoint_direction():
    source_hydrogens = stereo.HydrogenTransportState(1, 2, False)
    current_hydrogens = stereo.HydrogenTransportState(0, 0, True)
    atom_facts = (
        stereo.AtomTransportRecord(8, 18, 2, None, None, "first"),
        stereo.AtomTransportRecord(
            4,
            13,
            1,
            current_hydrogens,
            source_hydrogens,
            "rdkit-2026.09.1",
        ),
        stereo.AtomTransportRecord(99, None, None, None, None, "removed"),
        stereo.AtomTransportRecord(1, None, 0, None, None, "last"),
    )
    bond_facts = (
        stereo.BondTransportRecord((8, 4), "BEGINWEDGE"),
        stereo.BondTransportRecord((4, 8), "BEGINDASH"),
        stereo.BondTransportRecord((1, 3), "EITHERDOUBLE"),
        stereo.BondTransportRecord((8, 99), "removed"),
    )
    payload = stereo.StereoPayload(1, (), atom_facts, bond_facts)

    remapped = stereo_algebra._remap_stereo_payload(
        payload,
        {4: 10, 3: 40, 1: 30, 8: 20},
        allowed_bonds=((10, 20),),
    )

    assert tuple(fact.atom for fact in remapped.atom_facts) == (20, 10, 30)
    assert tuple(fact.source_namespace for fact in remapped.atom_facts) == (
        "first",
        "rdkit-2026.09.1",
        "last",
    )
    hydrogen_fact = remapped.atom_facts[1]
    assert hydrogen_fact.current_hydrogens is current_hydrogens
    assert hydrogen_fact.source_hydrogens is source_hydrogens
    assert hydrogen_fact.current_hydrogens != hydrogen_fact.source_hydrogens

    assert tuple(fact.atom_indices for fact in remapped.bond_facts) == (
        (20, 10),
        (10, 20),
    )
    assert tuple(fact.display_direction for fact in remapped.bond_facts) == (
        "BEGINWEDGE",
        "BEGINDASH",
    )
