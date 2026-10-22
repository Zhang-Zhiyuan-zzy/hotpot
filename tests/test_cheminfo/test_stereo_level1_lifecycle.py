from copy import copy

from hotpot.cheminfo import stereo
from hotpot.cheminfo.core import Molecule


def _atom_record(
        atom,
        descriptor,
        *,
        carriers=(),
        group=None,
):
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (atom,)),
        carriers=tuple(
            stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, carrier)
            for carrier in carriers
        ),
        descriptor=descriptor,
        group=group,
        source_namespace="lifecycle-test",
        source_code=descriptor,
    )


def _bond_record(first, second, descriptor, *, group=None):
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(
            stereo.StereoAnchorKind.BOND,
            (first, second),
        ),
        carriers=(),
        descriptor=descriptor,
        group=group,
        source_namespace="lifecycle-test",
        source_code=descriptor,
    )


def _axis_record(*atoms, descriptor, carriers=(), group=None):
    return stereo.StereoRecord(
        anchor=stereo.StereoAnchor(stereo.StereoAnchorKind.AXIS, tuple(atoms)),
        carriers=tuple(
            stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, carrier)
            for carrier in carriers
        ),
        descriptor=descriptor,
        group=group,
        source_namespace="lifecycle-test",
        source_code=descriptor,
    )


def _molecule(atomic_numbers):
    molecule = Molecule()
    for atomic_number in atomic_numbers:
        molecule._create_atom(atomic_number=atomic_number)
    return molecule


def _record_descriptors(molecule):
    return tuple(
        record.descriptor
        for record in stereo.get_stereo_payload(molecule).records
    )


def test_new_molecule_owns_the_exact_immutable_empty_payload():
    molecule = Molecule()

    payload = stereo.get_stereo_payload(molecule)

    assert payload is molecule._stereo_payload
    assert payload == stereo.StereoPayload(1, (), (), ())


def test_copy_preserves_payload_order_lineage_and_hydrogen_ledgers():
    molecule = _molecule((6, 7, 8))
    molecule._add_bond(0, 1, bond_order=1.0)
    hidden = molecule._add_bond(1, 2, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    molecule.hide_bonds(hidden, clear_conformers=False)
    group = stereo.StereoGroupRef(
        scope="source-scope",
        kind=stereo.StereoGroupKind.AND,
        id=4,
    )
    current_hydrogens = stereo.HydrogenTransportState(0, 0, True)
    source_hydrogens = stereo.HydrogenTransportState(1, 0, False)
    payload = stereo.StereoPayload(
        1,
        (
            _bond_record(1, 0, "active", group=group),
            _bond_record(2, 1, "hidden", group=group),
            _atom_record(1, "atom", carriers=(0,), group=group),
        ),
        (
            stereo.AtomTransportRecord(
                1,
                15,
                1,
                current_hydrogens,
                source_hydrogens,
                "rdkit-2026.09.1",
            ),
        ),
        (
            stereo.BondTransportRecord((2, 1), "BEGINDASH"),
        ),
    )
    stereo.set_stereo_payload(molecule, payload)

    clone = copy(molecule)
    cloned_payload = stereo.get_stereo_payload(clone)

    assert cloned_payload == payload
    assert cloned_payload is not payload
    assert tuple(record.group for record in cloned_payload.records) == (
        group,
        group,
        group,
    )
    assert cloned_payload.atom_facts[0].current_hydrogens is current_hydrogens
    assert cloned_payload.atom_facts[0].source_hydrogens is source_hydrogens


def test_components_filter_and_remap_by_active_and_internal_hidden_bonds():
    molecule = _molecule((6, 7, 8, 9, 15, 16))
    molecule._add_bond(0, 1, bond_order=1.0)
    molecule._add_bond(1, 2, bond_order=1.0)
    molecule._add_bond(2, 3, bond_order=1.0)
    molecule._add_bond(4, 5, bond_order=1.0)
    hidden_internal = molecule._add_bond(0, 2, bond_order=1.0)
    hidden_cross = molecule._add_bond(3, 4, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    molecule.hide_bonds(
        hidden_internal,
        hidden_cross,
        clear_conformers=False,
    )
    group = stereo.StereoGroupRef(
        scope="split-source",
        kind=stereo.StereoGroupKind.OR,
        id=9,
    )
    stereo.set_stereo_payload(
        molecule,
        stereo.StereoPayload(
            1,
            (
                _atom_record(2, "first-atom", carriers=(0,), group=group),
                _bond_record(2, 1, "first-active", group=group),
                _bond_record(2, 0, "first-hidden", group=group),
                _bond_record(3, 0, "first-missing", group=group),
                _bond_record(3, 4, "cross-hidden", group=group),
                _atom_record(5, "second-atom", carriers=(4,), group=group),
                _bond_record(5, 4, "second-active", group=group),
            ),
            (
                stereo.AtomTransportRecord(5, None, 0, None, None, "last"),
                stereo.AtomTransportRecord(2, 18, None, None, None, "middle"),
                stereo.AtomTransportRecord(0, 13, None, None, None, "first"),
            ),
            (
                stereo.BondTransportRecord((2, 0), "hidden"),
                stereo.BondTransportRecord((3, 0), "missing"),
                stereo.BondTransportRecord((5, 4), "active"),
                stereo.BondTransportRecord((3, 4), "cross"),
            ),
        ),
    )

    first, second = molecule.components
    first_payload = stereo.get_stereo_payload(first)
    second_payload = stereo.get_stereo_payload(second)

    assert _record_descriptors(first) == (
        "first-atom",
        "first-active",
        "first-hidden",
    )
    assert _record_descriptors(second) == ("second-atom", "second-active")
    assert first_payload.records[0].anchor.atom_indices == (2,)
    assert second_payload.records[0].anchor.atom_indices == (1,)
    assert second_payload.records[0].carriers[0].atom == 0
    assert all(record.group is group for record in first_payload.records)
    assert all(record.group is group for record in second_payload.records)
    assert tuple(fact.atom for fact in first_payload.atom_facts) == (2, 0)
    assert tuple(fact.atom for fact in second_payload.atom_facts) == (1,)
    assert tuple(fact.atom_indices for fact in first_payload.bond_facts) == (
        (2, 0),
    )
    assert tuple(fact.atom_indices for fact in second_payload.bond_facts) == (
        (1, 0),
    )


def test_add_component_offsets_incoming_rows_without_group_merging_or_sorting():
    target = _molecule((6, 8))
    target._add_bond(0, 1, bond_order=1.0)
    target._update_graph(clear_conformers=False)
    target_group = stereo.StereoGroupRef(
        "target-scope",
        stereo.StereoGroupKind.OR,
        3,
    )
    stereo.set_stereo_payload(
        target,
        stereo.StereoPayload(
            1,
            (_atom_record(1, "target", carriers=(0,), group=target_group),),
            (stereo.AtomTransportRecord(1, 17, None, None, None, "target"),),
            (stereo.BondTransportRecord((1, 0), "target"),),
        ),
    )
    component = _molecule((7, 16))
    hidden = component._add_bond(0, 1, bond_order=1.0)
    component._update_graph(clear_conformers=False)
    component.hide_bonds(hidden, clear_conformers=False)
    component_group = stereo.StereoGroupRef(
        "component-scope",
        stereo.StereoGroupKind.OR,
        3,
    )
    stereo.set_stereo_payload(
        component,
        stereo.StereoPayload(
            1,
            (
                _bond_record(1, 0, "component-bond", group=component_group),
                _atom_record(
                    0,
                    "component-atom",
                    carriers=(1,),
                    group=component_group,
                ),
            ),
            (stereo.AtomTransportRecord(0, 15, None, None, None, "component"),),
            (stereo.BondTransportRecord((1, 0), "component"),),
        ),
    )

    target.add_component(component)
    payload = stereo.get_stereo_payload(target)

    assert _record_descriptors(target) == (
        "target",
        "component-bond",
        "component-atom",
    )
    assert payload.records[1].anchor.atom_indices == (3, 2)
    assert payload.records[2].anchor.atom_indices == (2,)
    assert payload.records[2].carriers[0].atom == 3
    assert payload.records[0].group is target_group
    assert payload.records[1].group is component_group
    assert payload.records[0].group.id == payload.records[1].group.id
    assert payload.records[0].group.scope != payload.records[1].group.scope
    assert tuple(fact.atom for fact in payload.atom_facts) == (1, 2)
    assert tuple(fact.atom_indices for fact in payload.bond_facts) == (
        (1, 0),
        (3, 2),
    )


def test_atom_removal_filters_whole_records_and_remaps_retained_order():
    molecule = _molecule((6, 7, 8, 9))
    molecule._add_bond(0, 1, bond_order=1.0)
    molecule._add_bond(1, 2, bond_order=1.0)
    molecule._add_bond(2, 3, bond_order=2.0)
    hidden = molecule._add_bond(0, 2, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    molecule.hide_bonds(hidden, clear_conformers=False)
    stereo.set_stereo_payload(
        molecule,
        stereo.StereoPayload(
            1,
            (
                _atom_record(3, "retained", carriers=(2,)),
                _atom_record(3, "removed-carrier", carriers=(1,)),
                _bond_record(3, 2, "retained-bond"),
                _bond_record(2, 0, "retained-hidden"),
                _bond_record(1, 0, "removed-bond"),
            ),
            (
                stereo.AtomTransportRecord(3, None, 1, None, None, "last"),
                stereo.AtomTransportRecord(1, None, 2, None, None, "removed"),
                stereo.AtomTransportRecord(0, None, 3, None, None, "first"),
            ),
            (
                stereo.BondTransportRecord((3, 2), "retained"),
                stereo.BondTransportRecord((1, 0), "removed"),
                stereo.BondTransportRecord((2, 0), "hidden"),
            ),
        ),
    )

    molecule.remove_atom(molecule.atoms[1])
    payload = stereo.get_stereo_payload(molecule)

    assert _record_descriptors(molecule) == (
        "retained",
        "retained-bond",
        "retained-hidden",
    )
    assert payload.records[0].anchor.atom_indices == (2,)
    assert payload.records[0].carriers[0].atom == 1
    assert payload.records[1].anchor.atom_indices == (2, 1)
    assert payload.records[2].anchor.atom_indices == (1, 0)
    assert tuple(fact.atom for fact in payload.atom_facts) == (2, 0)
    assert tuple(fact.atom_indices for fact in payload.bond_facts) == (
        (2, 1),
        (1, 0),
    )


def test_batch_atom_removal_uses_one_stable_source_row_mapping():
    molecule = _molecule((6, 7, 8, 9, 15))
    molecule._add_bond(0, 1, bond_order=1.0)
    molecule._add_bond(1, 2, bond_order=1.0)
    molecule._add_bond(2, 3, bond_order=1.0)
    molecule._add_bond(3, 4, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    stereo.set_stereo_payload(
        molecule,
        stereo.StereoPayload(
            1,
            (
                _atom_record(4, "retained", carriers=(2,)),
                _atom_record(4, "removed-carrier", carriers=(3,)),
                _atom_record(1, "removed-anchor"),
            ),
            (
                stereo.AtomTransportRecord(4, None, 0, None, None, "last"),
                stereo.AtomTransportRecord(2, None, 0, None, None, "middle"),
                stereo.AtomTransportRecord(1, None, 0, None, None, "removed"),
            ),
            (),
        ),
    )

    molecule.remove_atoms((molecule.atoms[1], molecule.atoms[3]))
    payload = stereo.get_stereo_payload(molecule)

    assert _record_descriptors(molecule) == ("retained",)
    assert payload.records[0].anchor.atom_indices == (2,)
    assert payload.records[0].carriers[0].atom == 1
    assert tuple(fact.atom for fact in payload.atom_facts) == (2, 1)


def test_bond_hide_recover_is_exact_and_permanent_deletion_is_targeted():
    molecule = _molecule((6, 7, 8))
    first = molecule._add_bond(0, 1, bond_order=1.0)
    second = molecule._add_bond(1, 2, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    payload = stereo.StereoPayload(
        1,
        (
            _atom_record(0, "atom"),
            _bond_record(1, 0, "first"),
            _bond_record(2, 1, "second"),
            _bond_record(2, 0, "preexisting-missing"),
        ),
        (),
        (
            stereo.BondTransportRecord((0, 1), "first"),
            stereo.BondTransportRecord((2, 1), "second"),
            stereo.BondTransportRecord((0, 2), "preexisting-missing"),
        ),
    )
    stereo.set_stereo_payload(molecule, payload)

    molecule.hide_bonds(second, clear_conformers=False)
    assert stereo.get_stereo_payload(molecule) is payload
    molecule.recover_hided_covalent_bonds(clear_conformers=False)
    assert stereo.get_stereo_payload(molecule) is payload

    molecule.remove_bond(first)
    after_active_removal = stereo.get_stereo_payload(molecule)
    assert tuple(record.descriptor for record in after_active_removal.records) == (
        "atom",
        "second",
        "preexisting-missing",
    )
    assert tuple(
        fact.display_direction for fact in after_active_removal.bond_facts
    ) == ("second", "preexisting-missing")

    molecule.hide_bonds(second, clear_conformers=False)
    hidden_payload = stereo.get_stereo_payload(molecule)
    molecule.clear_hided_covalent_bonds()
    after_hidden_clear = stereo.get_stereo_payload(molecule)
    assert hidden_payload is after_active_removal
    assert tuple(record.descriptor for record in after_hidden_clear.records) == (
        "atom",
        "preexisting-missing",
    )
    assert tuple(
        fact.display_direction for fact in after_hidden_clear.bond_facts
    ) == ("preexisting-missing",)


def test_axis_lifecycle_tracks_constituent_edges_but_not_chords():
    molecule = _molecule((6, 6, 6, 6))
    first = molecule._add_bond(0, 1, bond_order=2.0)
    second = molecule._add_bond(1, 2, bond_order=1.0)
    molecule._add_bond(2, 3, bond_order=2.0)
    chord = molecule._add_bond(0, 2, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    payload = stereo.StereoPayload(
        1,
        (_axis_record(0, 1, 2, 3, descriptor="axis", carriers=(3,)),),
        (),
        (),
    )
    stereo.set_stereo_payload(molecule, payload)

    clone = copy(molecule)
    assert stereo.get_stereo_payload(clone) == payload

    molecule.hide_bonds(second, clear_conformers=False)
    assert stereo.get_stereo_payload(molecule) is payload
    molecule.recover_hided_covalent_bonds(clear_conformers=False)
    assert stereo.get_stereo_payload(molecule) is payload

    molecule.remove_bond(chord)
    assert _record_descriptors(molecule) == ("axis",)

    molecule.remove_bond(first)
    assert _record_descriptors(molecule) == ()

    stereo.set_stereo_payload(clone, payload)
    clone.remove_bond(clone.bonds[2])
    assert _record_descriptors(clone) == ()


def test_permanently_clearing_hidden_axis_edges_drops_the_record():
    for clear_method in (
        "clear_hided_covalent_bonds",
        "clear_hided_bonds",
    ):
        molecule = _molecule((6, 6, 6))
        molecule._add_bond(0, 1, bond_order=2.0)
        hidden = molecule._add_bond(1, 2, bond_order=2.0)
        molecule._update_graph(clear_conformers=False)
        stereo.set_stereo_payload(
            molecule,
            stereo.StereoPayload(
                1,
                (_axis_record(0, 1, 2, descriptor=clear_method),),
                (),
                (),
            ),
        )

        molecule.hide_bonds(hidden, clear_conformers=False)
        assert _record_descriptors(molecule) == (clear_method,)
        getattr(molecule, clear_method)()

        assert _record_descriptors(molecule) == ()


def test_axis_atom_component_and_add_component_lifecycle_is_exact():
    molecule = _molecule((6, 6, 6, 8, 7))
    molecule._add_bond(0, 1, bond_order=2.0)
    molecule._add_bond(1, 2, bond_order=1.0)
    molecule._add_bond(2, 3, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    stereo.set_stereo_payload(
        molecule,
        stereo.StereoPayload(
            1,
            (
                _axis_record(0, 1, 2, descriptor="kept", carriers=(2,)),
                _axis_record(1, 2, 3, descriptor="removed", carriers=(0,)),
            ),
            (),
            (),
        ),
    )

    molecule.remove_atom(molecule.atoms[3])
    assert _record_descriptors(molecule) == ("kept",)
    assert stereo.get_stereo_payload(molecule).records[0].anchor.atom_indices == (
        0,
        1,
        2,
    )

    axis_component = molecule.components[0]
    isolated_component = molecule.components[1]
    assert _record_descriptors(axis_component) == ("kept",)
    assert _record_descriptors(isolated_component) == ()

    target = _molecule((8, 6, 6))
    target._add_bond(0, 1, bond_order=1.0)
    target._add_bond(1, 2, bond_order=1.0)
    target._update_graph(clear_conformers=False)
    stereo.set_stereo_payload(
        target,
        stereo.StereoPayload(
            1,
            (_axis_record(0, 1, 2, descriptor="target-first"),),
            (),
            (),
        ),
    )
    target.add_component(axis_component)
    target_record, added = stereo.get_stereo_payload(target).records
    assert target_record.descriptor == "target-first"
    assert target_record.anchor.atom_indices == (0, 1, 2)
    assert added.anchor.atom_indices == (3, 4, 5)
    assert added.carriers[0].atom == 5


def test_atom_sequence_to_mol_filters_and_remaps_payload():
    molecule = _molecule((6, 7, 8))
    molecule._add_bond(0, 1, bond_order=1.0)
    selected_bond = molecule._add_bond(1, 2, bond_order=2.0)
    molecule._update_graph(clear_conformers=False)
    stereo.set_stereo_payload(
        molecule,
        stereo.StereoPayload(
            1,
            (
                _atom_record(2, "selected", carriers=(1,)),
                _atom_record(0, "not-selected"),
                _bond_record(2, 1, "selected-bond"),
            ),
            (
                stereo.AtomTransportRecord(2, 18, None, None, None, "oxygen"),
                stereo.AtomTransportRecord(0, 13, None, None, None, "carbon"),
            ),
            (
                stereo.BondTransportRecord((2, 1), "selected"),
                stereo.BondTransportRecord((1, 0), "not-selected"),
            ),
        ),
    )

    extracted = selected_bond.to_mol()
    payload = stereo.get_stereo_payload(extracted)

    assert _record_descriptors(extracted) == ("selected", "selected-bond")
    assert payload.records[0].anchor.atom_indices == (1,)
    assert payload.records[0].carriers[0].atom == 0
    assert payload.records[1].anchor.atom_indices == (1, 0)
    assert tuple(fact.atom for fact in payload.atom_facts) == (1,)
    assert tuple(fact.atom_indices for fact in payload.bond_facts) == ((1, 0),)
