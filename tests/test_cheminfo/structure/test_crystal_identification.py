"""Explicit-map chemistry witnesses for generated Crystal identifications."""

from dataclasses import fields, replace

import numpy as np
import pytest


from hotpot.cheminfo.structure._builder import MoleculeRecord, _freeze_value
from hotpot.cheminfo.structure._fields import ATOM_FIELDS, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure.stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
    StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
    StereoGroupKind, StereoGroupRef, StereoPayload, StereoRecord,
)
from hotpot.cheminfo.structure.symmetry import SpaceGroup

from hotpot.cheminfo.rules.chemistry.crystal_identification import _identify_generated_record as identify

I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
MATRIX = ((5., 0., 0.), (0., 5., 0.), (0., 0., 5.))
P1 = SpaceGroup(1)
P1_ID = P1.identity()
PAIR_CLASSES = ((0, 2), (1, 3))


def record(coordinates=((0., 0., 0.), (0., 0., 0.)), endpoints=(), images=None,
           classes_charge=(True, 0, "explicit"), payload=None, **changes):
    count = len(coordinates)
    width = len(ATOM_FIELDS)
    values = np.zeros((count, width), dtype="<f8")
    values[:, FIELD_INDEX["atomic_number"]] = 6
    xyz = tuple(FIELD_INDEX[name] for name in ("x", "y", "z"))
    values[:, xyz] = np.asarray(coordinates, dtype="<f8").reshape(count, 3)
    supplied = ("atomic_number", "formal_charge", "implicit_hydrogens", "x", "y", "z")
    mask = sum(FIELD_PRESENCE_BITS[FIELD_INDEX[name]] for name in supplied)
    presence = np.full(count, mask, dtype="<u4")
    sources = np.ones(values.shape, dtype="u1")
    edge_count = len(endpoints)
    bond_values = np.zeros((edge_count, 3), dtype="<f8")
    bond_values[:, 0] = 1.
    origin = (MATRIX, P1, tuple(("site", row) for row in range(count)),
              tuple(("bond", edge) for edge in range(edge_count)), I)
    image_map = tuple(((row, P1_ID, (0, 0, 0)),) for row in range(count))
    bond_origins = tuple(((edge, P1_ID, (0, 0, 0)),) for edge in range(edge_count))
    cell = ("hotpot.crystal/1", MATRIX, P1,
            ((), (None,) * count, (None,) * count),
            ((0, 0, 0),) * edge_count if images is None else images,
            "cell", origin, image_map, bond_origins)
    arguments = dict(
        atom_count=count, atom_values=values.tobytes(), atom_presence=presence.tobytes(),
        atom_sources=sources.tobytes(), atom_frac_valences=(), bond_endpoints=endpoints,
        bond_values=bond_values.tobytes(), bond_kinds=("single",) * edge_count,
        bond_directions=(None,) * edge_count, bond_sources=("explicit",) * edge_count,
        bond_metadata=tuple(_freeze_value({}) for _ in endpoints), bond_aromatic=(False,) * edge_count,
        charge=classes_charge, spin_mult=(True, 1, "explicit"), cell=cell,
        stereo_payload=payload, source_frames=(), conformer_results=(), frame_selection=None,
        angle_constraints=(), torsion_constraints=(), identifier=(True, "source"),
    )
    arguments.update(changes)
    return MoleculeRecord(**arguments)


def two_edges(**changes):
    return record(((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                  ((0, 1), (2, 3)), **changes)


def atom_change(value, row, name, item, *, present=True):
    array = np.frombuffer(value.atom_values, dtype="<f8").reshape(value.atom_count, -1).copy()
    masks = np.frombuffer(value.atom_presence, dtype="<u4").copy()
    column, bit = FIELD_INDEX[name], FIELD_PRESENCE_BITS[FIELD_INDEX[name]]
    array[row, column] = item
    if present:
        masks[row] |= np.uint32(bit)
    else:
        masks[row] &= np.uint32(~bit & ((1 << len(ATOM_FIELDS)) - 1))
    return replace(value, atom_values=array.tobytes(), atom_presence=masks.tobytes())


def state(value):
    return tuple(getattr(value, item.name) for item in fields(value))


def reject_unchanged(value, classes, reason):
    before = state(value)
    with pytest.raises(ValueError, match="CRYSTAL-UNSUPPORTED-PROJECTION:.*" + reason):
        identify(value, classes)
    assert state(value) == before


def atom_fact(row, *, isotope=12, radical=0, hydrogen=HydrogenTransportState(0, 0, False),
              source_hydrogen=None, namespace="native-a"):
    return AtomTransportRecord(row, isotope, radical, hydrogen, source_hydrogen, namespace)


def atom_stereo(anchor, carriers, descriptor, *, code="history", namespace="hotpot.stereo/2", group=None):
    return StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (anchor,)),
                        tuple(StereoCarrier(StereoCarrierKind.ATOM, row) for row in carriers),
                        descriptor, group, namespace, code)


def tetrahedra(second_descriptor="hotpot.stereo/2/TH/odd", second_carriers=(7, 6, 8, 9)):
    coordinates = ((0., 0., 0.), (1., 1., 1.), (1., -1., -1.), (-1., 1., -1.), (-1., -1., 1.))
    endpoints = tuple((center, center + offset) for center in (0, 5) for offset in range(1, 5))
    records = (atom_stereo(0, (1, 2, 3, 4), "hotpot.stereo/2/TH/even", code="original-even"),
               atom_stereo(5, second_carriers, second_descriptor, code="original-mirrored"))
    value = record(coordinates + coordinates, endpoints, payload=StereoPayload(1, records, (), ()))
    return value, tuple((row, row + 5) for row in range(5))


def test_no_collapse_returns_the_identical_record_without_rederiving_charge_or_spin():
    value = two_edges(classes_charge=(True, 17, "declared"))
    assert identify(value, ((0,), (1,), (2,), (3,))) is value
    empty = record((), classes_charge=(False, None, None))
    assert identify(empty, ()) is empty


def test_isolated_ordered_special_position_merges_with_all_origins_and_unknown_spin():
    value = record()
    before = state(value)
    result = identify(value, ((0, 1),))
    assert result is not value and result.atom_count == 1 and result.bond_endpoints == ()
    assert result.cell[7] == ((value.cell[7][0][0], value.cell[7][1][0]),)
    assert result.cell[6] == value.cell[6]
    assert result.charge == value.charge and result.spin_mult == (False, None, None)
    assert state(value) == before


def test_same_graph_mapping_preserves_first_facts_and_every_atom_bond_origin():
    value = two_edges(bond_sources=("source-a", "source-b"))
    sources = np.frombuffer(value.atom_sources, dtype="u1").reshape(4, -1).copy()
    sources[2:] = 3
    value = replace(value, atom_sources=sources.tobytes(), atom_frac_valences=((0, 1.5), (2, 1.5)))
    result = identify(value, PAIR_CLASSES)
    assert result.atom_count == 2 and result.bond_endpoints == ((0, 1),)
    assert result.bond_sources == ("source-a",) and result.bond_values == value.bond_values[:24]
    assert result.atom_sources == sources[:2].tobytes() and result.atom_frac_valences == ((0, 1.5),)
    assert result.cell[7] == tuple(tuple(value.cell[7][row][0] for row in members) for members in PAIR_CLASSES)
    assert result.cell[8] == ((value.cell[8][0][0], value.cell[8][1][0]),)
    assert result.cell[6] == value.cell[6] and result.identifier == value.identifier


@pytest.mark.parametrize("residual", (0., 1e-10, -1e-10))
def test_coordinate_residual_has_an_inclusive_absolute_angstrom_boundary(residual):
    result = identify(record(((0., 0., 0.), (residual, 0., 0.))), ((0, 1),))
    assert result.atom_count == 1


@pytest.mark.parametrize("xyz", ((1.00001e-10, 0., 0.), (-1.00001e-10, 0., 0.), (.8e-10, .8e-10, 0.)))
def test_coordinate_residual_rejects_outside_euclidean_boundary(xyz):
    reject_unchanged(record(((0., 0., 0.), xyz)), ((0, 1),), "coordinate residual")


@pytest.mark.parametrize(("field", "item"), (("atomic_number", 7), ("formal_charge", 1),
    ("implicit_hydrogens", 1), ("id", 9), ("is_aromatic", 1), ("x_constraint", 1)))
def test_incompatible_actual_scalar_or_axis_fact_rejects(field, item):
    value = atom_change(record(), 0, field, 0 if field == "x_constraint" else 6 if field == "atomic_number" else 0)
    value = atom_change(value, 1, field, item)
    reject_unchanged(value, ((0, 1),), "scalar facts|axis constraints")


def test_presence_disagreement_is_rejected_but_absent_storage_bytes_do_not_assert_a_fact():
    value = atom_change(record(), 1, "x_constraint", 0)
    reject_unchanged(value, ((0, 1),), "presence")
    value = atom_change(record(), 1, "id", 9000, present=False)
    assert identify(value, ((0, 1),)).atom_count == 1


@pytest.mark.parametrize("valences", (((0, 1.),), ((0, 1.), (1, 2.))))
def test_fractional_valence_presence_and_value_are_checked(valences):
    reject_unchanged(record(atom_frac_valences=valences), ((0, 1),), "fractional valence")


@pytest.mark.parametrize(("occupancies", "disorder"), (((1., .5), (None, None)),
    ((1., None), (None, None)), ((1., 1.), (("A", 1), ("A", 2)))))
def test_occupancy_and_disorder_are_current_site_facts(occupancies, disorder):
    value = record()
    value = replace(value, cell=(*value.cell[:3], ((), occupancies, disorder), *value.cell[4:]))
    reject_unchanged(value, ((0, 1),), "occupancy/disorder")


@pytest.mark.parametrize(("field", "item"), (("isotope", 13), ("radical_electrons", 1),
    ("current_hydrogens", HydrogenTransportState(1, 0, False))))
def test_current_isotope_radical_and_hydrogen_disagreements_reject(field, item):
    first = atom_fact(0)
    second = replace(atom_fact(1), **{field: item})
    reject_unchanged(record(payload=StereoPayload(1, (), (first, second), ())), ((0, 1),), "isotope/radical/hydrogen")


def test_known_current_stereo_atom_fact_does_not_merge_with_missing_fact():
    reject_unchanged(record(payload=StereoPayload(1, (), (atom_fact(0),), ())), ((0, 1),), "isotope/radical/hydrogen")


def test_contradictory_current_facts_reject_even_when_every_alias_repeats_the_conflict():
    facts = tuple(atom_fact(row, isotope=isotope) for row in (0, 1) for isotope in (12, 13))
    reject_unchanged(record(payload=StereoPayload(1, (), facts, ())), ((0, 1),), "contradictory current")


def test_historical_hydrogen_ledgers_and_namespaces_survive_compatible_current_facts():
    old = HydrogenTransportState(3, None, True)
    facts = (atom_fact(0, source_hydrogen=old), atom_fact(1, source_hydrogen=None, namespace="native-b"))
    value = record(payload=StereoPayload(1, (), facts, ()))
    result = identify(value, ((0, 1),))
    assert result.stereo_payload.atom_facts == tuple(replace(fact, atom=0) for fact in facts)
    assert result.stereo_payload.atom_facts[0].source_hydrogens is old


def test_incomplete_neighbor_identification_rejects_even_when_coordinates_coincide():
    reject_unchanged(two_edges(), ((0, 2), (1,), (3,)), "mapped incident edges")
    value = record(((0., 0., 0.), (1., 0., 0.), (0., 0., 0.)), ((0, 1),))
    reject_unchanged(value, ((0, 2), (1,)), "mapped incident edges")


@pytest.mark.parametrize("difference", ("kind", "order", "constraint", "id", "aromatic", "metadata"))
def test_incompatible_bond_meaning_rejects_complete_alias_map(difference):
    value = two_edges()
    if difference == "kind":
        value = replace(value, bond_kinds=("single", "double"))
    elif difference == "aromatic":
        value = replace(value, bond_aromatic=(False, True))
    elif difference == "metadata":
        value = replace(value, bond_metadata=(_freeze_value({"meaning": "a"}), _freeze_value({"meaning": "b"})))
    else:
        array = np.frombuffer(value.bond_values, dtype="<f8").reshape(-1, 3).copy()
        array[1, {"order": 0, "constraint": 1, "id": 2}[difference]] = 2
        value = replace(value, bond_values=array.tobytes())
    reject_unchanged(value, PAIR_CLASSES, "mapped incident edges")


def test_reversed_dative_edges_normalize_image_and_direction_then_keep_first_orientation():
    value = record(((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                   ((0, 1), (3, 2)), images=((1, -2, 3), (-1, 2, -3)),
                   bond_kinds=("dative", "dative"), bond_directions=("1->2", "2->1"))
    result = identify(value, PAIR_CLASSES)
    assert result.bond_endpoints == ((0, 1),) and result.bond_directions == ("1->2",)
    assert result.cell[4] == ((1, -2, 3),)
    wrong = replace(value, bond_directions=("1->2", "1->2"))
    reject_unchanged(wrong, PAIR_CLASSES, "mapped incident edges")


def test_opaque_direction_matches_only_exact_mapped_orientation():
    assert identify(two_edges(bond_directions=("opaque-native", "opaque-native")), PAIR_CLASSES).atom_count == 2
    value = record(((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                   ((0, 1), (3, 2)), bond_directions=("opaque-native", "opaque-native"))
    reject_unchanged(value, PAIR_CLASSES, "mapped incident edges")


def test_required_unknown_bond_image_rejects_but_unaffected_unknown_image_remains_unknown():
    reject_unchanged(two_edges(images=((0, 0, 0), None)), PAIR_CLASSES, "required image is unknown")
    value = record(((0., 0., 0.), (0., 0., 0.), (1., 0., 0.), (2., 0., 0.)),
                   ((2, 3),), images=(None,))
    result = identify(value, ((0, 1), (2,), (3,)))
    assert result.bond_endpoints == ((1, 2),) and result.cell[4] == (None,)


def test_self_quotient_edge_rejects_without_mutation():
    reject_unchanged(record(endpoints=((0, 1),)), ((0, 1),), "self quotient")


def test_distinct_periodic_edges_cannot_be_collapsed_into_one_bond():
    reject_unchanged(two_edges(images=((0, 0, 0), (1, 0, 0))), PAIR_CLASSES, "mapped incident edges|parallel")
    value = record(((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (1., 0., 0.)),
                   ((0, 1), (0, 1), (2, 3), (2, 3)))
    reject_unchanged(value, PAIR_CLASSES, "parallel")


def test_origin_aliases_are_stable_unique_and_source_history_is_unmodified():
    value = two_edges()
    keys = tuple(row[0] for row in value.cell[7])
    atom_origins = ((keys[0], keys[1]), (keys[1],), (keys[1], keys[2]), (keys[3],))
    bond_keys = tuple(row[0] for row in value.cell[8])
    bond_origins = ((bond_keys[0], bond_keys[1]), (bond_keys[1],))
    value = replace(value, cell=(*value.cell[:7], atom_origins, bond_origins))
    result = identify(value, PAIR_CLASSES)
    assert result.cell[7] == ((keys[0], keys[1], keys[2]), (keys[1], keys[3]))
    assert result.cell[8] == ((bond_keys[0], bond_keys[1]),)
    assert result.cell[6] == value.cell[6]


def test_same_tetrahedral_orientation_survives_mirror_history_and_odd_reference_reordering():
    value, classes = tetrahedra()
    result = identify(value, classes)
    assert result.atom_count == 5 and len(result.bond_endpoints) == 4
    assert len(result.stereo_payload.records) == 2
    first, second = result.stereo_payload.records
    assert first.anchor == second.anchor and first.carriers != second.carriers
    assert first.descriptor == "hotpot.stereo/2/TH/even" and second.descriptor == "hotpot.stereo/2/TH/odd"
    assert first.source_code == "original-even" and second.source_code == "original-mirrored"


def test_equal_descriptor_text_does_not_hide_opposite_tetrahedral_orientation():
    value, classes = tetrahedra(second_descriptor="hotpot.stereo/2/TH/even")
    reject_unchanged(value, classes, "incompatible orientation")


def test_missing_stereo_occurrence_on_one_alias_rejects():
    value, classes = tetrahedra()
    value = replace(value, stereo_payload=replace(value.stereo_payload, records=value.stereo_payload.records[:1]))
    reject_unchanged(value, classes, "incomplete or incompatible mapped stereo")


def test_square_planar_trans_pair_identity_normalizes_with_the_coordination_algebra():
    value, classes = tetrahedra("hotpot.stereo/2/SP/2", (6, 7, 9, 8))
    first, second = value.stereo_payload.records
    value = replace(value, stereo_payload=replace(value.stereo_payload, records=(
        replace(first, descriptor="hotpot.stereo/2/SP/1"), second)))
    result = identify(value, classes)
    assert tuple(item.descriptor for item in result.stereo_payload.records) == ("hotpot.stereo/2/SP/1", "hotpot.stereo/2/SP/2")
    bad = replace(value, stereo_payload=replace(value.stereo_payload, records=(
        value.stereo_payload.records[0], replace(second, descriptor="hotpot.stereo/2/SP/1"))))
    reject_unchanged(bad, classes, "incompatible orientation")


def test_opaque_stereo_matches_exact_mapped_facts_and_retains_all_occurrences():
    records = (atom_stereo(0, (), "NATIVE_UNEXPLAINED", namespace="opaque/1", code="native"),
               atom_stereo(1, (), "NATIVE_UNEXPLAINED", namespace="opaque/1", code="native"))
    value = record(payload=StereoPayload(1, records, (), ()))
    result = identify(value, ((0, 1),))
    assert len(result.stereo_payload.records) == 2
    changed = replace(value, stereo_payload=replace(value.stereo_payload, records=(
        records[0], replace(records[1], descriptor="NATIVE_DIFFERENT"))))
    reject_unchanged(changed, ((0, 1),), "incompatible orientation or meaning")


def test_identification_cannot_collapse_distinct_opaque_stereo_references():
    opaque = atom_stereo(0, (1, 2), "unexplained", namespace="opaque/1")
    value = record(((0., 0., 0.), (1., 0., 0.), (1., 0., 0.)), payload=StereoPayload(1, (opaque,), (), ()))
    reject_unchanged(value, ((0,), (1, 2)), "collapses distinct stereo references")


def test_stereo_display_direction_presence_and_orientation_are_compared_and_retained():
    facts = (BondTransportRecord((0, 1), "opaque-display"), BondTransportRecord((2, 3), "opaque-display"))
    value = two_edges(payload=StereoPayload(1, (), (), facts))
    result = identify(value, PAIR_CLASSES)
    assert result.stereo_payload.bond_facts == (BondTransportRecord((0, 1), "opaque-display"),) * 2
    bad = replace(value, stereo_payload=replace(value.stereo_payload, bond_facts=(facts[0], replace(facts[1], atom_indices=(3, 2)))))
    reject_unchanged(bad, PAIR_CLASSES, "mapped incident edges")
    missing = replace(value, stereo_payload=replace(value.stereo_payload, bond_facts=facts[:1]))
    reject_unchanged(missing, PAIR_CLASSES, "mapped incident edges")


def test_absent_display_meaning_has_no_endpoint_direction_and_all_occurrences_survive():
    facts = (BondTransportRecord((0, 1), "NONE"), BondTransportRecord((3, 2), "NONE"))
    value = two_edges(payload=StereoPayload(1, (), (), facts))
    result = identify(value, PAIR_CLASSES)
    assert result.stereo_payload.bond_facts == (BondTransportRecord((0, 1), "NONE"), BondTransportRecord((1, 0), "NONE"))


def test_compatible_angle_and_torsion_constraints_are_remapped_in_stable_order():
    coordinates = ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (1., 1., 1.)) * 2
    value = record(coordinates, angle_constraints=((0, 1, 2), (4, 5, 6)),
                   torsion_constraints=((0, 1, 2, 3), (4, 5, 6, 7)))
    classes = tuple((row, row + 4) for row in range(4))
    result = identify(value, classes)
    assert result.angle_constraints == ((0, 1, 2),) and result.torsion_constraints == ((0, 1, 2, 3),)
    bad = replace(value, angle_constraints=((0, 1, 2),))
    reject_unchanged(bad, classes, "mapped angle constraints")


def test_constraint_with_identified_distinct_sites_rejects():
    value = record(((0., 0., 0.), (1., 0., 0.), (0., 0., 0.)), angle_constraints=((0, 1, 2),))
    reject_unchanged(value, ((0, 2), (1,)), "distinct angle constraint sites")


def test_charge_is_rederived_from_complete_agreeing_declared_atom_charges():
    value = record(classes_charge=(True, -2, "declared-total"))
    value = atom_change(atom_change(value, 0, "formal_charge", -1), 1, "formal_charge", -1)
    result = identify(value, ((0, 1),))
    assert result.charge == (True, -1, "declared-total")
    assert result.spin_mult == (False, None, None)


@pytest.mark.parametrize("mode", ("mismatch", "missing-one", "missing-all"))
def test_declared_total_charge_cannot_be_projected_from_incomplete_or_disagreeing_formal_charges(mode):
    value = record(classes_charge=(True, 1 if mode == "mismatch" else 0, "declared-total"))
    if mode != "mismatch":
        value = atom_change(value, 1, "formal_charge", 0, present=False)
    if mode == "missing-all":
        value = atom_change(value, 0, "formal_charge", 0, present=False)
    reject_unchanged(value, ((0, 1),), "scalar presence|changed total charge")


def test_absent_total_charge_stays_absent_and_is_never_inferred():
    value = record(classes_charge=(False, None, None))
    value = atom_change(atom_change(value, 0, "formal_charge", 0, present=False), 1, "formal_charge", 0, present=False)
    assert identify(value, ((0, 1),)).charge == (False, None, None)


def test_active_equivalence_assertions_are_remapped_and_fully_identified_groups_drop():
    value = record()
    keys = ((0, P1_ID, (0, 0, 0)), (1, P1_ID, (0, 0, 0)))
    value = replace(value, cell=(*value.cell[:3], ((keys,), (None, None), (None, None)), *value.cell[4:]))
    result = identify(value, ((0, 1),))
    assert result.cell[3][0] == () and len(result.cell[7][0]) == 2



def test_full_triatom_stabilizer_and_oxygen_alias_map_is_complete_but_central_only_map_is_not():
    value = record(((0., 0., 0.), (0., 0., 0.), (1., 0., 0.), (4., 0., 0.),
                    (4., 0., 0.), (1., 0., 0.)),
                   ((0, 2), (1, 3), (0, 4), (1, 5)),
                   images=((0, 0, 0), (-1, 0, 0), (-1, 0, 0), (0, 0, 0)))
    for row in (2, 3, 4, 5):
        value = atom_change(value, row, "atomic_number", 8)
    result = identify(value, ((0, 1), (2, 5), (3, 4)))
    assert result.atom_count == 3 and result.bond_endpoints == ((0, 1), (0, 2))
    assert result.cell[4] == ((0, 0, 0), (-1, 0, 0))
    assert tuple(map(len, result.cell[7])) == (2, 2, 2)
    assert result.cell[8] == (value.cell[8][0] + value.cell[8][3], value.cell[8][1] + value.cell[8][2])
    reject_unchanged(value, ((0, 1), (2,), (3,), (4,), (5,)), "mapped incident edges")


@pytest.mark.parametrize("family", ("DB", "AX"))
def test_endpoint_reversal_with_two_distinct_virtual_h_occurrences_uses_existing_scoped_algebra(family):
    h = StereoCarrier(StereoCarrierKind.IMPLICIT_H)
    if family == "DB":
        coordinates = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 1., 0.))
        edges = ((0, 1), (0, 2), (1, 3))
        anchor = StereoAnchor(StereoAnchorKind.BOND, (0, 1))
        second_anchor = StereoAnchor(StereoAnchorKind.BOND, (5, 4))
        carriers = (StereoCarrier(StereoCarrierKind.ATOM, 2), h, StereoCarrier(StereoCarrierKind.ATOM, 3), h)
        second_carriers = (StereoCarrier(StereoCarrierKind.ATOM, 7), h, StereoCarrier(StereoCarrierKind.ATOM, 6), h)
        descriptor = "hotpot.stereo/2/DB/together"
    else:
        coordinates = ((0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (0., 1., 0.), (2., 1., 0.))
        edges = ((0, 1), (1, 2), (0, 3), (2, 4))
        anchor = StereoAnchor(StereoAnchorKind.AXIS, (0, 1, 2))
        second_anchor = StereoAnchor(StereoAnchorKind.AXIS, (7, 6, 5))
        carriers = (StereoCarrier(StereoCarrierKind.ATOM, 3), h, StereoCarrier(StereoCarrierKind.ATOM, 4), h)
        second_carriers = (StereoCarrier(StereoCarrierKind.ATOM, 9), h, StereoCarrier(StereoCarrierKind.ATOM, 8), h)
        descriptor = "hotpot.stereo/2/AX/even"
    width = len(coordinates)
    records = (StereoRecord(anchor, carriers, descriptor, None, "hotpot.stereo/2", "source-a"),
               StereoRecord(second_anchor, second_carriers, descriptor, None, "hotpot.stereo/2", "source-b"))
    endpoints = edges + tuple((first + width, second + width) for first, second in edges)
    value = record(coordinates * 2, endpoints, payload=StereoPayload(1, records, (), ()))
    result = identify(value, tuple((row, row + width) for row in range(width)))
    assert result.atom_count == width and len(result.stereo_payload.records) == 2
    assert result.stereo_payload.records[1].anchor.atom_indices == anchor.atom_indices[::-1]


def test_declared_total_charge_requires_formal_charge_on_unmerged_sites_too():
    value = record(((0., 0., 0.), (0., 0., 0.), (1., 0., 0.)))
    value = atom_change(value, 2, "formal_charge", 0, present=False)
    reject_unchanged(value, ((0, 1), (2,)), "changed total charge")


def test_enhanced_stereo_group_meaning_must_agree_across_identified_sites():
    value, classes = tetrahedra()
    first, second = value.stereo_payload.records
    first = replace(first, group=StereoGroupRef("source", StereoGroupKind.AND, 1))
    second = replace(second, group=StereoGroupRef("source", StereoGroupKind.OR, 1))
    value = replace(value, stereo_payload=replace(value.stereo_payload, records=(first, second)))
    reject_unchanged(value, classes, "incompatible orientation or meaning")



def test_equal_incidence_does_not_authorize_parallel_quotient_edges_at_each_alias():
    value = record(((0., 0., 0.), (0., 0., 0.), (1., 0., 0.), (1., 0., 0.)),
                   ((0, 2), (0, 3), (1, 2), (1, 3)))
    reject_unchanged(value, ((0, 1), (2, 3)), "parallel quotient Bonds")



def test_all_unknown_row_origins_remain_unknown_during_identification():
    value = two_edges()
    value = replace(value, cell=(*value.cell[:7], (None,) * 4, (None,) * 2))
    result = identify(value, PAIR_CLASSES)
    assert result.cell[7:] == ((None, None), (None,))
    assert result.cell[6] == value.cell[6]


def test_mixed_known_unknown_origin_aliases_cannot_silently_discard_history():
    value = record()
    value = replace(value, cell=(*value.cell[:7], (value.cell[7][0], None), value.cell[8]))
    reject_unchanged(value, ((0, 1),), "origin aliases mix known and unknown")


def test_mixed_known_unknown_bond_origins_cannot_silently_discard_history():
    value = two_edges()
    value = replace(value, cell=(*value.cell[:8], (value.cell[8][0], None)))
    reject_unchanged(value, PAIR_CLASSES, "origin aliases mix known and unknown")
