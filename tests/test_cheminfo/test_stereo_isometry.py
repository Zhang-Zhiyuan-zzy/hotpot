import json
from dataclasses import replace
from functools import cache
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.rules.chemistry.stereochemistry import algebra as stereo_algebra


_GOLDEN = json.loads(
    (
        Path(__file__).parents[1]
        / "data"
        / "stereo"
        / "level2_gate7_coordination_geometry_golden.json"
    ).read_text()
)["families"]
_CARRIER_COUNTS = {"TH": 4, "AX": 4, "DB": 4, "SP": 4, "TBP": 5, "SPY": 5, "OH": 6}
_DESCRIPTORS = (
    *((family, value) for family in ("TH", "AX") for value in ("even", "odd")),
    *(("DB", value) for value in ("together", "opposite")),
    *(
        (family, value)
        for family, frozen in _GOLDEN.items()
        for value in range(1, frozen["cardinality"] + 1)
    ),
)
_COORDINATION_DESCRIPTORS = tuple(
    (family, value)
    for family, frozen in _GOLDEN.items()
    for value in range(1, frozen["cardinality"] + 1)
)


def _record(family="TH", value="even", *, group=None):
    if family == "DB":
        anchor = stereo.StereoAnchor(stereo.StereoAnchorKind.BOND, (7, 2))
    elif family == "AX":
        anchor = stereo.StereoAnchor(stereo.StereoAnchorKind.AXIS, (7, 4, 2))
    else:
        anchor = stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,))
    return stereo.StereoRecord(
        anchor=anchor,
        carriers=tuple(
            stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row)
            for row in (17, 3, 11, 5, 23, 29)[:_CARRIER_COUNTS[family]]
        ),
        descriptor=f"hotpot.stereo/2/{family}/{value}",
        group=group,
        source_namespace="isometry-source-test",
        source_code=f"historical::{family}/{value}",
    )


def _payload(*records, atom_facts=(), bond_facts=()):
    return stereo.StereoPayload(1, records, atom_facts, bond_facts)


@cache
def _independent_ideal_configurations(family):
    frozen = _GOLDEN[family]
    sites = np.asarray(frozen["ideal_sites"])
    rotations = frozen["proper_site_rotations"]
    representatives = sorted(
        {
            min(tuple(arrangement[position] for position in rotation) for rotation in rotations)
            for arrangement in permutations(range(len(sites)))
        }
    )
    configurations = []
    for representative in representatives:
        directions = np.empty_like(sites)
        for site, carrier_position in enumerate(representative):
            directions[carrier_position] = sites[site]
        configurations.append(directions)
    return tuple(configurations)


def _independent_proper_fit_error(ideal, observed):
    left, _, right_transpose = np.linalg.svd(ideal.T @ observed)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right_transpose)
    fitted = ideal @ left @ correction @ right_transpose
    return float(np.max(np.abs(fitted - observed)))


@pytest.mark.parametrize("orientation", (1, -1))
def test_absent_payload_stays_absent(orientation):
    assert stereo_algebra._transform_stereo_isometry(None, orientation) is None


@pytest.mark.parametrize(("family", "value"), _DESCRIPTORS)
def test_every_supported_descriptor_is_an_involution(family, value):
    source = _payload(_record(family, value))
    reflected = stereo_algebra._transform_stereo_isometry(source, -1)
    assert stereo_algebra._transform_stereo_isometry(reflected, -1) == source
    assert stereo_algebra._transform_stereo_isometry(source, 1) is source
    assert source.records[0].descriptor == f"hotpot.stereo/2/{family}/{value}"


@pytest.mark.parametrize(
    ("family", "value", "expected"),
    (
        ("TH", "even", "odd"),
        ("TH", "odd", "even"),
        ("AX", "even", "odd"),
        ("AX", "odd", "even"),
        ("DB", "together", "together"),
        ("DB", "opposite", "opposite"),
        ("SP", 1, 1),
        ("SP", 2, 2),
        ("SP", 3, 3),
    ),
)
def test_fixed_reference_orientation_action(family, value, expected):
    source = _payload(_record(family, value))
    reflected = stereo_algebra._transform_stereo_isometry(source, -1)
    assert reflected.records[0].descriptor == f"hotpot.stereo/2/{family}/{expected}"
    assert reflected.records[0].anchor is source.records[0].anchor
    assert reflected.records[0].carriers is source.records[0].carriers


@pytest.mark.parametrize(("family", "value"), _COORDINATION_DESCRIPTORS)
def test_coordination_action_matches_independent_geometric_reflection(family, value):
    ideals = _independent_ideal_configurations(family)
    reflected_directions = ideals[value - 1] @ np.diag((1.0, -1.0, 1.0))
    matching = tuple(
        candidate_value
        for candidate_value, ideal in enumerate(ideals, start=1)
        if _independent_proper_fit_error(ideal, reflected_directions) < 1.0e-12
    )
    assert len(matching) == 1
    assert matching[0] == _GOLDEN[family]["mirror_y"][value - 1]
    source = _payload(_record(family, value))
    actual = stereo_algebra._transform_stereo_isometry(source, -1)
    assert actual.records[0].descriptor == f"hotpot.stereo/2/{family}/{matching[0]}"
    assert actual.records[0].carriers is source.records[0].carriers


def test_proper_action_retains_opaque_foreign_and_display_facts_by_identity():
    records = (
        replace(_record(), descriptor="opaque::future", source_namespace="future-toolkit"),
        replace(_record(), descriptor="CHI_TETRAHEDRAL_CW", source_namespace="rdkit-2026.09.1"),
        replace(_record(), descriptor="TH/CW/1", source_namespace="openbabel-stereo/1"),
    )
    source = _payload(
        *records,
        bond_facts=(stereo.BondTransportRecord((7, 2), "BEGINWEDGE"),),
    )
    assert stereo_algebra._transform_stereo_isometry(source, 1) is source


@pytest.mark.parametrize("namespace", ("rdkit-2026.09.1", "openbabel-stereo/1"))
@pytest.mark.parametrize("descriptor", ("hotpot.stereo/2/TH/even", "opaque-native"))
def test_improper_action_rejects_replayed_foreign_namespaces(namespace, descriptor):
    record = replace(_record(), source_namespace=namespace, descriptor=descriptor)
    source = _payload(record)
    with pytest.raises(ValueError, match="CRYSTAL-UNSUPPORTED-PROJECTION:.*replay namespace"):
        stereo_algebra._transform_stereo_isometry(source, -1)
    assert source.records == (record,)


@pytest.mark.parametrize(
    "descriptor",
    (
        "opaque::future",
        "CHI_TETRAHEDRAL_CW",
        "hotpot.stereo/1/TH/even",
        "hotpot.stereo/2/TH/R",
        "hotpot.stereo/2/AX/P",
        "hotpot.stereo/2/DB/E",
        "hotpot.stereo/2/SP/0",
        "hotpot.stereo/2/TBP/21",
        "hotpot.stereo/2/SPY/31",
        "hotpot.stereo/2/OH/31",
        "hotpot.stereo/2/UNKNOWN/1",
    ),
)
def test_improper_action_rejects_uninterpreted_descriptor_without_mutation(descriptor):
    record = replace(_record(), descriptor=descriptor)
    source = _payload(_record(), record)
    with pytest.raises(ValueError, match="CRYSTAL-UNSUPPORTED-PROJECTION:.*descriptor"):
        stereo_algebra._transform_stereo_isometry(source, -1)
    assert source.records[0].descriptor == "hotpot.stereo/2/TH/even"
    assert source.records[1] is record


@pytest.mark.parametrize("token", ("BEGINWEDGE", "", "UP"))
def test_improper_action_rejects_every_uninterpreted_display_token(token):
    fact = stereo.BondTransportRecord((7, 2), token)
    source = _payload(_record(), bond_facts=(fact,))
    with pytest.raises(ValueError, match="CRYSTAL-UNSUPPORTED-PROJECTION:.*display"):
        stereo_algebra._transform_stereo_isometry(source, -1)
    assert source.bond_facts == (fact,)
    assert source.records[0].descriptor == "hotpot.stereo/2/TH/even"


def test_groups_occurrence_order_and_source_ledgers_are_preserved():
    relative = stereo.StereoGroupRef("source-scope", stereo.StereoGroupKind.AND, 9)
    alternative = stereo.StereoGroupRef("source-scope", stereo.StereoGroupKind.OR, 4)
    absolute = stereo.StereoGroupRef("source-scope", stereo.StereoGroupKind.ABS, 12)
    implicit_h = stereo.StereoCarrier(stereo.StereoCarrierKind.IMPLICIT_H)
    tetrahedral = _record(group=relative)
    tetrahedral = replace(tetrahedral, carriers=(*tetrahedral.carriers[:3], implicit_h))
    axial = _record("AX", "odd", group=relative)
    axial = replace(axial, carriers=(axial.carriers[0], implicit_h, axial.carriers[2], implicit_h))
    current_h = stereo.HydrogenTransportState(1, 0, False)
    source_h = stereo.HydrogenTransportState(0, 1, True)
    first_atom_fact = stereo.AtomTransportRecord(17, 13, 1, current_h, source_h, "rdkit-2026.09.1")
    second_atom_fact = stereo.AtomTransportRecord(3, None, None, None, None, "openbabel")
    first_bond_fact = stereo.BondTransportRecord((7, 2), None)
    second_bond_fact = stereo.BondTransportRecord((2, 7), None)
    records = (
        tetrahedral,
        axial,
        tetrahedral,
        _record("TH", "odd", group=alternative),
        _record("AX", "even", group=alternative),
        _record("TH", "even", group=absolute),
        _record("DB", "together"),
    )
    source = replace(
        _payload(
            *records,
            atom_facts=(first_atom_fact, second_atom_fact, first_atom_fact),
            bond_facts=(first_bond_fact, second_bond_fact, first_bond_fact),
        ),
        schema_version=12,
    )
    actual = stereo_algebra._transform_stereo_isometry(source, -1)
    assert actual.schema_version == 12
    assert actual.atom_facts is source.atom_facts
    assert actual.bond_facts is source.bond_facts
    assert tuple(record.descriptor for record in actual.records) == (
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/AX/even",
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/TH/even",
        "hotpot.stereo/2/AX/odd",
        "hotpot.stereo/2/TH/odd",
        "hotpot.stereo/2/DB/together",
    )
    for original, transformed in zip(source.records, actual.records):
        assert transformed.anchor is original.anchor
        assert transformed.carriers is original.carriers
        assert transformed.group is original.group
        assert transformed.source_namespace == original.source_namespace
        assert transformed.source_code == original.source_code
    assert stereo_algebra._transform_stereo_isometry(actual, -1) == source


def test_row_remapping_retains_parity_and_commutes_with_geometry_action():
    source = _payload(_record())
    mapping = {0: 0, 17: 3, 3: 17, 11: 11, 5: 5}
    mapped = stereo_algebra._remap_stereo_payload(source, mapping)
    assert mapped.records[0].descriptor == "hotpot.stereo/2/TH/even"
    reflected_mapped = stereo_algebra._transform_stereo_isometry(mapped, -1)
    assert reflected_mapped == stereo_algebra._remap_stereo_payload(
        stereo_algebra._transform_stereo_isometry(source, -1), mapping
    )
    assert reflected_mapped.records[0].descriptor == "hotpot.stereo/2/TH/odd"


@pytest.mark.parametrize(("family", "code"), (("TH", "CHI_UNSPECIFIED"), ("DB", "STEREONONE")))
def test_exact_rdkit_absence_declarations_are_invariant_with_empty_carriers(family, code):
    group = stereo.StereoGroupRef("native-source", stereo.StereoGroupKind.ABS, 7)
    record = replace(
        _record(family, "even" if family == "TH" else "together", group=group),
        carriers=(), descriptor=code, source_code=code, source_namespace="rdkit-2026.09.1",
    )
    source = _payload(record, record, bond_facts=(stereo.BondTransportRecord((7, 2), "NONE"),))
    reflected = stereo_algebra._transform_stereo_isometry(source, -1)
    assert reflected == source
    assert reflected.records[0] is record and reflected.records[1] is record
    assert reflected.bond_facts is source.bond_facts
    assert reflected.records[0].group is group
    assert stereo_algebra._transform_stereo_isometry(reflected, -1) == source


@pytest.mark.parametrize("token", (None, "NONE"))
def test_absent_bond_display_is_invariant_while_hotpot_orientation_transforms(token):
    source = _payload(_record(), bond_facts=(stereo.BondTransportRecord((7, 2), token),))
    reflected = stereo_algebra._transform_stereo_isometry(source, -1)
    assert reflected.bond_facts is source.bond_facts
    assert reflected.records[0].descriptor == "hotpot.stereo/2/TH/odd"


@pytest.mark.parametrize(
    ("descriptor", "code", "carriers"),
    (
        ("CHI_UNSPECIFIED", "CHI_UNSPECIFIED", (stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, 17),)),
        ("STEREONONE", "STEREONONE", (stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, 17),)),
        ("CHI_UNSPECIFIED", "STEREONONE", ()),
        ("STEREONONE", "CHI_UNSPECIFIED", ()),
        ("hotpot.stereo/2/TH/even", "CHI_UNSPECIFIED", ()),
        ("CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", ()),
    ),
)
def test_inconsistent_or_ordered_native_absence_is_not_silently_reinterpreted(descriptor, code, carriers):
    record = replace(
        _record(), descriptor=descriptor, source_code=code, carriers=carriers,
        source_namespace="rdkit-2026.09.1",
    )
    source = _payload(record)
    with pytest.raises(ValueError, match="CRYSTAL-UNSUPPORTED-PROJECTION:.*replay namespace"):
        stereo_algebra._transform_stereo_isometry(source, -1)
    assert source.records[0] is record


@pytest.mark.parametrize("code", ("CHI_UNSPECIFIED", "STEREONONE"))
def test_rdkit_absence_spelling_does_not_authorize_another_foreign_namespace(code):
    record = replace(
        _record(), descriptor=code, source_code=code, carriers=(),
        source_namespace="openbabel-stereo/1",
    )
    with pytest.raises(ValueError, match="CRYSTAL-UNSUPPORTED-PROJECTION:.*replay namespace"):
        stereo_algebra._transform_stereo_isometry(_payload(record), -1)
