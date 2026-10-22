from __future__ import annotations

import inspect

from hotpot.cheminfo import stereo


def test_level_one_value_contracts_are_frozen():
    for contract in (
        stereo.StereoCarrier,
        stereo.StereoAnchor,
        stereo.StereoGroupRef,
        stereo.StereoRecord,
        stereo.HydrogenTransportState,
        stereo.AtomTransportRecord,
        stereo.BondTransportRecord,
        stereo.StereoPayload,
    ):
        assert contract.__dataclass_params__.frozen is True


def test_level_two_report_contracts_are_frozen():
    for contract in (
        stereo.StereoValidationIssue,
        stereo.StereoValidationReport,
        stereo.StereoAssignmentReport,
        stereo.StereoAlignmentReport,
    ):
        assert contract.__dataclass_params__.frozen is True


def test_transport_enums_cover_non_atom_and_coordination_facts():
    assert tuple(stereo.StereoAnchorKind.__members__) == ("ATOM", "BOND")
    assert tuple(stereo.StereoCarrierKind.__members__) == (
        "ATOM",
        "IMPLICIT_H",
        "LONE_PAIR",
        "VACANCY",
    )
    assert tuple(stereo.StereoGroupKind.__members__) == ("ABS", "AND", "OR")
    assert {
        "TETRAHEDRAL",
        "DOUBLE_BOND",
        "ALLENE",
        "ATROPISOMERIC",
        "SQUARE_PLANAR",
        "TRIGONAL_BIPYRAMIDAL",
        "SQUARE_PYRAMIDAL",
        "OCTAHEDRAL",
        "UNKNOWN",
    } == set(stereo.StereoType.__members__)


def test_group_and_hydrogen_contracts_separate_lineage_and_current_state():
    group = stereo.StereoGroupRef(
        scope="source-molecule-a",
        kind=stereo.StereoGroupKind.OR,
        id=1,
    )
    source_hydrogens = stereo.HydrogenTransportState(
        implicit_hydrogens=1,
        bracket_explicit_hydrogens=0,
        no_implicit=False,
    )
    current_hydrogens = stereo.HydrogenTransportState(
        implicit_hydrogens=0,
        bracket_explicit_hydrogens=0,
        no_implicit=True,
    )
    record = stereo.AtomTransportRecord(
        atom=0,
        isotope=None,
        radical_electrons=None,
        current_hydrogens=current_hydrogens,
        source_hydrogens=source_hydrogens,
        source_namespace="rdkit",
    )

    assert group.scope == "source-molecule-a"
    assert group.kind is stereo.StereoGroupKind.OR
    assert record.source_hydrogens.implicit_hydrogens == 1
    assert record.current_hydrogens.no_implicit is True


def test_level_one_function_signatures_are_stable():
    get_payload = inspect.signature(stereo.get_stereo_payload).parameters
    assert tuple(get_payload) == ("mol",)

    set_payload = inspect.signature(stereo.set_stereo_payload).parameters
    assert tuple(set_payload) == ("mol", "payload")


def test_level_two_function_signatures_are_stable():
    validate = inspect.signature(stereo.validate_stereochemistry).parameters
    assert tuple(validate) == ("mol", "coordinates")
    assert validate["coordinates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert validate["coordinates"].default is None

    assign = inspect.signature(stereo.assign_stereochemistry).parameters
    assert tuple(assign) == ("mol", "coordinates")
    assert assign["coordinates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert assign["coordinates"].default is None

    align = inspect.signature(stereo.align_stereochemistry).parameters
    assert tuple(align) == ("reference", "target", "atom_mapping")
