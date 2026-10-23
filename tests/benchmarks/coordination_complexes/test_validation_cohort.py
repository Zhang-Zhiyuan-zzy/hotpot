"""Focused contracts for the current Plan-17 semantic cohort."""

from __future__ import annotations

import subprocess
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from hotpot.cheminfo.stereo import (
    AtomTransportRecord,
    BondTransportRecord,
    HydrogenTransportState,
    StereoAnchor,
    StereoAnchorKind,
    StereoCarrier,
    StereoCarrierKind,
    StereoGroupKind,
    StereoGroupRef,
    StereoPayload,
    StereoRecord,
)

from . import validation_cohort as cohort_module
from .configuration import STANDARD_SETTINGS
from .validation_cohort import (
    CANONICAL_SOURCE_PATH,
    CANONICAL_SOURCE_SHA256,
    EXPECTED_COMPLEX_COUNT,
    EXPECTED_NO_CBOND_INDICES,
    EXPECTED_SOURCE_COUNT,
    build_validation_cohort,
    derive_c1_c2_native_graph,
    load_validation_cohort,
    semantic_case_molecule,
    write_validation_cohort,
)


class _BondKind(str, Enum):
    SINGLE = "single"
    DATIVE = "dative"


@dataclass(frozen=True)
class _Atom:
    id: int
    atomic_number: int
    formal_charge: int
    is_aromatic: bool
    symbol: str
    implicit_hydrogens: int


@dataclass(frozen=True)
class _Bond:
    atom1: _Atom
    atom2: _Atom
    bond_order: float
    bond_kind: _BondKind
    is_aromatic: bool
    bond_direction: str | None


def _stereo_payload() -> StereoPayload:
    hydrogen_state = HydrogenTransportState(
        implicit_hydrogens=1,
        bracket_explicit_hydrogens=None,
        no_implicit=False,
    )
    return StereoPayload(
        schema_version=1,
        records=(
            StereoRecord(
                anchor=StereoAnchor(
                    kind=StereoAnchorKind.ATOM,
                    atom_indices=(0,),
                ),
                carriers=(
                    StereoCarrier(kind=StereoCarrierKind.ATOM, atom=1),
                    StereoCarrier(
                        kind=StereoCarrierKind.IMPLICIT_H,
                        atom=None,
                    ),
                ),
                descriptor="fixture/tetrahedral/cw",
                group=StereoGroupRef(
                    scope="fixture",
                    kind=StereoGroupKind.ABS,
                    id=7,
                ),
                source_namespace="fixture",
                source_code="CW",
            ),
        ),
        atom_facts=(
            AtomTransportRecord(
                atom=0,
                isotope=15,
                radical_electrons=0,
                current_hydrogens=hydrogen_state,
                source_hydrogens=hydrogen_state,
                source_namespace="fixture",
            ),
        ),
        bond_facts=(
            BondTransportRecord(
                atom_indices=(0, 1),
                display_direction="begin_wedge",
            ),
        ),
    )


def _positive_result(atom_ids: tuple[int, int] = (17, 93)) -> object:
    donor = _Atom(
        id=atom_ids[0],
        atomic_number=7,
        formal_charge=0,
        is_aromatic=True,
        symbol="N",
        implicit_hydrogens=1,
    )
    metal = _Atom(
        id=atom_ids[1],
        atomic_number=95,
        formal_charge=3,
        is_aromatic=False,
        symbol="Am",
        implicit_hydrogens=0,
    )
    molecule = SimpleNamespace(
        atoms=(donor, metal),
        bonds=(
            _Bond(
                atom1=donor,
                atom2=metal,
                bond_order=1.0,
                bond_kind=_BondKind.SINGLE,
                is_aromatic=False,
                bond_direction=None,
            ),
        ),
        _stereo_payload=_stereo_payload(),
    )
    return SimpleNamespace(molecule=molecule, donor_indices=(0,))


def _fixture_builder(calls: list[tuple[int, str, str, object]]):
    def build(index, smiles, metal, settings):
        calls.append((index, smiles, metal, settings))
        if index in EXPECTED_NO_CBOND_INDICES:
            return None
        return _positive_result()

    return build


def _positive_cases(cohort: dict[str, object]) -> list[dict[str, object]]:
    return [case for case in cohort["cases"] if case["status"] == "cbond"]


def _reseal(cohort: dict[str, object]) -> None:
    for case in _positive_cases(cohort):
        case["semantic_graph_sha256"] = cohort_module.canonical_digest(
            {"graph": case["graph"], "coordination": case["coordination"]}
        )
    content = dict(cohort)
    content.pop("canonical_sha256", None)
    cohort["canonical_sha256"] = cohort_module.canonical_digest(content)


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_canonical_source_is_the_tracked_default() -> None:
    relative_path = CANONICAL_SOURCE_PATH.relative_to(cohort_module.REPOSITORY_ROOT)
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path.as_posix()],
        cwd=cohort_module.REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert CANONICAL_SOURCE_PATH.is_file()


def test_builder_calls_cbond_once_per_case_and_freezes_current_semantics() -> None:
    calls: list[tuple[int, str, str, object]] = []
    cohort = build_validation_cohort(cbond_builder=_fixture_builder(calls))

    assert len(calls) == EXPECTED_SOURCE_COUNT
    assert [call[0] for call in calls] == list(range(1, EXPECTED_SOURCE_COUNT + 1))
    assert all(call[2] == "Am" for call in calls)
    assert all(call[3] is STANDARD_SETTINGS for call in calls)
    assert cohort["source"] == {
        "sha256": CANONICAL_SOURCE_SHA256,
        "physical_line_count": EXPECTED_SOURCE_COUNT,
    }
    assert cohort["partition"]["cbond_count"] == EXPECTED_COMPLEX_COUNT
    assert cohort["partition"]["no_cbond_indices"] == list(EXPECTED_NO_CBOND_INDICES)

    first = _positive_cases(cohort)[0]
    assert first["seed"] == STANDARD_SETTINGS.seed + 1
    assert first["settings"]["first_cbond_threshold"] == -0.5
    assert first["settings"]["subsequent_cbond_threshold"] == -0.125
    assert first["graph"]["atoms"] == [
        {
            "row": 0,
            "atom_id": 17,
            "atomic_number": 7,
            "formal_charge": 0,
            "is_aromatic": True,
            "implicit_hydrogens": 1,
        },
        {
            "row": 1,
            "atom_id": 93,
            "atomic_number": 95,
            "formal_charge": 3,
            "is_aromatic": False,
            "implicit_hydrogens": 0,
        },
    ]
    assert first["graph"]["bonds"] == [
        {
            "row": 0,
            "begin": 0,
            "end": 1,
            "bond_order": 1.0,
            "bond_kind": "single",
            "is_aromatic": False,
            "bond_direction": None,
        }
    ]
    assert first["coordination"] == {
        "metal": {
            "row": 1,
            "atom_id": 93,
            "atomic_number": 95,
            "symbol": "Am",
        },
        "donor_rows": [0],
    }
    assert first["graph"]["stereo"]["records"][0]["group"] == {
        "scope": "fixture",
        "kind": "abs",
        "id": 7,
    }
    assert first["graph"]["stereo"]["atom_transport"][0]["isotope"] == 15
    assert first["graph"]["stereo"]["bond_transport"] == [
        {
            "atom_rows": [0, 1],
            "display_direction": "begin_wedge",
        }
    ]

    prohibited = {
        "coordinates",
        "runtime",
        "timestamp",
        "reference_root",
        "hotpot_validation",
        "hotpot_status",
    }
    assert prohibited.isdisjoint(set(_walk_keys(cohort)))


def test_builder_is_deterministic_and_json_round_trips(tmp_path: Path) -> None:
    first = build_validation_cohort(cbond_builder=_fixture_builder([]))
    second = build_validation_cohort(cbond_builder=_fixture_builder([]))
    cohort_path = tmp_path / "semantic-cohort.json"

    assert second == first
    assert second["canonical_sha256"] == first["canonical_sha256"]

    write_validation_cohort(cohort_path, first)
    loaded = load_validation_cohort(cohort_path)

    assert loaded == first
    assert loaded["canonical_sha256"] == first["canonical_sha256"]


def test_schema_one_cohort_is_rejected() -> None:
    cohort = build_validation_cohort(cbond_builder=_fixture_builder([]))
    cohort["schema_version"] = 1
    _reseal(cohort)

    with pytest.raises(ValueError, match="unsupported validation cohort schema 1"):
        cohort_module.validate_validation_cohort(cohort)


@pytest.mark.parametrize("value", [None, True, -1, 1.0, "1"])
def test_invalid_implicit_hydrogen_state_is_rejected(value: object) -> None:
    cohort = build_validation_cohort(cbond_builder=_fixture_builder([]))
    atom = _positive_cases(cohort)[0]["graph"]["atoms"][0]
    if value is None:
        atom.pop("implicit_hydrogens")
    else:
        atom["implicit_hydrogens"] = value
    _reseal(cohort)

    with pytest.raises(ValueError, match="implicit-hydrogen counts are invalid"):
        cohort_module.validate_validation_cohort(cohort)


def test_implicit_hydrogen_state_changes_semantic_identity() -> None:
    original = build_validation_cohort(cbond_builder=_fixture_builder([]))
    changed = deepcopy(original)
    original_case = _positive_cases(original)[0]
    changed_case = _positive_cases(changed)[0]
    changed_case["graph"]["atoms"][0]["implicit_hydrogens"] += 1
    _reseal(changed)

    cohort_module.validate_validation_cohort(changed)
    assert changed_case["semantic_graph_sha256"] != original_case[
        "semantic_graph_sha256"
    ]
    assert changed["canonical_sha256"] != original["canonical_sha256"]


def test_builder_normalizes_non_unique_default_atom_ids_to_rows() -> None:
    def build(index, smiles, metal, settings):
        del smiles, metal, settings
        if index in EXPECTED_NO_CBOND_INDICES:
            return None
        return _positive_result((0, 0))

    cohort = build_validation_cohort(cbond_builder=build)
    first = _positive_cases(cohort)[0]
    source_graph = cohort_module._graph_document(_positive_result((0, 0)).molecule)

    assert [atom["atom_id"] for atom in source_graph["atoms"]] == [0, 0]
    assert [atom["atom_id"] for atom in first["graph"]["atoms"]] == [0, 1]
    assert first["coordination"]["metal"]["atom_id"] == 1
    assert [atom.id for atom in semantic_case_molecule(first).atoms] == [0, 1]


def test_native_projection_and_hotpot_reconstruction_reuse_the_same_graph() -> None:
    cohort = build_validation_cohort(cbond_builder=_fixture_builder([]))
    case = _positive_cases(cohort)[0]

    native = derive_c1_c2_native_graph(case)
    rebuilt = semantic_case_molecule(case)

    assert native["semantic_graph_sha256"] == case["semantic_graph_sha256"]
    assert native["atoms"] == case["graph"]["atoms"]
    assert native["source_bonds"] == case["graph"]["bonds"]
    assert native["bonds"] == [
        {
            "row": 0,
            "begin": 0,
            "end": 1,
            "bond_order": 1.0,
            "bond_kind": "dative",
            "is_aromatic": False,
            "bond_direction": "atom1_to_atom2",
        }
    ]
    assert native["generation_transform"] == (
        "coordination_edges_to_donor_metal_dative"
    )
    assert native["stereo"] == case["graph"]["stereo"]
    assert native["coordination"] == case["coordination"]
    assert cohort_module._graph_document(rebuilt) == case["graph"]
    assert [atom.idx for atom in rebuilt.metals] == [1]
    assert rebuilt.atoms[0].implicit_hydrogens == 1
