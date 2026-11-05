"""Current factual SDF transport and ordinary-GFN active-electron boundaries."""

from dataclasses import fields, replace
import json
import os
import re

import pytest

import hotpot
from hotpot.cheminfo.operations.calculation.scientific_state.facts import (
    extract_molecular_facts, _molecule_state_source_identity,
)
from hotpot.cheminfo.structure._builder import MoleculeRecord
from hotpot.cheminfo.structure._cif_facts import _facts_from_data, _facts_to_data
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.structure.stereo import (
    AtomTransportRecord, StereoPayload,
)
from hotpot.cheminfo.stereo import get_stereo_payload, set_stereo_payload
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import ScientificStateIdentity
from hotpot.plugins.xtb.contracts import XTBMethod
from hotpot.plugins.xtb.state_resolution import resolve_xtb_state
from tests.test_cheminfo.calculator.test_automatic_scientific_state import (
    candidate_profile, _molecule as _coordination_fixture,
)
from hotpot.plugins.xtb.state_projection import _effective_nuclear_charge
from hotpot.plugins.xtb.stream import (
    XTBStreamError, XTBStreamMetadata, XTBStreamProvenance, XTBStreamRecord,
    read_sdf_records, write_sdf_records,
)


_FACTS_TAG = "HOTPOT_FACTS_CURRENT"


def _molecule(*numbers):
    mol = Molecule()
    for row, number in enumerate(numbers):
        mol.create_atom(atomic_number=number, coordinates=(row * 2.0, 0.0, 0.0),
                        implicit_hydrogens=0)
    return mol


def _metadata(mol, *, charge=0, uhf=0, method=XTBMethod.GFN2_XTB, bound=True):
    return XTBStreamMetadata(
        charge, uhf, method, -1.0,
        XTBStreamProvenance(
            "6.7.1", None, "a" * 64,
            ScientificStateIdentity("b" * 64) if bound else None,
            extract_molecular_facts(mol).identity if bound else None,
            "pure_spin_highest_ms" if bound else None,
        ),
    )


def _facts_payload(source):
    value = re.search(rf">  <{_FACTS_TAG}>\n([^\n]*)", source).group(1)
    return json.loads(value)


def _replace_facts(source, payload):
    value = json.dumps(payload, separators=(",", ":"))
    return re.sub(rf"(>  <{_FACTS_TAG}>\n)[^\n]*", lambda match: match[1] + value, source, count=1)


@pytest.mark.parametrize("metadata_present", (False, True))
@pytest.mark.parametrize("direction", ("donor_to_acceptor", "acceptor_to_donor"))
def test_hotpot_stream_preserves_canonical_dative_kind_direction_and_identity(metadata_present, direction):
    mol = _molecule(7, 30)
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE, bond_direction=direction,
                 bond_source="test:coordination")
    facts = extract_molecular_facts(mol)
    metadata = _metadata(mol, charge=2, uhf=1) if metadata_present else None

    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]

    assert restored.mol.bonds[0].bond_kind is BondKind.DATIVE
    assert restored.mol.bonds[0].bond_direction == direction
    assert extract_molecular_facts(restored.mol).identity == facts.identity
    assert restored.metadata == metadata


@pytest.mark.parametrize("declared", (False, True))
def test_facts_only_transport_preserves_fresh_source_identity_and_radical_namespace(declared):
    mol = _molecule(6, 8)
    mol.atoms[0].formal_charge = -1
    payload = StereoPayload(1, (), (
        AtomTransportRecord(0, None, 1, None, None, "test:radical"),
    ), ())
    set_stereo_payload(mol, payload)
    if declared:
        mol.charge = -1
        mol.spin_mult = 2
    before, _ = mol._capture_factual_record(include_frames=False)
    facts = extract_molecular_facts(mol)
    identity = _molecule_state_source_identity(mol, facts)

    source = write_sdf_records((XTBStreamRecord(mol),))
    restored = read_sdf_records(source)[0]
    after, _ = restored.mol._capture_factual_record(include_frames=False)

    assert isinstance(after, MoleculeRecord)
    assert restored.metadata is None
    assert after.charge == before.charge
    assert after.spin_mult == before.spin_mult
    assert after.atom_presence == before.atom_presence
    assert after.atom_sources == before.atom_sources
    assert get_stereo_payload(restored.mol).atom_facts == payload.atom_facts
    assert _molecule_state_source_identity(restored.mol, extract_molecular_facts(restored.mol)) == identity
    current = mol._capture_factual_record(include_frames=False)[0]
    assert all(getattr(current, field.name) == getattr(before, field.name) for field in fields(before))


def test_plain_external_metal_single_bond_keeps_ordinary_import_semantics():
    mol = _molecule(7, 30)
    mol.add_bond(0, 1)

    restored = read_sdf_records(mol.write(fmt="sdf", write_single=True))[0]

    assert restored.metadata is None
    assert restored.mol.bonds[0].bond_kind is BondKind.SINGLE


@pytest.mark.parametrize("field,value", (
    ("schema", "hotpot.sdf.facts/99"),
    ("structure_sha256", "a" * 64),
    ("structure_sha256", 7),
    ("facts", {}),
))
def test_malformed_facts_schema_or_structure_binding_is_named(field, value):
    source = write_sdf_records((XTBStreamRecord(_molecule(8)),))
    payload = _facts_payload(source)
    payload[field] = value

    with pytest.raises(XTBStreamError, match="facts.*(schema|binding)"):
        read_sdf_records(_replace_facts(source, payload))


def test_changed_structure_rejects_facts_only_binding():
    source = write_sdf_records((XTBStreamRecord(_molecule(8)),))
    changed = source.replace("    0.0000", "    0.1000", 1)
    assert changed != source

    with pytest.raises(XTBStreamError, match="facts binding does not match"):
        read_sdf_records(changed)


def test_duplicate_facts_property_is_rejected():
    source = write_sdf_records((XTBStreamRecord(_molecule(8)),))
    duplicate = source.replace("$$$$", source[source.index(f">  <{_FACTS_TAG}>"):source.index("$$$$")] + "$$$$")

    with pytest.raises(XTBStreamError, match="Duplicate reserved"):
        read_sdf_records(duplicate)


def test_legacy_xtb_metadata_without_private_facts_property_still_reads():
    mol = hotpot.read_mol("[H]O[H]", "smi")
    metadata = _metadata(mol, bound=False)
    source = write_sdf_records((XTBStreamRecord(mol, metadata),))
    legacy = re.sub(rf">  <{_FACTS_TAG}>\n[^\n]*\n\n", "", source, count=1)

    restored = read_sdf_records(legacy)[0]

    assert restored.metadata == metadata


@pytest.mark.parametrize("method", (XTBMethod.GFN0_XTB, XTBMethod.GFN1_XTB, XTBMethod.GFN2_XTB))
def test_ceiii_active_uhf_zero_preserves_physical_doublet(method):
    mol = _molecule(58)
    mol.charge = 3
    mol.spin_mult = 2
    metadata = _metadata(mol, charge=3, uhf=0, method=method)

    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]

    assert restored.metadata == metadata
    assert restored.mol.spin_mult == 2
    assert restored.mol.charge == 3


def test_gfnff_retains_recorded_gfn_uhf_under_active_electron_convention():
    mol = _molecule(58)
    mol.charge = 3
    mol.spin_mult = 2
    metadata = _metadata(mol, charge=3, uhf=0, method=XTBMethod.GFNFF)

    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]

    assert restored.metadata.unpaired_electrons == 0
    assert restored.mol.spin_mult == 2


@pytest.mark.parametrize("number,charge,uhf,message", (
    (58, 3, 1, "outside its domain"),
    (58, 2, 0, "electron parity"),
    (58, 4, 0, "negative electron count"),
    (8, 0, 7, "outside its domain"),
    (95, 0, 0, "unsupported atomic numbers"),
))
def test_active_electron_hard_boundaries(number, charge, uhf, message):
    mol = _molecule(number)
    metadata = _metadata(mol, charge=charge, uhf=uhf)

    with pytest.raises(XTBStreamError, match=message):
        write_sdf_records((XTBStreamRecord(mol, metadata),))


@pytest.mark.parametrize("number,expected", (
    (1, 1), (2, 2), (3, 1), (10, 8), (11, 1), (18, 8),
    (19, 1), (29, 11), (30, 2), (36, 8), (37, 1), (47, 11),
    (48, 2), (54, 8), (55, 1), (57, 3), (58, 3), (63, 3),
    (71, 3), (72, 4), (79, 11), (80, 2), (86, 8),
))
def test_shared_effective_electron_arithmetic(number, expected):
    assert _effective_nuclear_charge(number, XTBMethod.GFN2_XTB) == expected
    assert _effective_nuclear_charge(number, XTBMethod.GFNFF) == number


def test_shared_current_factual_codec_retains_an_absent_cell_without_adding_one():
    mol = _molecule(6, 8)
    record, _ = mol._capture_factual_record(include_frames=False)

    payload = _facts_to_data(record, ("C1", "O2"), True)
    restored, labels, contained = _facts_from_data(json.loads(json.dumps(payload)))

    assert payload["record"]["cell"] is None
    assert all(getattr(restored, field.name) == getattr(record, field.name) for field in fields(record))
    assert restored.cell is None
    assert labels == ("C1", "O2") and contained is True
    assert Molecule._build_molecule(restored)._crystal is None


def test_automatic_state_roundtrip_reconstructs_without_promoting_projection_to_source(
    candidate_profile, fake_xtb_factory,
):
    mol = _coordination_fixture()
    for row, atom in enumerate(mol.atoms):
        atom.coordinates = (row * 2.5, 0.0, 0.0)
    executable = fake_xtb_factory()
    options = dict(executable=executable, environment=dict(os.environ),
                   charge=None, uhf=None, policy=None)
    original = resolve_xtb_state(mol, XTBMethod.GFN2_XTB, **options)
    projection = original.projection
    metadata = _metadata(mol, charge=projection.charge, uhf=projection.uhf)
    metadata = replace(metadata, provenance=replace(
        metadata.provenance, scientific_state_identity=projection.scientific_state_identity,
    ))

    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]
    reconstructed = resolve_xtb_state(
        restored.mol, XTBMethod.GFN2_XTB, **options,
        expected_scientific_identity=metadata.provenance.scientific_state_identity,
        expected_facts_identity=metadata.provenance.facts_identity,
    )

    assert not restored.mol._charge_present and not restored.mol._spin_present
    assert reconstructed.projection.scientific_state_identity == projection.scientific_state_identity
    assert reconstructed.projection.charge == 3
    assert reconstructed.projection.uhf == 0


def test_stream_export_freezes_original_facts_before_toolkit_hydrogen_preparation():
    mol = _coordination_fixture()
    before, _ = mol._capture_factual_record(include_frames=False)

    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol),)))[0].mol
    after, _ = restored._capture_factual_record(include_frames=False)
    current, _ = mol._capture_factual_record(include_frames=False)

    assert after.atom_values == before.atom_values
    assert after.atom_presence == before.atom_presence
    assert after.atom_sources == before.atom_sources
    assert all(getattr(current, field.name) == getattr(before, field.name) for field in fields(before))
