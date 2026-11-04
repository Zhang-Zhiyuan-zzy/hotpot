"""Native Open Babel stereo facts, independent identity and projection limits."""
from dataclasses import astuple, replace
import pickle
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
from openbabel import openbabel as ob, pybel
from rdkit import Chem

from hotpot import Molecule
from hotpot.cheminfo.operations import conversion
from hotpot.cheminfo.structure._builder import _freeze_value
from hotpot.cheminfo.stereo import (
    StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
    StereoGroupKind, StereoGroupRef, StereoIssueCode, StereoPayload, StereoRecord,
    BondTransportRecord, HydrogenTransportState, get_stereo_payload, set_stereo_payload,
    validate_stereochemistry, assign_stereochemistry, align_stereochemistry,
)


_NAMESPACE = 'openbabel-stereo/1'
_FIXTURES = (
    'F[C@](Cl)(Br)I', 'F[C@@](Cl)(Br)I',
    'N[C@H](C)O', 'N[C@@H](C)O',
    'F/C=C/F', r'F/C=C\F',
    'FC(Cl)(Br)I', 'FC=CF',
    'F/C(Cl)=C(Br)/I', r'F/C(Cl)=C(Br)\I',
    '[H][C@](F)(Cl)Br', '[2H][C@@](F)(Cl)Br',
    'C[S@](=O)N', 'C[S@@](=O)N',
)


def _facts(molecule):
    record, metadata = molecule._capture_factual_record()
    return astuple(record), _freeze_value(metadata)


def _configs(molecule):
    """Read normalized native configs directly, without facade perception."""
    atoms = tuple(ob.OBMolAtomIter(molecule))
    rows = {atom.GetId(): row for row, atom in enumerate(atoms)}
    implicit = int(ob.OBStereo.ImplicitRef) & 0xffffffff
    def carrier(reference, center):
        if int(reference) & 0xffffffff == implicit:
            return ('H' if atoms[rows[center]].GetImplicitHCount() else 'LP', None)
        return ('atom', rows[int(reference)])
    result = []
    for value in molecule.GetAllData(ob.StereoData):
        kind = ob.toStereoBase(value).GetType()
        if kind == ob.OBStereo.Tetrahedral:
            config = ob.toTetrahedralStereo(value).GetConfig()
            result.append(('TH', (rows[config.center],), tuple(
                carrier(reference, config.center)
                for reference in (config.from_or_towards, *config.refs)
            ), int(config.winding), bool(config.specified)))
        elif kind == ob.OBStereo.CisTrans:
            config = ob.toCisTransStereo(value).GetConfig()
            result.append(('CT', (rows[config.begin], rows[config.end]), tuple(
                carrier(reference, config.begin if index < 2 else config.end)
                for index, reference in enumerate(config.refs)
            ), int(config.shape), bool(config.specified)))
        elif kind == ob.OBStereo.SquarePlanar:
            config = ob.toSquarePlanarStereo(value).GetConfig()
            result.append(('SP', (rows[config.center],), tuple(
                carrier(reference, config.center) for reference in config.refs
            ), int(config.shape), bool(config.specified)))
        else:
            raise AssertionError(f'Unexpected test fixture stereo family {kind}')
    return tuple(result)


def _ob_facts(molecule):
    return (
        tuple((atom.GetIdx(), atom.GetId(), atom.GetAtomicNum(), atom.GetFormalCharge(),
               atom.GetImplicitHCount(), atom.GetIsotope(), atom.GetX(), atom.GetY(), atom.GetZ())
              for atom in ob.OBMolAtomIter(molecule)),
        tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondOrder(), bond.IsAromatic())
              for bond in ob.OBMolBondIter(molecule)),
        _configs(molecule), molecule.HasChiralityPerceived(),
    )


def _canonical(molecule):
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)))


def _native_value(text):
    return pybel.readstring('smi', text).OBMol


@pytest.mark.parametrize('text', _FIXTURES)
@pytest.mark.parametrize('kind', ('obmol', 'pybel'))
def test_native_input_keeps_configs_source_rows_hydrogens_and_identity(text, kind):
    source = _native_value(text)
    before = _ob_facts(source)
    molecule = conversion.to_hotpot_mol(source if kind == 'obmol' else pybel.Molecule(source))
    payload = get_stereo_payload(molecule)
    held = molecule.to_obmol()
    held_before = _ob_facts(held)
    facts = _facts(molecule)

    assert len(payload.records) == len(before[2])
    assert all(record.source_namespace == _NAMESPACE and record.descriptor == record.source_code
               for record in payload.records)
    assert tuple(atom.atomic_number for atom in molecule.atoms) == tuple(row[2] for row in before[0])
    assert _configs(held) == before[2]
    assert _canonical(molecule.to_rdmol()) == _canonical(Chem.MolFromSmiles(text))
    for presentation in (molecule.smiles, molecule.kekulize_smiles):
        assert _canonical(Chem.MolFromSmiles(presentation)) == _canonical(Chem.MolFromSmiles(text))
    projected, mapping = conversion.mol2obmol(molecule)
    assert mapping == {row: row + 1 for row in range(len(molecule.atoms))}
    assert projected is not held
    assert _configs(projected) == before[2]
    assert _ob_facts(held) == held_before
    assert _ob_facts(source) == before
    assert _facts(molecule) == facts


@pytest.mark.parametrize('text', _FIXTURES)
def test_rdkit_source_stereo_projects_natively_without_rewriting_source_facts(text):
    source = Chem.MolFromSmiles(text)
    before = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    molecule = conversion.to_hotpot_mol(source)
    molecule.to_obmol()
    molecule.to_rdmol()
    facts = _facts(molecule)

    projected = molecule.to_obmol()
    actual = Chem.MolFromSmiles(pybel.Molecule(projected).write('can').split()[0])

    assert _canonical(actual) == _canonical(source)
    assert _canonical(molecule.to_rdmol()) == _canonical(source)
    assert _facts(molecule) == facts
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == before
    assert projected.HasChiralityPerceived()


@pytest.mark.parametrize('shape', ('ShapeU', 'ShapeZ', 'Shape4'))
def test_native_square_planar_preserves_normalized_ligand_pairs_and_rejects_rdkit(shape):
    source = _native_value('Cl[Pt@SP1](N)(Br)I')
    data = ob.toSquarePlanarStereo(source.GetAllData(ob.StereoData)[0])
    config = data.GetConfig()
    config.shape = getattr(ob.OBStereo, shape)
    data.SetConfig(config)
    before = _ob_facts(source)

    molecule = conversion.to_hotpot_mol(source)

    assert get_stereo_payload(molecule).records[0].source_code == 'SP/U/1'
    assert _configs(molecule.to_obmol()) == before[2]
    with pytest.raises(ValueError, match='square.planar|SP'):
        molecule.to_rdmol()
    assert _ob_facts(source) == before


@pytest.mark.parametrize('family', ('TH', 'CT', 'SP'))
def test_present_unspecified_native_configs_keep_their_flags(family):
    source = _native_value({'TH':'F[C@](Cl)(Br)I', 'CT':'F/C=C/F', 'SP':'Cl[Pt@SP1](N)(Br)I'}[family])
    cast = {'TH':ob.toTetrahedralStereo, 'CT':ob.toCisTransStereo, 'SP':ob.toSquarePlanarStereo}[family]
    data = cast(source.GetAllData(ob.StereoData)[0])
    config = data.GetConfig()
    config.specified = False
    data.SetConfig(config)
    before = _configs(source)

    molecule = conversion.to_hotpot_mol(source)

    assert get_stereo_payload(molecule).records[0].source_code.endswith('/0')
    assert _configs(molecule.to_obmol()) == before
    if family != 'SP':
        projected = molecule.to_rdmol()
        assert all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in projected.GetAtoms())
        assert all(bond.GetStereo() in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
                   for bond in projected.GetBonds())


@pytest.mark.parametrize('specified', (False, True))
def test_native_unknown_winding_remains_unknown(specified):
    source = _native_value('F[C@](Cl)(Br)I')
    data = ob.toTetrahedralStereo(source.GetAllData(ob.StereoData)[0])
    config = data.GetConfig()
    config.winding = ob.OBStereo.UnknownWinding
    config.specified = specified
    data.SetConfig(config)
    before = _configs(source)

    molecule = conversion.to_hotpot_mol(source)

    assert get_stereo_payload(molecule).records[0].source_code == f'TH/UNKNOWN/{int(specified)}'
    assert _configs(molecule.to_obmol()) == before
    assert all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
               for atom in molecule.to_rdmol().GetAtoms())


def test_native_lone_pair_is_not_a_hydrogen_and_rdkit_source_three_carriers_remain_valid():
    source = _native_value('C[S@](=O)N')
    molecule = conversion.to_hotpot_mol(source)
    record = get_stereo_payload(molecule).records[0]

    assert tuple(carrier.kind for carrier in record.carriers).count(StereoCarrierKind.LONE_PAIR) == 1
    assert all(carrier.kind is not StereoCarrierKind.IMPLICIT_H for carrier in record.carriers)
    assert molecule.atoms[1].implicit_hydrogens == 0
    assert molecule.to_rdmol().GetAtomWithIdx(1).GetTotalNumHs() == 0
    rdkit_molecule = conversion.to_hotpot_mol(Chem.MolFromSmiles('C[S@](=O)N'))
    assert len(get_stereo_payload(rdkit_molecule).records[1].carriers) == 3
    assert _canonical(rdkit_molecule.to_rdmol()) == _canonical(Chem.MolFromSmiles('C[S@](=O)N'))


def test_native_ids_are_not_assumed_to_be_indices_or_contiguous_rows():
    source = _native_value('[He].F[C@](Cl)(Br)I')
    source.DeleteAtom(source.GetAtom(1))
    assert tuple(atom.GetId() for atom in ob.OBMolAtomIter(source)) != tuple(range(source.NumAtoms()))
    before = _configs(source)

    molecule = conversion.to_hotpot_mol(source)

    assert _configs(molecule.to_obmol()) == before
    assert _canonical(molecule.to_rdmol()) == _canonical(Chem.MolFromSmiles('F[C@](Cl)(Br)I'))


@pytest.mark.parametrize('text', ('F[C@](Cl)(Br)I', 'F/C=C/F', 'C[S@](=O)N'))
def test_native_records_survive_copy_pickle_and_nonmonotone_reorder(text):
    molecule = conversion.to_hotpot_mol(_native_value(text))
    payload = get_stereo_payload(molecule)
    assert payload.records
    before = _facts(molecule)
    rows = tuple(reversed(range(len(molecule.atoms))))
    record, metadata = molecule._capture_factual_record(rows=rows)
    reordered = Molecule()._install_molecule(record, bond_metadata=metadata)

    for value in (molecule.copy(), pickle.loads(pickle.dumps(molecule)), reordered):
        assert _canonical(value.to_rdmol()) == _canonical(Chem.MolFromSmiles(text))
        assert _canonical(Chem.MolFromSmiles(value.smiles)) == _canonical(Chem.MolFromSmiles(text))
    assert _facts(molecule) == before


def test_preconfigured_empty_target_keeps_schema_old_records_and_bond_facts():
    old = StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (81,)), (), 'opaque', None, 'future', 'raw')
    fact = BondTransportRecord((81, 82), 'future')
    target = Molecule()
    set_stereo_payload(target, StereoPayload(47, (old,), (), (fact,)))

    assert conversion.obmol2mol(_native_value('F[C@](Cl)(Br)I'), target) is target

    payload = get_stereo_payload(target)
    assert payload.schema_version == 47 and payload.records[0] == old
    assert payload.records[1].source_namespace == _NAMESPACE
    assert payload.bond_facts == (fact,)


@pytest.mark.parametrize('kind', tuple(StereoGroupKind))
def test_ob_projection_rejects_enhanced_group_loss(kind):
    molecule = conversion.to_hotpot_mol(Chem.MolFromSmiles('F[C@](Cl)(Br)I'))
    payload = get_stereo_payload(molecule)
    set_stereo_payload(molecule, replace(payload, records=tuple(
        replace(record, group=StereoGroupRef('scope', kind, 3)) if record.anchor.atom_indices == (1,) else record
        for record in payload.records
    )))
    with pytest.raises(ValueError, match='group'):
        molecule.to_obmol()


@pytest.mark.parametrize('namespace,code', (('future', 'TH/CW/1'), (_NAMESPACE, 'TH/FUTURE/1')))
@pytest.mark.parametrize('method', ('to_obmol', 'to_rdmol'))
def test_unregistered_namespace_or_native_code_is_not_silently_discarded(namespace, code, method):
    molecule = conversion.to_hotpot_mol(Chem.MolFromSmiles('F[C@](Cl)(Br)I'))
    old = get_stereo_payload(molecule).records[1]
    set_stereo_payload(molecule, replace(get_stereo_payload(molecule), records=(
        replace(old, source_namespace=namespace, source_code=code, descriptor=code),
    )))
    with pytest.raises(ValueError, match='namespace|code'):
        getattr(molecule, method)()


@pytest.mark.parametrize('family', ('ExtendedTetrahedral', 'ExtendedCisTrans', 'TrigonalBipyramidal', 'Octahedral'))
def test_native_extractor_explicitly_rejects_other_stereo_data_types(monkeypatch, family):
    source = _native_value('F[C@](Cl)(Br)I')
    monkeypatch.setattr(ob, 'toStereoBase', lambda data: SimpleNamespace(GetType=lambda: getattr(ob.OBStereo, family)))
    with pytest.raises(ValueError, match='family|type'):
        conversion.to_hotpot_mol(source)


def test_native_square_planar_implicit_reference_is_explicitly_unsupported():
    source = _native_value('Cl[Pt@SP1](N)(Br)I')
    data = ob.toSquarePlanarStereo(source.GetAllData(ob.StereoData)[0])
    config = data.GetConfig()
    config.refs = ob.OBStereo.MakeRefs(config.refs[0], config.refs[1], config.refs[2], int(ob.OBStereo.ImplicitRef) & 0xffffffff)
    data.SetConfig(config)
    with pytest.raises(ValueError, match='implicit|virtual|SP'):
        conversion.to_hotpot_mol(source)


def test_level2_keeps_raw_native_records_unresolved_and_preserved():
    molecule = conversion.to_hotpot_mol(_native_value('F[C@](Cl)(Br)I'))
    payload = get_stereo_payload(molecule)
    assert payload.records
    validation = validate_stereochemistry(molecule)
    assignment = assign_stereochemistry(molecule)
    alignment = align_stereochemistry(molecule, molecule, {row:row for row in range(len(molecule.atoms))})

    assert not validation.is_valid
    assert validation.issues[0].code is StereoIssueCode.INVALID_DESCRIPTOR
    assert assignment.payload == payload and assignment.unresolved == validation.issues
    assert not alignment.is_equivalent and alignment.aligned_payload == payload
    assert get_stereo_payload(molecule) == payload


@pytest.mark.parametrize('toolkit', ('openbabel', 'rdkit'))
def test_native_codec_imports_only_the_selected_toolkit_in_a_fresh_process(toolkit):
    program = textwrap.dedent("""
        import sys
        from hotpot.cheminfo.adapters.openbabel import stereo_transport as codec
        from hotpot.cheminfo.structure.stereo import (
            StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
            StereoPayload, StereoRecord,
        )
        assert not any(name.split('.')[0] in ('rdkit', 'openbabel') for name in sys.modules)
        if sys.argv[1] == 'openbabel':
            from openbabel import openbabel as ob
            reader = ob.OBConversion()
            reader.SetInFormat('smi')
            source = ob.OBMol()
            assert reader.ReadString(source, 'F[C@](Cl)(Br)I')
            records = codec._extract_stereo_records(source)
            target = ob.OBMol(source)
            target.DeleteData(ob.StereoData)
            codec._apply_stereo_payload(target, StereoPayload(1, records, (), ()),
                                       {row:row + 1 for row in range(target.NumAtoms())},
                                       tuple(atom.GetImplicitHCount() for atom in ob.OBMolAtomIter(target)))
            assert len(target.GetAllData(ob.StereoData)) == 1
            assert not any(name.split('.')[0] == 'rdkit' for name in sys.modules)
        else:
            from rdkit import Chem
            from hotpot.cheminfo.adapters.rdkit.stereo_transport import _apply_stereo_payload
            target = Chem.RWMol(Chem.MolFromSmiles('FC(Cl)(Br)I'))
            record = StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (1,)),
                                  tuple(StereoCarrier(StereoCarrierKind.ATOM, row) for row in (0, 2, 3, 4)),
                                  'TH/CW/1', None, 'openbabel-stereo/1', 'TH/CW/1')
            _apply_stereo_payload(target, StereoPayload(1, (record,), (), ()))
            assert target.GetAtomWithIdx(1).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            assert not any(name.split('.')[0] == 'openbabel' for name in sys.modules)
    """)
    result = subprocess.run([sys.executable, '-c', program, toolkit], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('text,opposite', (
    ('F[C@](Cl)(Br)I', 'F[C@@](Cl)(Br)I'),
    ('N[C@H](C)O', 'N[C@@H](C)O'),
    ('C[S@](=O)N', 'C[S@@](=O)N'),
))
def test_declared_native_counterclockwise_code_projects_the_opposite_isomer(text, opposite):
    molecule = conversion.to_hotpot_mol(_native_value(text))
    payload = get_stereo_payload(molecule)
    record, = payload.records
    assert record.source_code == 'TH/CW/1'
    set_stereo_payload(molecule, replace(payload, records=(
        replace(record, source_code='TH/CCW/1', descriptor='TH/CCW/1'),
    )))
    before = _facts(molecule)

    assert _canonical(molecule.to_rdmol()) == _canonical(Chem.MolFromSmiles(opposite))
    assert _canonical(Chem.MolFromSmiles(molecule.smiles)) == _canonical(Chem.MolFromSmiles(opposite))
    assert _facts(molecule) == before


def test_rdkit_unknown_alkene_stays_unspecified_in_native_projection():
    source = Chem.MolFromSmiles('FC=CF')
    source.GetBondBetweenAtoms(1, 2).SetStereo(Chem.BondStereo.STEREOANY)
    molecule = conversion.to_hotpot_mol(source)
    before = _facts(molecule)

    native = molecule.to_obmol()
    record, = get_stereo_payload(conversion.to_hotpot_mol(native)).records

    assert record.source_code == 'CT/U/0'
    assert _configs(native)[0][-1] is False
    assert molecule.to_rdmol().GetBondBetweenAtoms(1, 2).GetStereo() == Chem.BondStereo.STEREOANY
    assert _facts(molecule) == before


@pytest.mark.parametrize('text', ('N[C@H](C)O', 'F/C=C/F'))
def test_native_virtual_hydrogen_materialization_preserves_order_and_chemistry(text):
    molecule = conversion.to_hotpot_mol(_native_value(text))
    original = get_stereo_payload(molecule)
    expected = _canonical(Chem.MolFromSmiles(text))
    assert any(carrier.kind is StereoCarrierKind.IMPLICIT_H
               for record in original.records for carrier in record.carriers)

    molecule.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(433))
    payload = get_stereo_payload(molecule)

    assert payload.atom_facts == original.atom_facts
    for old, new in zip(original.records, payload.records):
        assert old.anchor == new.anchor and old.source_code == new.source_code
        for previous, current in zip(old.carriers, new.carriers):
            if previous.kind is StereoCarrierKind.IMPLICIT_H:
                assert current.kind is StereoCarrierKind.ATOM
                assert molecule.atoms[current.atom].atomic_number == 1
                assert all(fact.atom != current.atom for fact in payload.atom_facts)
            else:
                assert current == previous
    molecule.to_rdmol()
    molecule.to_obmol()
    assert get_stereo_payload(molecule) == payload
    before = _facts(molecule)
    assert _canonical(molecule.to_rdmol()) == expected
    assert _canonical(Chem.MolFromSmiles(molecule.smiles)) == expected
    assert _facts(molecule) == before


@pytest.mark.parametrize('text', ('F[C@](Cl)(Br)I', 'F/C(Cl)=C(Br)/I'))
def test_ob_projection_rejects_graph_carriers_plus_extra_declared_hydrogen(text):
    molecule = conversion.to_hotpot_mol(Chem.MolFromSmiles(text))
    payload = get_stereo_payload(molecule)
    center = 1
    set_stereo_payload(molecule, replace(payload, atom_facts=tuple(
        replace(fact, current_hydrogens=HydrogenTransportState(0, 1, True))
        if fact.atom == center else fact for fact in payload.atom_facts
    )))
    molecule.atoms[center].implicit_hydrogens = 1
    before = get_stereo_payload(molecule)

    with pytest.raises(ValueError, match='neighbor|carrier'):
        molecule.to_obmol()
    assert get_stereo_payload(molecule) == before


@pytest.mark.parametrize('method', ('to_obmol', 'to_rdmol'))
@pytest.mark.parametrize('text,hydrogens', (('F/C(Cl)=C(Br)/I', 1), ('F/C=C/F', 2)))
def test_native_alkene_rejects_extra_virtual_hydrogens_on_an_ordered_side(method, text, hydrogens):
    molecule = conversion.to_hotpot_mol(_native_value(text))
    molecule.atoms[1].implicit_hydrogens = hydrogens
    before = get_stereo_payload(molecule)

    with pytest.raises(ValueError, match='neighbor|carrier'):
        getattr(molecule, method)()
    assert get_stereo_payload(molecule) == before
