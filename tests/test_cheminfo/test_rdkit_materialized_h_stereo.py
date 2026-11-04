"""RDKit export preserves declared stereo through graph-H materialization."""
from dataclasses import replace
import json

import numpy as np
import pytest
from rdkit import Chem

from hotpot import Molecule
from hotpot.cheminfo.adapters.rdkit.stereo_transport import _apply_stereo_payload
from hotpot.cheminfo.obWrappers import build
from hotpot.cheminfo.operations.chemistry import _require_non_graph_hydrogens
from hotpot.cheminfo.operations.conversion import from_rdmol
from hotpot.cheminfo.stereo import get_stereo_payload, set_stereo_payload


def _canonical(value):
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(value)), isomericSmiles=True)


def _graph(mol):
    return (tuple((atom.atomic_number, atom.formal_charge) for atom in mol.atoms),
            tuple((bond.atom1.idx, bond.atom2.idx, bond.bond_order, bond.bond_kind)
                  for bond in mol.bonds), mol.charge)


def _numeric_rdmol(mol, *, reverse_bonds=False):
    """Independent graph/H/coordinate projection, with no stereo exporter."""
    graph = Chem.RWMol()
    for atom, hydrogens in zip(mol.atoms, _require_non_graph_hydrogens(mol)):
        value = Chem.Atom(atom.atomic_number)
        value.SetFormalCharge(atom.formal_charge)
        value.SetNumExplicitHs(hydrogens)
        value.SetNoImplicit(True)
        isotope = getattr(atom, 'isotope', None)
        if isotope is not None:
            value.SetIsotope(isotope)
        graph.AddAtom(value)
    bond_types = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
                  3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
    bonds = tuple(reversed(mol.bonds)) if reverse_bonds else mol.bonds
    for bond in bonds:
        graph.AddBond(bond.atom1.idx, bond.atom2.idx, bond_types[bond.bond_order])
    result = graph.GetMol()
    Chem.SanitizeMol(result)
    conformer = Chem.Conformer(len(mol.atoms))
    for row, position in enumerate(mol.coordinates):
        conformer.SetAtomPosition(row, position)
    result.AddConformer(conformer)
    return result


def _geometry_centers(mol):
    projected = _numeric_rdmol(mol)
    Chem.RemoveStereochemistry(projected)
    Chem.AssignStereochemistryFrom3D(projected, replaceExistingTags=True)
    Chem.AssignStereochemistry(projected, cleanIt=True, force=True)
    return tuple(atom.GetProp('_CIPCode') for atom in projected.GetAtoms()
                 if atom.HasProp('_CIPCode'))


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl',
                                 '[NH3+][C@H](F)C', '[NH3+][C@@H](F)C'))
@pytest.mark.parametrize('materialize_h', (False, True))
def test_rdkit_stereo_export_preserves_graph_charge_and_source_facts(text, materialize_h):
    source = Chem.MolFromSmiles(text)
    source_bytes = source.ToBinary(Chem.PropertyPickleOptions.AllProps)
    mol = from_rdmol(source, Molecule())
    original = get_stereo_payload(mol)
    if materialize_h:
        mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    payload, graph, coordinates = get_stereo_payload(mol), _graph(mol), mol.coordinates.copy()

    exported = mol.to_rdmol()

    assert _canonical(exported) == _canonical(source)
    assert Chem.GetFormalCharge(exported) == Chem.GetFormalCharge(source)
    assert _graph(mol) == graph
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    assert get_stereo_payload(mol) == payload
    assert payload.records == original.records
    assert source.ToBinary(Chem.PropertyPickleOptions.AllProps) == source_bytes


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl'))
@pytest.mark.parametrize('operation', ('copy', 'reverse_rows'))
def test_materialized_h_survives_copy_and_atom_row_remapping(text, operation):
    source = Chem.MolFromSmiles(text)
    mol = from_rdmol(source, Molecule())
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    original = get_stereo_payload(mol)
    if operation == 'copy':
        target = mol.copy()
    else:
        rows = tuple(reversed(range(len(mol.atoms))))
        record, metadata = mol._capture_factual_record(rows=rows)
        target = Molecule()._install_molecule(record, bond_metadata=metadata)
    target_payload = get_stereo_payload(target)

    assert _canonical(target.to_rdmol()) == _canonical(source)
    assert get_stereo_payload(target) == target_payload
    assert get_stereo_payload(mol) == original


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl'))
def test_materialized_h_binding_preserves_parity_when_native_bond_order_is_reversed(text):
    source = Chem.MolFromSmiles(text)
    mol = from_rdmol(source, Molecule())
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    payload = get_stereo_payload(mol)
    projected = Chem.RWMol(_numeric_rdmol(mol, reverse_bonds=True))

    _apply_stereo_payload(projected, payload)

    assert _canonical(projected) == _canonical(source)
    assert get_stereo_payload(mol) == payload


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl'))
def test_export_and_reimport_preserve_independently_derived_built_geometry(text, tmp_path):
    source = Chem.MolFromSmiles(text)
    expected = tuple(atom.GetProp('_CIPCode') for atom in source.GetAtoms()
                     if atom.HasProp('_CIPCode'))
    mol = from_rdmol(source, Molecule())
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    assert build(mol).succeeded
    before, graph = get_stereo_payload(mol), _graph(mol)
    before_centers = _geometry_centers(mol)
    assert before_centers == expected

    exported = mol.to_rdmol()
    reimported = from_rdmol(exported, Molecule())
    after_centers = _geometry_centers(reimported)
    observation = {'input_smiles': text, 'expected_centers': expected,
                   'before_roundtrip_centers': before_centers,
                   'after_roundtrip_centers': after_centers,
                   'graph_preserved': _graph(mol) == graph,
                   'canonical_payload_preserved': get_stereo_payload(mol) == before}
    (tmp_path / 'roundtrip-geometry.json').write_text(json.dumps(observation, indent=2) + '\n')
    np.save(tmp_path / 'before-coordinates.npy', mol.coordinates)
    np.save(tmp_path / 'roundtripped-coordinates.npy', reimported.coordinates)

    assert _canonical(exported) == _canonical(source)
    assert after_centers == expected
    np.testing.assert_array_equal(reimported.coordinates, mol.coordinates)
    assert get_stereo_payload(mol) == before
    assert _graph(mol) == graph


@pytest.mark.parametrize('text', ('C[S@](=O)N', 'C[S@@](=O)N'))
def test_lone_pair_three_carrier_inventory_remains_supported(text):
    source = Chem.MolFromSmiles(text)
    mol = from_rdmol(source, Molecule())
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    assert _canonical(mol.to_rdmol()) == _canonical(source)


@pytest.mark.parametrize('materialize_h', (False, True))
def test_true_implicit_h_carrier_keeps_its_existing_transport(materialize_h):
    source = Chem.MolFromSmiles('FC(Cl)Br')
    source.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    mol = from_rdmol(source, Molecule())
    if materialize_h:
        mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    assert _canonical(mol.to_rdmol()) == _canonical(source)


def test_three_carriers_never_bind_a_uniquely_remaining_heavy_neighbor():
    mol = from_rdmol(Chem.MolFromSmiles('F[C@](Cl)(Br)I'), Molecule())
    payload = get_stereo_payload(mol)
    record = next(record for record in payload.records if len(record.carriers) == 4)
    missing_heavy = replace(record, carriers=record.carriers[:3])
    set_stereo_payload(mol, replace(payload, records=(missing_heavy,)))
    graph = _graph(mol)

    with pytest.raises(ValueError, match='carrier count'):
        mol.to_rdmol()
    assert _graph(mol) == graph
    assert get_stereo_payload(mol).records == (missing_heavy,)
