"""Declared configurations must control direct-native built geometry."""
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build, single_optimize
from hotpot.cheminfo.obWrappers.native import _native_module, _native_molecule_data
from hotpot.cheminfo.stereo import get_stereo_payload, set_stereo_payload


CASE54 = 'C[C@H](CC(C)(C)C)CP(=S)(S)C[C@H](C)CC(C)(C)C'


def _geometry_stereo(mol):
    """Use only graph and coordinates; erase all supplied stereo declarations."""
    from hotpot.cheminfo.operations.chemistry import _require_non_graph_hydrogens

    graph = Chem.RWMol()
    hydrogens = _require_non_graph_hydrogens(mol)
    for atom, hydrogen_count in zip(mol.atoms, hydrogens):
        value = Chem.Atom(atom.atomic_number)
        value.SetFormalCharge(atom.formal_charge)
        value.SetNumExplicitHs(hydrogen_count)
        value.SetNoImplicit(True)
        isotope = getattr(atom, 'isotope', None)
        if isotope is not None:
            value.SetIsotope(isotope)
        graph.AddAtom(value)
    bond_types = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE,
                  3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
    for bond in mol.bonds:
        graph.AddBond(bond.atom1.idx, bond.atom2.idx, bond_types[bond.bond_order])
    projected = graph.GetMol()
    Chem.SanitizeMol(projected)
    coordinates = Chem.Conformer(len(mol.atoms))
    for row, position in enumerate(mol.coordinates):
        coordinates.SetAtomPosition(row, position)
    projected.AddConformer(coordinates)
    Chem.RemoveStereochemistry(projected)
    Chem.AssignStereochemistryFrom3D(projected, replaceExistingTags=True)
    Chem.AssignStereochemistry(projected, force=True, cleanIt=True)
    atoms = tuple((atom.GetIdx(), atom.GetProp('_CIPCode'))
                  for atom in projected.GetAtoms() if atom.HasProp('_CIPCode'))
    bonds = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetStereo()))
                  for bond in projected.GetBonds() if bond.GetStereo() in (
                      Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ))
    return atoms, bonds


def _declared_stereo(text):
    source = Chem.MolFromSmiles(text)
    atoms = tuple((atom.GetIdx(), atom.GetProp('_CIPCode'))
                  for atom in source.GetAtoms() if atom.HasProp('_CIPCode'))
    bonds = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetStereo()))
                  for bond in source.GetBonds() if bond.GetStereo() in (
                      Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ))
    return atoms, bonds


def _graph(mol):
    return (tuple((atom.atomic_number, atom.formal_charge) for atom in mol.atoms),
            tuple((bond.atom1.idx, bond.atom2.idx, bond.bond_order, bond.bond_kind)
                  for bond in mol.bonds))


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl', 'F/C=C/F', r'F/C=C\F', CASE54))
@pytest.mark.parametrize('materialize_h', (False, True))
def test_native_builder_geometry_matches_declared_stereo(text, materialize_h):
    mol = read_mol(text)
    if materialize_h:
        mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    before_payload, before_graph = get_stereo_payload(mol), _graph(mol)
    _native_module().seed_random(42)

    report = build(mol)

    assert report.succeeded
    assert _geometry_stereo(mol) == _declared_stereo(text)
    assert get_stereo_payload(mol) == before_payload
    assert _graph(mol) == before_graph


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl', 'F/C=C/F', r'F/C=C\F'))
def test_public_native_optimization_retains_built_stereo(text):
    mol = read_mol(text)
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    assert build(mol).succeeded
    payload, graph = get_stereo_payload(mol), _graph(mol)

    single_optimize(mol, forcefield='UFF', steps=10)

    assert _geometry_stereo(mol) == _declared_stereo(text)
    assert get_stereo_payload(mol) == payload
    assert _graph(mol) == graph


@pytest.mark.parametrize('namespace,code', (('future', 'TH/CW/1'), ('openbabel-stereo/1', 'TH/FUTURE/1')))
def test_private_native_boundary_rejects_unregistered_stereo(namespace, code):
    mol = read_mol('C[C@H](F)Cl')
    payload = get_stereo_payload(mol)
    record = next(record for record in payload.records if record.carriers)
    set_stereo_payload(mol, replace(payload, records=(replace(
        record, source_namespace=namespace, source_code=code, descriptor=code),)))
    with pytest.raises(ValueError, match='namespace|code'):
        _native_molecule_data(mol)


def test_native_input_owns_the_captured_declared_configuration():
    mol = read_mol('C[C@H](F)Cl')
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    captured = _native_molecule_data(mol)
    payload = get_stereo_payload(mol)
    set_stereo_payload(mol, replace(payload, records=()))

    result = _native_module().build(captured)
    mol.coordinates = result.coordinates

    assert result.succeeded
    assert _geometry_stereo(mol)[0] == ((1, 'R'),)
    assert get_stereo_payload(mol).records == ()


def test_projection_and_numeric_packing_share_the_mutation_lock(monkeypatch):
    from hotpot.cheminfo.obWrappers import native as boundary
    from hotpot.cheminfo.adapters.openbabel import stereo_transport as codec

    mol = read_mol('C[C@H](F)Cl')
    pack, project = boundary._pack_molecule, codec._project_stereo_payload
    phases = []

    def packed(*args, **kwargs):
        assert mol._mutation_lock._is_owned()
        phases.append('packed')
        return pack(*args, **kwargs)

    def projected(*args, **kwargs):
        assert mol._mutation_lock._is_owned()
        phases.append('projected')
        return project(*args, **kwargs)

    monkeypatch.setattr(boundary, '_pack_molecule', packed)
    monkeypatch.setattr(codec, '_project_stereo_payload', projected)
    _native_molecule_data(mol)
    assert phases == ['packed', 'projected']


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl', 'F/C=C/F', r'F/C=C\F'))
def test_openbabel_source_configs_reach_the_direct_native_builder(text, monkeypatch):
    from hotpot.cheminfo.operations.conversion import to_hotpot_mol
    from hotpot.cheminfo.adapters.openbabel._runtime import pybel

    mol = to_hotpot_mol(pybel.readstring('smi', text).OBMol)
    payload = get_stereo_payload(mol)
    graph = _graph(mol)

    def no_python_ob_projection(*args, **kwargs):
        pytest.fail('native builder must not use a SWIG molecule projection')

    monkeypatch.setattr(type(mol), 'to_obmol', no_python_ob_projection)
    assert build(mol).succeeded
    assert _geometry_stereo(mol) == _declared_stereo(text)
    assert get_stereo_payload(mol) == payload
    assert _graph(mol) == graph


@pytest.mark.parametrize('seed', (42, 43, 44))
def test_seeded_public_workflow_preserves_case54_stereo(seed):
    from hotpot.cheminfo.operations.forcefields import build3d

    mol = read_mol(CASE54)
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(seed))
    payload, graph = get_stereo_payload(mol), _graph(mol)

    report = build3d(mol, add_hydrogens=False, seed=seed)

    assert report.atom_count == len(mol.atoms)
    assert _geometry_stereo(mol)[0] == ((1, 'R'), (12, 'R'))
    assert get_stereo_payload(mol) == payload
    assert _graph(mol) == graph


def test_undeclared_native_path_does_not_prepare_h_or_reconstruct_payload(monkeypatch):
    from hotpot.cheminfo.operations import chemistry
    from hotpot.cheminfo.structure import _stereo

    mol = read_mol('CCO')
    assert mol._stereo_records == ()

    def forbidden(*args, **kwargs):
        pytest.fail('undeclared numeric native input must not prepare stereo inventory')

    monkeypatch.setattr(chemistry, '_require_non_graph_hydrogens', forbidden)
    monkeypatch.setattr(_stereo, '_get_stereo_payload', forbidden)
    assert _native_molecule_data(mol).atom_count == 3


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl', 'F/C=C/F', r'F/C=C\F'))
@pytest.mark.parametrize('materialize_h', (False, True))
def test_rdkit_source_configs_reach_the_direct_native_builder(text, materialize_h):
    from hotpot.cheminfo.operations.conversion import to_hotpot_mol

    mol = to_hotpot_mol(Chem.MolFromSmiles(text))
    if materialize_h:
        mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    payload, graph = get_stereo_payload(mol), _graph(mol)
    assert any(record.source_namespace == 'rdkit-2026.09.1' for record in payload.records)

    assert build(mol).succeeded
    assert _geometry_stereo(mol) == _declared_stereo(text)
    if materialize_h:
        single_optimize(mol, forcefield='UFF', steps=10)
        assert _geometry_stereo(mol) == _declared_stereo(text)
    assert get_stereo_payload(mol) == payload
    assert _graph(mol) == graph


@pytest.mark.parametrize('text,excluded_endpoints,message', (
    ('C[C@H](F)Cl', (1, 2), 'target neighbor count'),
    ('F/C=C/F', (1, 2), 'double bond'),
))
def test_selected_native_input_rejects_missing_stereo_graph(text, excluded_endpoints, message):
    from hotpot.cheminfo.structure._selection import _capture_topology_selection

    mol = read_mol(text)
    excluded = next(bond for bond in mol.bonds
                    if {bond.atom1.idx, bond.atom2.idx} == set(excluded_endpoints))
    selection = _capture_topology_selection(mol, excluded_bonds=(excluded,))
    payload, graph = get_stereo_payload(mol), _graph(mol)
    with pytest.raises(ValueError, match=message):
        _native_molecule_data(mol, selection=selection)
    assert get_stereo_payload(mol) == payload
    assert _graph(mol) == graph


def test_selected_native_input_keeps_unaffected_stereo():
    from hotpot.cheminfo.structure._selection import _capture_topology_selection

    mol = read_mol('C[C@H](F)Cl.CC', fmt='smi')
    selection = _capture_topology_selection(mol, excluded_bonds=(mol.bonds[-1],))
    data = _native_molecule_data(mol, selection=selection)
    assert data.atom_count == len(mol.atoms)
    assert data.bond_count == len(mol.bonds) - 1


@pytest.mark.parametrize('limitation', ('group', 'carrier'))
def test_private_native_boundary_preserves_adapter_rejections(limitation):
    from hotpot.cheminfo.stereo import (
        StereoCarrier, StereoCarrierKind, StereoGroupKind, StereoGroupRef,
    )

    mol = read_mol('C[C@H](F)Cl')
    payload = get_stereo_payload(mol)
    record = payload.records[0]
    if limitation == 'group':
        record = replace(record, group=StereoGroupRef('scope', StereoGroupKind.AND, 1))
    else:
        record = replace(record, carriers=(StereoCarrier(StereoCarrierKind.VACANCY), *record.carriers[1:]))
    set_stereo_payload(mol, replace(payload, records=(record,)))
    with pytest.raises(ValueError) as adapter_error:
        mol.to_obmol()
    with pytest.raises(ValueError) as native_error:
        _native_molecule_data(mol)
    assert str(native_error.value) == str(adapter_error.value)


def test_rdkit_three_carriers_do_not_bind_a_remaining_heavy_neighbor():
    from hotpot.cheminfo.operations.conversion import to_hotpot_mol

    mol = to_hotpot_mol(Chem.MolFromSmiles('F[C@](Cl)(Br)I'))
    payload = get_stereo_payload(mol)
    record = next(record for record in payload.records if len(record.carriers) == 4)
    record = replace(record, carriers=record.carriers[:3])
    set_stereo_payload(mol, replace(payload, records=(record,)))
    graph = _graph(mol)

    with pytest.raises(ValueError, match='carriers') as adapter_error:
        mol.to_obmol()
    with pytest.raises(ValueError, match='carriers') as native_error:
        _native_molecule_data(mol)
    assert str(native_error.value) == str(adapter_error.value)
    assert _graph(mol) == graph
    assert get_stereo_payload(mol).records == (record,)


@pytest.mark.parametrize('text', ('C[C@H](F)Cl', 'C[C@@H](F)Cl'))
def test_shared_adapter_projection_binds_materialized_rdkit_h_with_the_same_order(text):
    from hotpot.cheminfo.adapters.openbabel._runtime import ob
    from hotpot.cheminfo.adapters.openbabel.stereo_transport import _project_stereo_payload
    from hotpot.cheminfo.obWrappers.packing import _pack_molecule
    from hotpot.cheminfo.operations.chemistry import _require_non_graph_hydrogens
    from hotpot.cheminfo.operations.conversion import to_hotpot_mol

    mol = to_hotpot_mol(Chem.MolFromSmiles(text))
    source_carriers = next(record.carriers for record in get_stereo_payload(mol).records
                           if len(record.carriers) == 3)
    mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(42))
    payload, graph = get_stereo_payload(mol), _graph(mol)
    buffers = _pack_molecule(mol)
    bonds = ((int(a), int(b), float(order))
             for (a, b), order in zip(buffers.bond_indices, buffers.bond_orders))
    configs = _project_stereo_payload(payload, dict(enumerate(buffers.atomic_numbers)),
                                     bonds, _require_non_graph_hydrogens(mol))
    config = next(config for config in configs if config.family == 'TH')
    assert config.references[:3] == tuple(carrier.atom for carrier in source_carriers)
    assert mol.atoms[config.references[3]].atomic_number == 1
    assert config.references[3] in tuple(atom.idx for atom in mol.atoms[1].neighbours)

    projected = mol.to_obmol()
    assert ob.OBBuilder().Build(projected)
    mol.coordinates = np.array([(atom.GetX(), atom.GetY(), atom.GetZ())
                                for atom in ob.OBMolAtomIter(projected)])
    assert _geometry_stereo(mol) == _declared_stereo(text)
    assert get_stereo_payload(mol) == payload
    assert _graph(mol) == graph
