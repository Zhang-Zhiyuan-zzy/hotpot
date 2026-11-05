"""Independent external task values and carrier rollback at conversion boundaries."""
from dataclasses import astuple
import numpy as np
import pytest
from openbabel import openbabel as ob
from rdkit import Chem

from hotpot import Molecule, read_mol
from hotpot.cheminfo import BondKind
from hotpot.cheminfo.adapters.openbabel import perception as ob_perception
from hotpot.cheminfo.operations import chemistry, conversion
from hotpot.cheminfo.structure._builder import _freeze_value
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._results import _import_partial_charges


def _facts(mol):
    record, metadata = mol._capture_factual_record()
    return astuple(record), _freeze_value(metadata)


def _prepared(text='CCO'):
    mol = read_mol(text, 'smi')
    mol.coordinates = np.arange(len(mol.atoms) * 3, dtype=float).reshape(-1, 3) / 8
    mol.to_rdmol()
    mol.to_obmol()
    return mol


def _ob_task_values(molecule):
    return (
        tuple(
            (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetImplicitHCount(),
             atom.GetIsotope(), atom.GetX(), atom.GetY(), atom.GetZ())
            for atom in ob.OBMolAtomIter(molecule)
        ),
        tuple(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondOrder(), bond.IsAromatic())
            for bond in ob.OBMolBondIter(molecule)
        ),
    )


@pytest.mark.parametrize('text,charge,total_hydrogens,isotopes', [
    ('CC(=O)O', 0, 4, ()),
    ('CC(=O)[O-]', -1, 3, ()),
    ('O=CO', 0, 2, ()),
    ('OC(=O)c1ccccc1', 0, 6, ()),
    ('CC(=O)OC', 0, 6, ()),
    ('c1ccccc1', 0, 6, ()),
    ('c1cc[nH]c1', 0, 5, ()),
    ('[NH4+]', 1, 4, ()),
    ('C[N+](=O)[O-]', 0, 3, ()),
    ('[13CH3]C(=O)O', 0, 4, ((6, 13),)),
    ('[Na+].CC(=O)[O-]', 0, 3, ()),
])
@pytest.mark.parametrize('property_name', ['smiles', 'kekulize_smiles'])
def test_smiles_preserves_graph_charge_hydrogens_isotopes_and_source_facts(
    text, charge, total_hydrogens, isotopes, property_name,
):
    molecule = _prepared(text)
    expected = Chem.MolFromSmiles(text)
    prior_ob, prior_mapping = conversion.mol2obmol(molecule)
    prior_rdkit = molecule.to_rdmol()
    ob_values = _ob_task_values(prior_ob)
    rdkit_binary = prior_rdkit.ToBinary(Chem.PropertyPickleOptions.AllProps)
    before = _facts(molecule)

    exported = Chem.MolFromSmiles(getattr(molecule, property_name))

    assert exported is not None
    assert Chem.GetFormalCharge(exported) == charge
    assert sum(atom.GetTotalNumHs() for atom in exported.GetAtoms()) == total_hydrogens
    assert tuple(sorted(
        (atom.GetAtomicNum(), atom.GetIsotope())
        for atom in exported.GetAtoms() if atom.GetIsotope()
    )) == isotopes
    # RDKit supplies an independent graph comparison; canonical source row order
    # need not match Hotpot's original rows or the Open Babel writer's order.
    assert Chem.MolToSmiles(exported) == Chem.MolToSmiles(expected)
    assert _facts(molecule) == before
    assert _ob_task_values(prior_ob) == ob_values
    assert prior_rdkit.ToBinary(Chem.PropertyPickleOptions.AllProps) == rdkit_binary
    fresh_ob, fresh_mapping = conversion.mol2obmol(molecule)
    assert fresh_ob is not prior_ob and fresh_mapping is not prior_mapping
    assert prior_mapping == fresh_mapping == {
        row: row + 1 for row in range(len(molecule.atoms))
    }


@pytest.mark.parametrize('text', ['F[C@](Cl)(Br)I', 'F/C=C/F'])
def test_smiles_export_does_not_change_the_existing_stereo_ledger(text):
    molecule = conversion.to_hotpot_mol(Chem.MolFromSmiles(text))
    molecule.to_obmol()
    molecule.to_rdmol()
    before = _facts(molecule)
    assert molecule._capture_factual_record()[0].stereo_payload.records

    molecule.smiles
    molecule.kekulize_smiles

    assert _facts(molecule) == before


@pytest.mark.parametrize('text', ['CCO', '[NH4+]', 'c1ccccc1', 'F[C@](Cl)(Br)I', 'F/C=C/F', '[13CH3][O-]'])
def test_ob_projection_and_mapping_are_fresh_without_carrier_storage(text):
    mol = _prepared(text)
    before = _facts(mol)
    first, first_map = conversion.mol2obmol(mol)
    second, second_map = conversion.mol2obmol(mol)
    public = mol.to_obmol()
    assert first is not second and second is not public and first is not public
    assert first_map is not second_map
    assert first_map == second_map == {row: row + 1 for row in range(len(mol.atoms))}
    first_map.clear()
    first.GetAtom(1).SetAtomicNum(2)
    first.GetAtom(1).SetVector(99, 98, 97)
    assert second_map == {row: row + 1 for row in range(len(mol.atoms))}
    assert second.GetAtom(1).GetAtomicNum() == mol.atoms[0].atomic_number
    assert public.GetAtom(1).GetX() == mol.coordinates[0, 0]
    assert _facts(mol) == before
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


@pytest.mark.parametrize('kind', ['openbabel', 'pybel', 'rdkit'])
def test_external_mutation_does_not_change_other_values_or_complete_carrier_facts(kind):
    mol = _prepared()
    before = _facts(mol)
    if kind == 'rdkit':
        first, second = mol.to_rdmol(), mol.to_rdmol()
        first.GetAtomWithIdx(0).SetAtomicNum(2)
        first.SetProp('caller_owned', 'first-only')
        first.GetConformer().SetAtomPosition(0, (90., 91., 92.))
        assert second.GetAtomWithIdx(0).GetAtomicNum() == 6
        assert not second.HasProp('caller_owned')
        assert tuple(second.GetConformer().GetAtomPosition(0)) == tuple(mol.coordinates[0])
    else:
        method = mol.to_pybel_mol if kind == 'pybel' else mol.to_obmol
        first, second = method(), method()
        assert first is not second
        left, right = (first.OBMol, second.OBMol) if kind == 'pybel' else (first, second)
        assert left is not right
        left.GetAtom(1).SetAtomicNum(2)
        left.GetAtom(1).SetPartialCharge(9.)
        left.GetAtom(1).SetVector(90., 91., 92.)
        assert right.GetAtom(1).GetAtomicNum() == 6
        restored = conversion.obmol2mol(right, Molecule())
        assert restored.atoms[0].partial_charge == mol.atoms[0].partial_charge
        assert right.GetAtom(1).GetX() == mol.coordinates[0, 0]
    assert _facts(mol) == before
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


def test_carrier_mutation_changes_future_projection_without_rewriting_old_task_value():
    mol = _prepared()
    prior = mol.to_obmol()
    charge = prior.GetAtom(1).GetFormalCharge()
    mol.atoms[0].formal_charge = charge + 1
    mol.atoms[0].x = 7.5
    current = mol.to_obmol()
    assert prior.GetAtom(1).GetFormalCharge() == charge
    assert prior.GetAtom(1).GetX() == 0.
    assert current.GetAtom(1).GetFormalCharge() == charge + 1
    assert current.GetAtom(1).GetX() == 7.5


def test_transaction_rollback_restores_complete_facts_without_invalidating_held_values():
    mol = _prepared()
    before = _facts(mol)
    prior = mol.to_obmol()
    with pytest.raises(RuntimeError, match='rollback after projection'):
        with _FactTransaction(mol, producer=13):
            mol.atoms[0].formal_charge = 2
            mol.atoms[0].x = 8.
            staged = mol.to_obmol()
            raise RuntimeError('rollback after projection')
    assert _facts(mol) == before
    assert prior.GetAtom(1).GetFormalCharge() == 0
    assert prior.GetAtom(1).GetX() == 0.
    assert staged.GetAtom(1).GetFormalCharge() == 2
    assert staged.GetAtom(1).GetX() == 8.
    current = mol.to_obmol()
    assert current.GetAtom(1).GetX() == 0.
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


def test_partial_charge_assignment_keeps_previous_external_values_owned_by_caller(monkeypatch):
    mol = _prepared()
    old = mol.to_obmol()
    old_charges = tuple(atom.GetPartialCharge() for atom in ob.OBMolAtomIter(old))
    supplied = (.25, -.5, .25)
    monkeypatch.setattr(Molecule, 'get_partial_charge', lambda self, *args, **kwargs: supplied)
    mol.assign_partial_charge()
    assert tuple(atom.GetPartialCharge() for atom in ob.OBMolAtomIter(old)) == old_charges
    current = mol.to_obmol()
    # Inspect the transported scalars without Open Babel lazily calculating
    # another charge model on this caller-owned projection.
    current.SetPartialChargesPerceived(True)
    assert tuple(atom.GetPartialCharge() for atom in ob.OBMolAtomIter(current)) == supplied
    assert tuple(atom.partial_charge for atom in mol.atoms) == supplied


def test_ob_capture_empty_carrier_has_no_rows_or_cell():
    assert conversion._capture_ob_values(Molecule(), ()) == ((), (), None)


@pytest.mark.parametrize('partial,coordinates,aromatic', [
    (-0.0, (-0.0, 1.25, -2.5), 2.5),
    (.125, (float('-inf'), -0.0, 0.0), -0.0),
    (-.25,
     (np.frombuffer(bytes.fromhex('020000000000f87f'), dtype='<f8')[0], 3.5, -0.0), 1.0),
])
def test_ob_capture_preserves_scalar_codecs_raw_aromatic_and_float_bytes(partial, coordinates, aromatic):
    # The literal integer expectations distinguish scalar codecs from raw slots;
    # nonfinite values are inspected before any external toolkit packing.
    raw = np.zeros(19)
    raw[[0, 7, 9]] = (6.875, -1.875, aromatic)
    raw[10:13] = coordinates
    mol = Molecule()
    mol._create_atom_from_array(raw)
    parameters = (11.25, 12.5, 13.75, 88.5, 91.25, 103.75)
    original_cell = mol.create_crystal(*parameters)
    _import_partial_charges(mol, (partial,), producer=1)
    before = mol._atom_table.matrix().tobytes()
    atom_rows, bond_rows, cell = conversion._capture_ob_values(mol, ())
    assert type(atom_rows) is tuple and len(atom_rows) == 1
    row = atom_rows[0]
    assert type(row) is tuple and len(row) == 5
    assert row[:2] == (6, -1)
    assert tuple(type(value) for value in row[:3]) == (int, int, float)
    assert type(row[3]) is tuple and tuple(type(value) for value in row[3]) == (float,) * 3
    assert type(row[4]) is float
    assert np.asarray((row[2], *row[3], row[4]), dtype='<f8').tobytes() == np.asarray(
        (partial, *coordinates, aromatic), dtype='<f8').tobytes()
    assert bond_rows == ()
    assert cell is not original_cell
    assert tuple(getattr(cell, name) for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')) == parameters
    original_cell.a = 99.
    assert cell.a == parameters[0]
    assert mol._atom_table.matrix().tobytes() == before


@pytest.mark.parametrize('order', [
    -0.0, 1.25, float('inf'), float('-inf'),
    np.frombuffer(bytes.fromhex('030000000000f87f'), dtype='<f8')[0],
])
def test_ob_capture_keeps_raw_bond_order_float_and_directed_endpoint_order(order):
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=8)
    bond = mol.add_bond(1, 0, order, bond_kind=BondKind.UNKNOWN)
    _atoms, bonds, cell = conversion._capture_ob_values(mol, (bond,))
    assert type(bonds) is tuple and len(bonds) == 1
    row = bonds[0]
    assert row[:3] == ('unknown', 1, 0)
    assert tuple(type(value) for value in row) == (str, int, int, float, bool)
    assert np.asarray((row[3],), dtype='<f8').tobytes() == np.asarray((order,), dtype='<f8').tobytes()
    assert row[4] is False and cell is None


def test_ob_capture_tracks_removed_replaced_and_reordered_members():
    mol = Molecule()
    for number, x in ((1, 9.), (6, 1.), (7, 2.), (8, 3.)):
        mol.create_atom(atomic_number=number, formal_charge=0, coordinates=(x, 0., 0.))
    first = mol.add_bond(3, 1, 1.)
    second = mol.add_bond(1, 2, 2.)
    detached = mol.replace_atom(2, 6)
    mol.atoms[2].coordinates = (4., 0., 0.)
    mol.remove_atom(0)
    atoms, bonds, _cell = conversion._capture_ob_values(mol, (second, first))
    assert tuple(row[0] for row in atoms) == (6, 6, 8)
    assert tuple(row[3] for row in atoms) == ((1., 0., 0.), (4., 0., 0.), (3., 0., 0.))
    assert bonds == (('double', 0, 1, 2., False), ('single', 2, 0, 1., False))
    assert detached.atomic_number == 7 and detached.coordinates == (2., 0., 0.)
    record, metadata = mol._capture_factual_record(rows=(2, 0, 1), include_frames=False)
    reordered = Molecule()._install_molecule(record, bond_metadata=metadata)
    atoms, bonds, _cell = conversion._capture_ob_values(reordered, tuple(reversed(reordered.bonds)))
    assert atoms == (
        (8, 0, None, (3., 0., 0.), 0.),
        (6, 0, None, (1., 0., 0.), 0.),
        (6, 0, None, (4., 0., 0.), 0.),
    )
    assert bonds == (('double', 1, 2, 2., False), ('single', 0, 1, 1., False))


def test_ob_capture_uses_complete_public_graph_or_explicit_perception_rows(monkeypatch):
    mol = Molecule()
    for number in (6, 7, 8, 6, 30):
        mol.create_atom(atomic_number=number, formal_charge=0)
    first = mol.add_bond(2, 0, 2.)
    hidden_covalent = mol.add_bond(0, 1, 1.)
    second = mol.add_bond(1, 3, 1.)
    active_metal = mol.add_bond(3, 4, 1.)
    hidden_metal = mol.add_bond(2, 4, 1.)
    captures = []
    capture = conversion._capture_ob_values
    def observe(owner, bonds):
        result = capture(owner, bonds)
        captures.append((tuple(bonds), result[1]))
        return result
    monkeypatch.setattr(conversion, '_capture_ob_values', observe)
    public, mapping = conversion.mol2obmol(mol)
    assert public.NumAtoms() == 5 and public.NumBonds() == 5
    assert mapping == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    assert captures == [((first, hidden_covalent, second, active_metal, hidden_metal), (
        ('double', 2, 0, 2., False), ('single', 0, 1, 1., False),
        ('single', 1, 3, 1., False), ('single', 3, 4, 1., False), ('single', 2, 4, 1., False)))]
    observed = []
    marker = object()
    def perceive(obmol, idx_to_row):
        observed.append((obmol.NumAtoms(), obmol.NumBonds(), idx_to_row))
        return marker
    monkeypatch.setattr(ob_perception, '_perceive_bond_orders', perceive)
    assert chemistry._perceive_bond_orders(mol, (second, first)) is marker
    assert captures[1] == ((second, first), (
        ('single', 1, 3, 1., False), ('double', 2, 0, 2., False)))
    assert observed == [(5, 2, {1: 0, 2: 1, 3: 2, 4: 3, 5: 4})]
    assert mol.bonds == [first, hidden_covalent, second, active_metal, hidden_metal]


def test_ob_capture_first_dative_does_not_read_later_bond_or_cell(monkeypatch):
    mol = Molecule()
    mol.create_atom(atomic_number=7)
    mol.create_atom(atomic_number=6)
    dative = mol.add_bond(0, 1, bond_kind=BondKind.DATIVE, bond_direction='atom1_to_atom2')
    def unread_cell(owner):
        raise AssertionError('cell was read after first dative')
    monkeypatch.setattr(Molecule, 'crystal', property(unread_cell))
    # object() has no Bond fields: any attempt to capture the later entry fails.
    atoms, bonds, cell = conversion._capture_ob_values(mol, (dative, object()))
    assert len(atoms) == 2
    assert bonds == (('dative',),) and cell is None
