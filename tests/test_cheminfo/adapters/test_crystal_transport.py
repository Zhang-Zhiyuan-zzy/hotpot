"""Six factual cell scalars, external transport and the retained pickle envelope."""
import copy
import pickle

import numpy as np
import pytest
from openbabel import openbabel as ob

from hotpot import read_mol
from hotpot.cheminfo.adapters.openbabel.crystal import to_obcell
from hotpot.cheminfo.operations import conversion
from hotpot.cheminfo.structure.crystal import Crystal
from hotpot.cheminfo.structure._state import _restore_molecule_extra
from tests.test_cheminfo._stage5_fixtures import immutable_snapshot

NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
VALUES = (5., 6., 7., 75., 85., 95.)


def _facts(mol):
    record, metadata = mol._capture_factual_record()
    return immutable_snapshot(record), immutable_snapshot(metadata)


def _parameters(cell):
    return tuple(getattr(cell, name) for name in NAMES)


def _molecule():
    mol = read_mol('CCO', 'smi')
    mol.create_crystal(*VALUES)
    mol.coordinates = np.arange(9.).reshape(3, 3) / 8
    return mol


def test_crystal_has_only_six_initial_ordinary_writable_values():
    cell = Crystal(*VALUES)
    assert cell.__module__ == 'hotpot.cheminfo.structure.crystal'
    assert tuple(vars(cell)) == NAMES
    assert _parameters(cell) == VALUES
    cell.a = 12.
    cell.note = 'ordinary user attribute'
    assert cell.a == 12. and cell.note == 'ordinary user attribute'
    assert not any(hasattr(cell, name) for name in ('mol', '_obcell', 'obcell', 'matrix'))
    with pytest.raises(TypeError):
        Crystal(*VALUES, mol=object())


def test_oblique_row_angle_convention_and_subclass_dispatch_are_preserved():
    matrix = np.array(((1., 0., 0.), (.5, np.sqrt(3.) / 2., 0.), (0., 0., 2.)))
    np.testing.assert_allclose(_parameters(Crystal.from_matrix(matrix)), (1., 1., 2., 60., 90., 90.), atol=1e-12)
    class Specialized(Crystal):
        @staticmethod
        def _matrix_to_params(value):
            return *np.diag(value), 80., 85., 95.
    cell = Specialized.from_matrix(np.diag((2., 3., 5.)))
    assert type(cell) is Specialized
    assert _parameters(cell) == (2., 3., 5., 80., 85., 95.)


def test_ob_unit_cells_are_fresh_and_keep_exact_unrenamed_angle_order():
    cell = Crystal(*VALUES)
    first, second = to_obcell(cell), to_obcell(cell)
    assert first is not second
    getters = ('GetA', 'GetB', 'GetC', 'GetAlpha', 'GetBeta', 'GetGamma')
    assert tuple(getattr(second, name)() for name in getters) == pytest.approx(VALUES)
    first.SetData(1., 2., 3., 90., 90., 90.)
    assert _parameters(cell) == VALUES
    assert tuple(getattr(second, name)() for name in getters) == pytest.approx(VALUES)


def test_ob_molecule_cell_roundtrip_uses_factual_cell_without_attached_backend():
    mol = _molecule()
    value, _ = conversion.mol2obmol(mol)
    unit = ob.toUnitCell(value.GetData(ob.UnitCell))
    assert tuple(getattr(unit, name)() for name in ('GetA', 'GetB', 'GetC', 'GetAlpha', 'GetBeta', 'GetGamma')) == pytest.approx(VALUES)
    restored = conversion.obmol2mol(value, type(mol)())
    assert _parameters(restored.crystal) == pytest.approx(VALUES)
    assert tuple(vars(restored.crystal)) == NAMES
    unit.SetData(1., 1., 1., 90., 90., 90.)
    assert _parameters(mol.crystal) == VALUES
    assert _parameters(restored.crystal) == pytest.approx(VALUES)


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda mol: pickle.loads(pickle.dumps(mol, protocol=5))])
def test_cell_copy_and_pickle_preserve_complete_facts_and_no_host_backend_state(clone):
    mol = _molecule()
    restored = clone(mol)
    assert _facts(restored) == _facts(mol)
    assert restored.crystal is not mol.crystal
    assert tuple(vars(restored.crystal)) == NAMES
    assert not {'_obmol', '_row2idx'} & restored.__dict__.keys()


def test_molecule_pickle_extra_keeps_four_slots_and_cell_in_factual_record():
    mol = _molecule()
    state = mol.__getstate__()
    extra = state[-1]
    assert len(extra) == 4
    attributes, metadata, atom_state, bond_state = extra
    assert state[0].cell == VALUES
    assert '_crystal' not in attributes
    assert immutable_snapshot(metadata) == immutable_snapshot(mol._capture_factual_record()[1])
    assert len(atom_state) == len(mol.atoms)
    assert len(bond_state) == len(mol.bonds)
    restored = pickle.loads(pickle.dumps(mol, protocol=5))
    before = _facts(restored)
    assert before == _facts(mol)
    cell = restored.crystal
    _restore_molecule_extra(restored, restored.__getstate__()[-1])
    assert _facts(restored) == before
    assert restored.crystal is cell
    assert _parameters(restored.crystal) == VALUES
    assert tuple(vars(restored.crystal)) == NAMES
    assert not {'_obmol', '_row2idx'} & restored.__dict__.keys()
