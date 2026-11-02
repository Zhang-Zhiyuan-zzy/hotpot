"""Native coordinate updates preserve buffers, conformers and partial errors."""
import gc

import numpy as np
import pytest
from openbabel import openbabel as ob

from hotpot import read_mol
from hotpot.cheminfo.adapters.openbabel.conversion import (
    extract_obmol_coordinates,
    set_obmol_coordinates,
)


def _state(mol):
    flags = ('HasChiralityPerceived', 'HasHybridizationPerceived',
             'HasRingAtomsAndBondsPerceived', 'HasSSSRPerceived',
             'HasAromaticPerceived', 'HasPartialChargesPerceived',
             'HasClosureBondsPerceived')
    return (mol.GetMod(), mol.NumConformers(),
            tuple(getattr(mol, flag)() for flag in flags),
            tuple((data.GetAttribute(), data.GetDataType(), data.GetOrigin(), data.GetValue())
                  for data in mol.GetData()))


@pytest.mark.parametrize('layout', ['c', 'fortran', 'sliced', 'ieee'])
@pytest.mark.parametrize('perceived', [False, True])
def test_update_copies_exact_values_into_current_conformer_only(layout, perceived):
    mol = read_mol('CCC', 'smi').to_obmol()
    if perceived:
        ob.PerceiveStereo(mol)
    original = extract_obmol_coordinates(mol).copy()
    native_frame = ob.double_array(np.arange(9, dtype=float))
    mol.AddConformer(native_frame)
    native_frame.thisown = False
    mol.SetConformer(1)
    pointer, before = int(mol.GetCoordinates()), _state(mol)
    values = np.arange(9, dtype=float).reshape(3, 3) + .25
    if layout == 'fortran':
        values = np.asfortranarray(values)
    elif layout == 'sliced':
        values = np.repeat(values, 2, axis=1)[:, ::2]
    elif layout == 'ieee':
        bits = [0x8000000000000000, 0x7ff8000000000042, 0x7ff0000000000000,
                0xfff0000000000000, 0, 0x3ff0000000000000, 1, 2, 3]
        values = np.array(bits, dtype=np.uint64).view(np.float64).reshape(3, 3)
    expected_bits = values.copy().view(np.uint64)

    assert set_obmol_coordinates(mol, values) is None
    values[:] = 99.
    del values
    gc.collect()

    assert int(mol.GetCoordinates()) == pointer
    assert _state(mol) == before
    np.testing.assert_array_equal(extract_obmol_coordinates(mol).view(np.uint64), expected_bits)
    mol.SetConformer(0)
    np.testing.assert_array_equal(extract_obmol_coordinates(mol), original)
    mol.SetConformer(1)
    np.testing.assert_array_equal(extract_obmol_coordinates(mol).view(np.uint64), expected_bits)


@pytest.mark.parametrize('modifying', [False, True])
def test_update_without_buffer_preserves_modification_and_conformer_lifecycle(modifying):
    mol = ob.OBMol()
    if modifying:
        mol.BeginModify()
    for _ in range(3):
        mol.NewAtom().SetAtomicNum(6)
    assert not mol.GetCoordinates()
    before = _state(mol)
    values = np.arange(9, dtype=float).reshape(3, 3)

    set_obmol_coordinates(mol, values)

    assert _state(mol) == before
    assert not mol.GetCoordinates()
    np.testing.assert_array_equal(extract_obmol_coordinates(mol), values)
    if modifying:
        mol.EndModify()
        np.testing.assert_array_equal(extract_obmol_coordinates(mol), values)


def test_invalid_object_keeps_original_partial_writes_and_error():
    mol = read_mol('CCC', 'smi').to_obmol()
    before = _state(mol)
    expected = extract_obmol_coordinates(mol)
    values = (np.arange(9, dtype=float).reshape(3, 3) + 10).astype(object)
    expected[:2] = values[:2]
    values[-1, 0] = object()

    with pytest.raises(TypeError, match="in method 'OBAtom_SetVector'"):
        set_obmol_coordinates(mol, values)

    assert _state(mol) == before
    np.testing.assert_array_equal(extract_obmol_coordinates(mol), expected)
