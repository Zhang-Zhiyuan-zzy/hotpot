"""Contracts for the consumer-neutral field-selective molecule payload."""

from __future__ import annotations

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo._native.buffers import (
    NativeBondKindCode,
    NativeMoleculeBuffers,
    NativeMoleculeField,
    pack_molecule,
)


def test_empty_selection_preserves_dimensions_without_materializing_fields():
    mol = read_mol("CCO")

    buffers = pack_molecule(mol, frozenset())

    assert isinstance(buffers, NativeMoleculeBuffers)
    assert buffers.atom_count == len(mol.atoms)
    assert buffers.bond_count == len(mol.bonds)
    assert all(
        getattr(buffers, field.value) is None for field in NativeMoleculeField
    )


def test_selected_fields_are_exact_contiguous_arrays():
    mol = read_mol("c1ccccc1P(=O)(O)O")
    fields = frozenset(
        {
            NativeMoleculeField.ATOMIC_NUMBERS,
            NativeMoleculeField.COORDINATES,
            NativeMoleculeField.BOND_INDICES,
            NativeMoleculeField.BOND_KINDS,
        }
    )

    buffers = pack_molecule(mol, fields)

    assert buffers.atomic_numbers.dtype == np.int32
    assert buffers.coordinates.dtype == np.float64
    assert buffers.coordinates.shape == (buffers.atom_count, 3)
    assert buffers.bond_indices.dtype == np.int32
    assert buffers.bond_indices.shape == (buffers.bond_count, 2)
    assert buffers.bond_kinds.dtype == np.uint8
    assert all(
        array.flags.c_contiguous
        for array in (
            buffers.atomic_numbers,
            buffers.coordinates,
            buffers.bond_indices,
            buffers.bond_kinds,
        )
    )
    assert buffers.formal_charges is None
    assert buffers.partial_charges is None


def test_empty_present_bond_field_differs_from_absence():
    mol = read_mol("[Eu+3]")

    present = pack_molecule(
        mol,
        frozenset({NativeMoleculeField.BOND_INDICES}),
    )
    absent = pack_molecule(mol, frozenset())

    assert present.bond_indices.shape == (0, 2)
    assert absent.bond_indices is None


def test_common_bond_codes_retain_dative_semantics():
    assert NativeBondKindCode.ZERO == 5
    assert NativeBondKindCode.DATIVE == 6
    assert NativeBondKindCode.UNKNOWN == 7


@pytest.mark.parametrize(
    ("bond_kind", "bond_direction", "expected_endpoints"),
    (
        (BondKind.SINGLE, None, [[0, 1]]),
        (BondKind.DATIVE, "atom1_to_atom2", [[0, 1]]),
        (BondKind.DATIVE, "atom2_to_atom1", [[1, 0]]),
    ),
)
def test_common_bond_indices_encode_dative_donor_to_acceptor_direction(
    bond_kind,
    bond_direction,
    expected_endpoints,
):
    mol = Molecule()
    mol.create_atom(atomic_number=7)
    mol.create_atom(atomic_number=29)
    mol.add_bond(
        0,
        1,
        bond_order=1.0,
        bond_kind=bond_kind,
        bond_direction=bond_direction,
    )

    buffers = pack_molecule(
        mol,
        frozenset(
            {
                NativeMoleculeField.BOND_INDICES,
                NativeMoleculeField.BOND_KINDS,
            }
        ),
    )

    assert buffers.bond_indices.tolist() == expected_endpoints
    assert buffers.bond_kinds.tolist() == [
        NativeBondKindCode.DATIVE
        if bond_kind is BondKind.DATIVE
        else NativeBondKindCode.SINGLE
    ]


@pytest.mark.parametrize("values", [None, (None, 0.0, -0.25)])
def test_general_charge_request_returns_none_for_absent_or_incomplete_selection(values):
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    mol = read_mol("CCO")
    if values is not None:
        _import_partial_charges(mol, values, producer=ATOM_SOURCE_CODES["explicit"])

    buffers = pack_molecule(mol, frozenset({NativeMoleculeField.PARTIAL_CHARGES}))

    assert buffers.partial_charges is None
    assert tuple(atom.partial_charge for atom in mol.atoms) == (
        (None,) * 3 if values is None else values
    )


def test_general_charge_request_preserves_order_zero_and_owned_contiguous_values():
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    mol = read_mol("CCO")
    _import_partial_charges(mol, (-0.25, 0.0, 0.25), producer=ATOM_SOURCE_CODES["explicit"])
    buffers = pack_molecule(mol, frozenset({NativeMoleculeField.PARTIAL_CHARGES}))

    assert buffers.partial_charges.dtype == np.float64
    assert buffers.partial_charges.flags.c_contiguous
    assert buffers.partial_charges.tolist() == [-0.25, 0.0, 0.25]
    buffers.partial_charges[0] = 9.0
    assert mol.atoms[0].partial_charge == -0.25


def test_stale_charge_request_raises_but_other_field_requests_do_not_read_charge(monkeypatch):
    from hotpot.cheminfo.structure import _results
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES

    mol = read_mol("CCO")
    _results._import_partial_charges(mol, (-0.25, 0.0, 0.25), producer=ATOM_SOURCE_CODES["explicit"])
    mol.atoms[0].x = 1.0

    with pytest.raises(RuntimeError, match="geometry"):
        pack_molecule(mol, frozenset({NativeMoleculeField.PARTIAL_CHARGES}))

    def forbidden(*args):
        pytest.fail("unrequested charge selection was consulted")

    monkeypatch.setattr(_results, "_current_partial_charge_result", forbidden)
    buffers = pack_molecule(mol, frozenset({NativeMoleculeField.ATOMIC_NUMBERS}))
    assert buffers.atomic_numbers.tolist() == [6, 6, 8]
    assert buffers.partial_charges is None
