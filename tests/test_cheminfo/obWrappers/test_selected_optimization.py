"""Selected force-field inputs retain the intact source's aromatic facts."""

from math import isfinite

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.obWrappers.packing import _pack_molecule
from hotpot.cheminfo.operations.forcefields import backend
from hotpot.cheminfo.operations.topology_queries import _project_molecule
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._selection import _capture_topology_selection


def _fixed_aromatic_ring():
    mol = read_mol("c1ccccc1")
    mol.add_hydrogens()
    coordinates = np.zeros((len(mol.atoms), 3))
    for row, atom in enumerate(mol.atoms):
        center = row if atom.atomic_number == 6 else atom.neighbours[0].idx
        angle = center * np.pi / 3.
        radius = 1.4 if atom.atomic_number == 6 else 2.48
        coordinates[row, :2] = radius * np.array([np.cos(angle), np.sin(angle)])
    coordinates[0, 2] = 0.04
    mol.coordinates = coordinates
    return mol


def test_real_selected_aromatic_uff_preserves_source_facts_and_publishes_only_coordinates(monkeypatch):
    mol = _fixed_aromatic_ring()
    atoms, bonds = tuple(mol.atoms), tuple(mol.bonds)
    ring_edge = next(bond for bond in bonds if all(atom.atomic_number == 6 for atom in bond.atoms))
    selection = _capture_topology_selection(mol, excluded_bonds=(ring_edge,))
    full = _pack_molecule(mol)
    source = _capture_source(mol, ('topology', 'representation', 'cell'))
    initial_coordinates = mol.coordinates.copy()

    opened = _project_molecule(mol, type(mol), selection=selection)
    with pytest.raises(RuntimeError, match=r'AROMATIC\.NONCYCLIC_SOURCE'):
        _pack_molecule(opened)
    _check_source(mol, source)

    invoke_native = backend._single_optimize_data
    reports = []

    def observe(data, forcefield, steps):
        assert data.atom_count == len(atoms)
        assert data.bond_count == len(bonds) - 1
        assert forcefield == 'UFF' and steps == 3
        report = invoke_native(data, forcefield, steps)
        reports.append(report)
        np.testing.assert_array_equal(mol.coordinates, initial_coordinates)
        _check_source(mol, source)
        return report

    monkeypatch.setattr(backend, '_single_optimize_data', observe)
    result = backend._single_ob_optimization(mol, 'UFF', 3, selection=selection)

    assert len(reports) == 1
    report = reports[0]
    assert isfinite(report.energy) and report.energy_unit == 'kJ/mol'
    assert not report.exploded
    assert result.energy == report.energy and result.energy_unit == report.energy_unit
    assert result.exploded == report.exploded
    assert report.coordinates.shape == initial_coordinates.shape
    assert np.isfinite(report.coordinates).all()
    np.testing.assert_array_equal(mol.coordinates, report.coordinates)
    assert not np.array_equal(mol.coordinates, initial_coordinates)
    assert all(current is original for current, original in zip(mol.atoms, atoms))
    assert all(current is original for current, original in zip(mol.bonds, bonds))
    assert len(mol.atoms) == len(atoms) and len(mol.bonds) == len(bonds)
    _check_source(mol, source)
    final = _pack_molecule(mol)
    np.testing.assert_array_equal(final.atom_aromatic, full.atom_aromatic)
    for name in ('bond_indices', 'bond_orders', 'bond_kinds', 'bond_aromatic'):
        np.testing.assert_array_equal(getattr(final, name), getattr(full, name))
