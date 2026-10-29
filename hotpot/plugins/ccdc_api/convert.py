# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : convert
 Created   : 2025/5/17 22:19
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import numpy as np
from ccdc.molecule import Molecule as CMol, Atom as CAtom, Bond as CBond
from hotpot import Molecule as Mol
from hotpot.cheminfo.elements import elements
from hotpot.cheminfo.structure._builder import _make_molecule_record
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_PRESENCE_BITS

ccdc_all_bond_type = CBond.BondType.all_bond_types()

def _c_atom_is_aromatic(c_atom: CAtom) -> bool:
    return any(ring.is_aromatic for ring in c_atom.rings)

def _c_bond_is_aromatic(c_bond: CBond) -> bool:
    return any(ring.is_aromatic for ring in c_bond.rings)

def mol_from_ccdc_to_hp(c_mol: CMol):
    c_mol.normalise_labels()
    label_to_idx = {a.label: i for i, a in enumerate(c_mol.atoms)}

    atoms = tuple(c_mol.atoms)
    values = np.zeros((len(atoms), 19), dtype=np.float64)
    supplied = (0, 7, 8, 9, 10, 11, 12)
    for row, atom in enumerate(atoms):
        values[row] = (
            atom.atomic_number, *elements.electron_configs[atom.atomic_number],
            atom.formal_charge, atom.partial_charge, float(_c_atom_is_aromatic(atom)),
            *atom.coordinates, 0, 0, 0, 0, 0, 0,
        )
    sources = np.zeros(values.shape, dtype=np.uint8)
    sources[:, supplied] = ATOM_SOURCE_CODES["ccdc"]
    sources[:, 1:7] = ATOM_SOURCE_CODES["electron_configuration"]
    presence = np.full(len(values), sum(FIELD_PRESENCE_BITS[index] for index in range(13)), dtype=np.uint32)
    bonds = tuple(c_mol.bonds)
    orders = [ccdc_all_bond_type[str(bond.bond_type)] for bond in bonds]
    raw_bonds = np.zeros((len(bonds), 3), dtype=np.float64)
    raw_bonds[:, 0] = orders
    record = _make_molecule_record(
        atom_values=values, atom_presence=presence, atom_sources=sources,
        atom_isotopes=(), atom_frac_valences=(),
        bond_endpoints=tuple(tuple(label_to_idx[atom.label] for atom in bond.atoms) for bond in bonds),
        bond_values=raw_bonds, bond_partitions=(0,) * len(bonds),
        bond_kinds=tuple({1: "single", 1.5: "aromatic", 2: "double", 3: "triple"}.get(order, "unknown") for order in orders),
        bond_directions=(None,) * len(bonds), bond_sources=(None,) * len(bonds),
        bond_metadata=(None,) * len(bonds), charge=(False, None, None), spin_mult=(False, None, None),
        cell=None, stereo_payload=None, legacy_frame_values=(None,) * 9, legacy_frame_index=0,
        angle_constraints=(), torsion_constraints=(), identifier=(False, None),
    )
    return Mol._build_molecule(record)
