"""Read-only carrier projections and factual record packing for inference."""
from __future__ import annotations
from typing import Optional, Iterable, Any
from itertools import product
from operator import attrgetter
import numpy as np
from hotpot.cheminfo.structure._builder import _make_molecule_record
from hotpot.cheminfo.structure.stereo import AtomTransportRecord, StereoPayload
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._results import _partial_charge_projection


def mca_molecule_record(atoms, bonds, formal_charge):
    values = np.zeros((len(atoms), 19), dtype=np.float64)
    sources = np.zeros(values.shape, dtype=np.uint8)
    supplied = (0, 7, 9, 10, 11, 12, 14)
    for row, atom in enumerate(atoms):
        values[row, supplied] = (atom.atomic_number,
                       atom.formal_charge, atom.is_aromatic,
                       *atom.coordinates, atom.implicit_hydrogens)
    sources[:, supplied] = ATOM_SOURCE_CODES["mca_dto"]
    presence = np.full(len(values), sum(FIELD_PRESENCE_BITS[index] for index in supplied), dtype=np.uint32)
    for row, atom in enumerate(atoms):
        if atom.partial_charge is not None:
            values[row, 8] = atom.partial_charge
            sources[row, 8] = ATOM_SOURCE_CODES["mca_dto"]
            presence[row] |= FIELD_PRESENCE_BITS[8]
    raw_bonds = np.zeros((len(bonds), 3), dtype=np.float64)
    raw_bonds[:, 0] = [bond.bond_order for bond in bonds]
    record = _make_molecule_record(
        atom_values=values, atom_presence=presence, atom_sources=sources,
        atom_frac_valences=(),
        bond_endpoints=tuple((bond.begin_atom_index, bond.end_atom_index) for bond in bonds),
        bond_values=raw_bonds,
        bond_kinds=tuple(bond.bond_kind for bond in bonds),
        bond_directions=tuple(bond.bond_direction for bond in bonds),
        bond_sources=(None,) * len(bonds), bond_metadata=(None,) * len(bonds),
        charge=(True, formal_charge, "mca_dto"), spin_mult=(False, None, None),
        cell=None, stereo_payload=StereoPayload(1, (), tuple(
            AtomTransportRecord(row, atom.isotope, None, None, None, None)
            for row, atom in enumerate(atoms)), ()),
        source_frames=(), conformer_results=(),
        frame_selection=(0, None), angle_constraints=(), torsion_constraints=(), identifier=(False, None),
    )
    return record


def direct_edge_to_indirect(attr_or_index: np.ndarray) -> np.ndarray:
    return np.concatenate([attr_or_index, np.flip(attr_or_index, axis=0)], axis=1)

_default_catom_elements = {'O', 'N', 'S', 'P', 'Si', 'B'}


def extract_potentials_cbonds(
        mol,
        catoms_symbols: Optional[Iterable[str]] = None,
        data: dict[str, np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """ Just a numpy version of hotpot.plugins.PyG.data.utils.extract_potentials_cbonds """
    if not catoms_symbols:
        catoms_symbols = _default_catom_elements
    else:
        catoms_symbols = set(catoms_symbols)

    cbond_index = []
    is_cbond = []
    all_potentials = [a for a in mol.atoms if a.symbol in catoms_symbols]
    for metal, pca in product(mol.metals, all_potentials):
        cbond_index.append([metal.idx, pca.idx])
        try:
            _ = mol.bond(metal.idx, pca.idx)
            is_cbond.append(1)
        except KeyError:
            is_cbond.append(0)

    assert len(is_cbond) == len(cbond_index)
    is_cbond = np.array(is_cbond, dtype=np.int32)
    cbond_index = np.array(cbond_index, dtype=np.int64).T

    if isinstance(data, dict):
        data.update({'cbond_index': cbond_index, 'is_cbond': is_cbond})
    else:
        data = {'cbond_index': cbond_index, 'is_cbond': is_cbond}

    return data


def extract_atom_attrs(mol, data: dict[str, Any] = None, atomic_number_only: bool = False, *, projected=None, fields=()) -> dict:
    if atomic_number_only:
        x_names = ['atomic_number']
        x = mol._atom_table.column('atomic_number').astype(np.int32)
    else:
        x_names = fields + ('is_metal',)
        x = np.empty((len(mol._atoms), 16), dtype=np.float64)
        x[:, :15] = projected
        # Task encoding preserves missing/stale charge as zero without installing facts.
        x[:, fields.index('partial_charge')] = tuple(
            0.0 if value is None else value
            for value in _partial_charge_projection(mol)
        )
        x[:, 15] = [atom.is_metal for atom in mol._atoms]
    if isinstance(data, dict):
        data.update({'x': x, 'x_names': x_names})
    else:
        data = {'x': x, 'x_names': x_names}
    return data


def extract_bond_attrs(mol, edge_attr_names: Iterable[str] = None, data: dict[str, Any] = None) -> dict:
    if (link_matrix := mol.link_matrix).ndim == 2:
        edge_index = direct_edge_to_indirect(link_matrix.T)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    if edge_attr_names:
        bond_attr_getter = attrgetter(*edge_attr_names)
        edge_attr = direct_edge_to_indirect(np.array([(bond_attr_getter(b)) for b in mol.bonds], dtype=np.float64))
        edge_attr_dict = {'edge_attr': edge_attr, 'edge_attr_names': edge_attr_names}
    else:
        edge_attr_dict = {}

    if isinstance(data, dict):
        data.update({'edge_index': edge_index})
    else:
        data = {'edge_index': edge_index}
    data.update(edge_attr_dict)

    return data


def extract_atom_pairs(atom_pairs, pair_attr_names, data: dict[str, Any] = None) -> dict:
    if (idx_matrix := atom_pairs.idx_matrix).ndim == 2:
        pair_index = idx_matrix
    else:
        pair_index = np.empty((2, 0), dtype=np.int64)

    pair_attr = np.array([p.attrs for k, p in atom_pairs.items()], dtype=np.float64).reshape(-1, len(pair_attr_names))

    if isinstance(data, dict):
        data.update({'pair_index': pair_index, 'pair_attr_names': pair_attr_names, 'pair_attr': pair_attr})
    else:
        data = {'pair_index': pair_index, 'pair_attr_names': pair_attr_names, 'pair_attr': pair_attr}

    return data


def extract_ring_attrs(rings, rings_attr_names: Iterable[str] = None, data: dict[str, Any] = None) -> dict:
    if rings:
        rings_node_index = [r.atoms_indices for r in rings]
        rings_node_nums = [len(rni) for rni in rings_node_index]

        rings_node_index = np.array(sum(rings_node_index, start=[]), dtype=np.int64)
        mol_rings_nums = np.array([len(rings_node_nums)], dtype=np.int64)
        rings_node_nums = np.array(rings_node_nums, dtype=np.int32)
        mol_rings_node_nums = np.array([rings_node_nums.sum()], dtype=np.int32)

    else:
        mol_rings_nums = np.array([0], dtype=np.int64)
        rings_node_index = np.array([], dtype=np.int64)
        rings_node_nums = np.array([], dtype=np.int32)
        mol_rings_node_nums = np.array([], dtype=np.int32)

    rings_struct_dict = {
        'mol_rings_nums': mol_rings_nums,
        'rings_node_index': rings_node_index,
        'rings_node_nums': rings_node_nums,
        'mol_rings_node_nums': mol_rings_node_nums,
    }

    if rings_attr_names:
        rings_attr_names = list(rings_attr_names)
        ring_attr_getter = attrgetter(*rings_attr_names)
        rings_attr = np.array(
            [ring_attr_getter(r) for r in rings], dtype=np.float64
        ).reshape(len(rings), len(rings_attr_names))
        rings_attr_dict = {'rings_attr': rings_attr, 'rings_attr_names': rings_attr_names}
    else:
        rings_attr_dict = {}


    if isinstance(data, dict):
        data.update(rings_struct_dict)
    else:
        data = rings_struct_dict
    data.update(rings_attr_dict)

    return data
