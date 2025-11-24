"""
@File Name:        pairs
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/16 21:44
@Project:          Hotpot
"""
import os.path as osp
from typing import Iterable, Any, Union, Optional
from itertools import chain, combinations

import pebble
from pebble import ProcessExpired

import torch

import hotpot as hp
from .utils import convert_hp_mol_to_pyg_data

def _full_combinations(
        x: Iterable[Any],
        get_list: bool = True,
        include_empty: bool = False,
        include_self: bool = True,
        _start: Optional[int] = None,
        _end: Optional[int] = None,
) -> Union[list[Any], chain]:
    start = 0 if include_empty else 1
    end = len(x) +1 if include_self else len(x)

    # User specified start and end
    if isinstance(_start, int):
        start = _start
    if isinstance(_end, int):
        end = _end

    if get_list:
        return list(chain.from_iterable(combinations(x, r) for r in range(start, end)))
    return chain.from_iterable(combinations(x, r) for r in range(start, end))


default_catoms_elements = {'O', 'N', 'S', 'P', 'Si', 'B'}
def convert_ml_pairs_to_cbond_broken_data(path_struct: str, data_dir: str, catoms_elements=None) -> Optional[set[str]]:
    struct_name = osp.splitext(osp.basename(path_struct))[0]

    if not catoms_elements:
        catoms_elements = default_catoms_elements

    mol = next(hp.MolReader(path_struct))
    metal = mol.metals[0]
    metal_idx = metal.idx

    metal_neighbours = [a for a in metal.neighbours]

    # Exclude pairs with extra metal-neigh atoms outside the catoms_elements
    metal_neigh_symbol = {a.symbol for a in metal.neighbours}
    if extra_natom := metal_neigh_symbol.difference(catoms_elements):
        # print(f"Exclude {struct_name} with extra catom {extra_natom} outside of {catoms_elements}")
        return extra_natom

    list_catoms_index = [a.idx for a in metal_neighbours]
    full_catom_options = {a.idx for a in mol.atoms if a.symbol in catoms_elements}

    # This block to choose combinations of coordination bonds from the raw M-L pairs.
    # The chosen bonds in a combination will be retained in the M-L pairs and the others
    # will be broken.
    full_catoms_combinations = _full_combinations(list_catoms_index, _start=0, _end=1)  # Full combination of catom indices
    smiles_set = set()  # Recoding smiles to exclude redundant pairs with same 2d graph as the former
    for i, chosen_catoms in enumerate(full_catoms_combinations):
        clone = mol.copy()

        # Specify which cbond to be broken
        broken_cbond_catoms = set(list_catoms_index) - set(chosen_catoms)
        broken_cbond = [clone.bond(metal.idx, a_idx) for a_idx in broken_cbond_catoms]
        clone.remove_bonds(broken_cbond)  # Breaking the unchosen cbond

        # Check the smiles
        if clone.smiles in smiles_set:
            continue
        else:
            smiles_set.add(clone.smiles)

        # Specify the cbond options in the processed M-L pair and if they are true cbonds
        catom_options = full_catom_options - set(chosen_catoms)  # Exclude retrained cbond from possible cbond options
        cbond_options = [[metal_idx, ca_idx] for ca_idx in catom_options]
        # If above `cbond_options` are true of cbond
        is_true_cbond = [float(cb[1] in list_catoms_index) for cb in cbond_options]

        # Assign the pair identifier
        clone.identifier = f'{struct_name}_C{len(list_catoms_index)}_{len(chosen_catoms)}_O{i}'

        data = convert_hp_mol_to_pyg_data(
            clone,
            identifier=clone.identifier,
            cbond_index=torch.tensor(cbond_options, dtype=torch.long).mT if cbond_options else torch.tensor(cbond_options, dtype=torch.long),
            is_cbond=torch.tensor(is_true_cbond, dtype=torch.int)
        )
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))
