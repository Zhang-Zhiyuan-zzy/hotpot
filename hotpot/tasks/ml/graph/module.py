"""
python v3.9.0
@Project: hotpot
@File   : graph
@Auther : Zhiyuan Zhang
@Data   : 2023/8/8
@Time   : 13:28
"""
from pathlib import Path
import json

import torch
from torch import nn
import torch_geometric as pg
from openbabel import openbabel as ob

import hotpot as hp
from hotpot import data_root
from hotpot.cheminfo import Atom


def get_atom_energy_tensor(
        method: str, basis: str,
        solvent: str = None,
        charges: list[int] = None,
        end_element: int = 58,
        padding_miss: bool = False
):
    """"""
    path_atom_single_point = Path(data_root).joinpath('atom_single_point.json')
    _atom_single_point = json.load(open(path_atom_single_point))

    if solvent is None:
        solvent = "null"

    if isinstance(charges, list):
        charges = [0] + charges

    energies = [0.0]
    for i in range(1, end_element):
        atom = Atom(atomic_number=i)

        if charges:
            c = charges[i]
        elif atom.is_metal:
            c = atom.stable_charge
        else:
            c = 0

        try:
            energy = _atom_single_point[atom.symbol][method][basis][solvent][str(c)]
        except KeyError as e:
            if padding_miss:
                energy = 0.0
            else:
                print(KeyError(atom.symbol))
                raise e

        energies.append(energy)

    return torch.tensor(energies)


class MolNet(nn.Module):
    """"""
    def __init__(self, element_energy: torch.Tensor):
        """

        Args:
            element_energy: elemental energies
        """

    def forward(self, x, e):
        """"""


class BDENet(nn.Module):
    """"""
    def __init__(self):
        """"""

    def forward(self, x, e):
        """"""
