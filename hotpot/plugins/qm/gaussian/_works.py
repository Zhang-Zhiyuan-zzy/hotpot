"""
python v3.9.0
@Project: hotpot
@File   : _works
@Auther : Zhiyuan Zhang
@Data   : 2023/11/29
@Time   : 13:30

Notes:
    defining some convenient workflow to run Gaussian
"""
from os import PathLike
from typing import Union

from hotpot.cheminfo import Molecule
from hotpot.plugins.qm.gaussian import parse_gjf, reorganize_gjf


def ladder_opti(mol: Molecule, ladder: list[str], *args, **kwargs):
    """"""


def update_gjf_coordinates(old_gjf_file: Union[str, PathLike], log_file: Union[str, PathLike]):
    mol = Molecule.read_from(log_file, 'g16log')
    return update_gjf(
        old_gjf_file, {'coordinates': [
            f'{atom.symbol:18}{"   ".join(map(lambda x: f"{x:f}", atom.coordinate))}' for atom in mol.atoms
        ]}
    )


def update_gjf(old_gjf_file: Union[str, PathLike], update_dict: dict):
    data = parse_gjf(old_gjf_file)
    data.update(update_dict)
    return data


if __name__ == '__main__':
    new_gjf = reorganize_gjf(update_gjf_coordinates(
        '/mnt/c/Users/zhang/OneDrive/Papers/Gibbs with logK/results/g16/gjf/pairs/81_81_C20H28N2O6P2Am.gjf',
        '/mnt/c/Users/zhang/OneDrive/Papers/Gibbs with logK/results/g16/log/pairs/81_81_C20H28N2O6P2Am.log'
    ))
