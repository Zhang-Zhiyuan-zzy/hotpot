"""
python v3.7.9
@Project: hotpot
@File   : dundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/22
@Time   : 3:18
"""
from os import PathLike
from typing import *
from pathlib import Path
from tqdm import tqdm
import numpy as np
from cheminfo import Molecule

# Typing Annotations
GraphFormatName = Literal['Pytorch', 'numpy']


class MolBundle:
    """"""
    def __init__(self, mols: Union[list[Molecule], Generator[Molecule, None, None]] = None):
        self.mols = mols
        self.mols_generator = True if isinstance(mols, Generator) else False

    @classmethod
    def read_from_dir(
            cls, fmt: str,
            read_dir: Union[str, PathLike],
            match_pattern: str = '*',
            generate: bool = False
    ):
        def mol_generator():
            nonlocal read_dir

            if isinstance(read_dir, str):
                read_dir = Path(read_dir)
            elif not isinstance(read_dir, PathLike):
                raise TypeError(f'the read_dir should be a str or PathLike, instead of {type(read_dir)}')

            for path_mol in read_dir.glob(match_pattern):
                mol = Molecule.readfile(path_mol, fmt)
                yield mol

        if generate:
            return cls(mol_generator())
        else:
            return cls([m for m in tqdm(mol_generator(), 'reading molecules')])

    def graph_represent(self, graph_fmt: GraphFormatName):
        """ Transform mols to the molecule to graph representation,
        the transformed graph with 'numpy.ndarray' or 'PyTorch.Tensor' format """

