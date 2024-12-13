"""
python v3.9.0
@Project: hotpot
@File   : io
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 10:08
"""
import os
from os import PathLike
from pathlib import Path
import io
from typing import Literal, Generator

from openbabel import openbabel as ob, pybel as pb

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.obconvert import obmol2mol, mol2obmol

class MolReader:
    def __init__(self, src, fmt=None, **kwargs):
        """"""
        self.src = src
        self.src_type = self._src_checks(src)
        self.fmt = self._determine_fmt(src, fmt)
        self.kwargs = kwargs

        self._generator = self.get_generator()
        self._items = None

    @staticmethod
    def _src_checks(src) -> Literal['path', 'str', 'bytes', 'StringIO', 'BytesIO', 'FileIO']:
        if isinstance(src, str):
            if os.path.exists(src):
                return 'path'
            else:
                return 'str'
        elif isinstance(src, PathLike):
            return 'path'
        elif isinstance(src, bytes):
            return 'bytes'
        elif isinstance(src, io.StringIO):
            return 'StringIO'
        elif isinstance(src, io.BytesIO):
            return 'BytesIO'
        elif isinstance(src, io.FileIO):
            return 'FileIO'
        else:
            raise TypeError(f'get unsupported input type {type(src)}')

    def _determine_fmt(self, src, fmt):
        if not fmt:
            if isinstance(src, (str, PathLike)):
                p_src = Path(src)
                self.src_type = 'path'
                if p_src.is_file():
                    fmt = p_src.suffix[1:]
                else:
                    raise FileNotFoundError(f'file {p_src} not exist!')

        if not fmt:
            raise ValueError('the fmt has not been known!')

        return fmt

    def read_openbabel(self):
        if self.src_type == 'IOString':
            src = self.src.read()
        else:
            src = self.src

        if self.src_type == 'path':
            reader = pb.readfile(self.fmt, src, self.kwargs)
        else:
            reader = pb.readstring(self.fmt, src, self.kwargs)

        def make_generator():
            nonlocal reader
            if isinstance(reader, pb.Molecule):
                reader = [reader]

            for pmol in reader:
                yield pmol

        return make_generator()

    def get_generator(self):
        for obmol in self.read_openbabel():
            yield obmol2mol(obmol.OBMol, Molecule())

    def refresh(self):
        self._generator = self.get_generator()

    def __next__(self):
        return next(self._generator)

    def __getitem__(self, item: int):
        if not self._items:
            self._items = list(self.get_generator())

        return self._items[item]

    def __iter__(self):
        return self.get_generator()


if __name__ == "__main__":
    from tqdm import tqdm
    reader = MolReader('../../cpp/data/Compound_127500001_128000000.sdf')
    for mol in tqdm(reader):
        print(mol.weight)
        break
        # pass

    for cycle in mol.simple_cycles:
        print(cycle)

    mol.build3d()
    mol.write('mol2', '/mnt/d/mol2.mol2', overwrite=True)

    # obMol, i2r = mol2obmol(mol)
    # # obMol.PerceiveBondOrders()
    # # obMol.MakeDativeBonds()
    # pmol = pb.Molecule(obMol)
    # pmol.localopt()
    # # pmol.removeh()
    # pmol.write('mol2', '/mnt/d/mol2.mol2', overwrite=True)
    # print(pmol.write('can'))
    # print(mol.smiles)
