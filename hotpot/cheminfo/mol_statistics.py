import os.path as osp
from typing import Callable

from glob import glob
from tqdm import tqdm
import hotpot as hp

def get_stem(filename):
    return osp.splitext(osp.basename(filename))[0]

class ComplexStatistics(object):
    def __init__(self, complex_dir, *skips):
        self.complex_dir = complex_dir
        self.skips = skips
        self.atom_counts = {}
        self.metal_counts = {}
        self.metal_types = {}
        self.mol_weights = []
        self.solvents = {}

    def statistic(self):
        statistician = {
            n: getattr(self, n) for n in self.__dir__()
            if n.startswith('_stat_') and isinstance(getattr(self, n), Callable) and
               '_'.join(n[1:].split('_')[1:]) not in self.skips
        }

        for file in tqdm(glob(osp.join(self.complex_dir, '*.mol2'))):
            identifier = get_stem(file)

            try:
                mol = next(hp.MolReader(file))
            except StopIteration:
                print(file)
                continue

            for name, stat in statistician.items():
                stat(mol, identifier)

    def _stat_atom_counts(self, mol, identifier):
        list_identifier = self.atom_counts.setdefault(len(mol.atoms), [])
        list_identifier.append(identifier)

    def _stat_metal_counts(self, mol, identifier):
        list_identifier = self.metal_counts.setdefault(len(mol.metals), [])
        list_identifier.append(identifier)

    def _stat_mol_weights(self, mol, identifier):
        self.mol_weights.append(mol.weight)

    def _stat_metal_types(self, mol, identifier):
        metal_symbols = {m.symbol for m in mol.metals}
        for symbol in metal_symbols:
            list_identifier = self.metal_types.setdefault(symbol, [])
            list_identifier.append(identifier)

