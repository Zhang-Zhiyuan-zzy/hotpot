import os.path as osp
from typing import Callable
from glob import glob
from tqdm import tqdm

from ccdc import io

def get_stem(filename):
    return osp.splitext(osp.basename(filename))[0]

def get_metals(mol):
    return [a for a in mol.atoms if a.is_metal]

class ComplexStatistics(object):
    def __init__(self, ccdc_complex_dir, *skips):
        self.ccdc_complex_dir = ccdc_complex_dir
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

        for file in tqdm(glob(osp.join(self.ccdc_complex_dir, '*.mol2'))):
            identifier = get_stem(file)

            try:
                reader = io.MoleculeReader(file)
                mol = reader[0]
            except StopIteration:
                print(file)
                continue

            for name, stat in statistician.items():
                stat(mol, identifier)

    def _stat_atom_counts(self, mol, identifier):
        list_identifier = self.atom_counts.setdefault(len(mol.atoms), [])
        list_identifier.append(identifier)

    def _stat_metal_counts(self, mol, identifier):
        list_identifier = self.metal_counts.setdefault(len(get_metals(mol)), [])
        list_identifier.append(identifier)

    def _stat_mol_weights(self, mol, identifier):
        self.mol_weights.append(mol.molecular_weight)

    def _stat_metal_types(self, mol, identifier):
        metal_symbols = {m.atomic_symbol for m in get_metals(mol)}
        for symbol in metal_symbols:
            list_identifier = self.metal_types.setdefault(symbol, [])
            list_identifier.append(identifier)
