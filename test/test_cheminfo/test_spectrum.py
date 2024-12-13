"""
python v3.9.0
@Project: hotpot
@File   : spectrum
@Auther : Zhiyuan Zhang
@Data   : 2024/8/23
@Time   : 16:02
"""
import unittest as ut
from itertools import combinations

import numpy as np

import hotpot as hp
from hotpot.dataset import load_dataset


class TestSpectrum(ut.TestCase):

    def test_similarity(self):
        mol1 = next(hp.Molecule.read('c1ccc(cc1)C(N1CCCC1)CNc1ccc2n(n1)cnn2', 'smi'))
        mol2 = next(hp.Molecule.read('Fc1ccc(cc1)C1(C)NC(=O)N(C1=O)CCn1c(C)csc1=O', 'smi'))
        mol3 = next(hp.Molecule.read('c1ccc(cc1)C(N1CCCC1)CNc1ccc2n(n1)cnn2', 'smi'))

        print(mol1.spectral_similarity(mol2))
        print(mol1.spectral_similarity(mol3))

    def test_permutation_invariance(self):
        smiles_loader = load_dataset('SMILES')

        for smi in smiles_loader:
            mol = next(hp.Molecule.read(smi, 'smi'))
            perm_mol = next(hp.Molecule.read(mol.smiles, 'smi'))

            adj1 = mol.adjacency
            adj2 = perm_mol.adjacency

            if np.any(adj1 != adj2):
                print(mol.spectral_similarity(perm_mol))

    def test_any_two_similarity(self):
        smiles_loader = load_dataset('SMILES')

        mols = []
        for i, smi in enumerate(smiles_loader):
            if i > 10000:
                break

            mols.append(next(hp.Molecule.read(smi, 'smi')))

        for mol1, mol2 in combinations(mols, 2):
            s = mol1.spectral_similarity(mol2)
            print(s)

