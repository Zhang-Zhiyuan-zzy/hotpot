"""
python v3.9.0
@Project: hp5
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/6/5
@Time   : 19:15
"""
from pathlib import Path
import unittest as ut
from copy import copy

import hotpot as hp
from hotpot.cheminfo.core import InternalCoordinates
from hotpot.utils.tools import show_time

import numpy as np

import networkx as nx
import time

import test


outdir = Path(test.output_dir)


class TestChemInfo(ut.TestCase):

    def test_molecule(self):

        # Read molecule with two component
        reader = hp.MolReader(
            'CCCC(CCCCCCC)CCCCN(CCCC(CC(CCCC(CCCC(CCCC(CC)(CC)(CC)CCC(CC)(CC)(CC)CC)CC(CCCCCC)CC(CC(CCCC(C)'
            '(C)(C)CCCC)C(C(C)(C)(C))CCC(CCC(C(C)(CC)(C))CCCCCCC)CCCCCC)CCCC(CCCCCCCCC)CCCC)CCC(CC)CCC)CCC)CCCCCCCC'
            'C)C(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CCC(CCC(CC)CCCCCCC)CCCCC)CCCCCCCC)C=C1'
            '.'
            'CCN(CC)P(OCCC)(OCCC)(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1',
            'smi'
        )

        mol: hp.Molecule = next(reader)

        self.assertEqual(2, len(mol.components))
        self.assertEqual(223, len(mol.atoms))
        self.assertEqual(227, len(mol.bonds))
        self.assertEqual((227, 2), mol.link_matrix.shape)

        a = mol.atoms[20]
        self.assertIsInstance(a.neighbours[0], hp.Atom)
        self.assertIsInstance(a.bonds[0], hp.Bond)

        self.assertEqual(len(hp.Atom._attrs_enumerator), a.attrs.shape[0])

        t1 = time.time()
        print(mol.atom_attr_matrix)
        print('Calculate implicit hydrogens')
        hs1 = tuple(a.implicit_hydrogens for a in mol.atoms)
        for a in mol.atoms:
            a.implicit_hydrogens = 0
        hs2 = tuple(a.implicit_hydrogens for a in mol.atoms)
        for a in mol.atoms:
            a.calc_implicit_hydrogens()
        hs3 = tuple(a.implicit_hydrogens for a in mol.atoms)
        print(hs1)
        print(hs2)
        print(hs3)
        self.assertEqual(hs1, hs3)
        t2 = time.time()
        print(t2-t1)

        print(mol.smiles)
        print(mol.coordinates)
        print(len(mol.atoms))
        print(len(mol.bonds))
        mol.add_hydrogens()
        print(len(mol.atoms))
        print(len(mol.bonds))

        t1 = time.time()
        clone = copy(mol)
        t2 = time.time()
        print(t2-t1)

        self.assertIsNot(clone, mol)
        self.assertTrue(np.all(clone.link_matrix == mol.link_matrix))
        self.assertTrue(np.all(clone.atom_attr_matrix == mol.atom_attr_matrix))
        self.assertTrue(all(a1.label == a2.label for a1, a2 in zip(mol.atoms, clone.atoms)))
        self.assertFalse(any(a1 is a2 for a1, a2 in zip(mol.atoms, clone.atoms)))

        self.assertRaises(PermissionError, copy, clone.atoms[20])
        self.assertIs(clone.sssr[0].mol, clone)

        c1, c2 = clone.components
        assert isinstance(c1, hp.Molecule)
        assert isinstance(c2, hp.Molecule)

        c1.build3d()
        print(c1.coordinates)
        c1.coordinates += 1
        print(c1.coordinates)
        t2 = time.time()
        print(t2-t1)

        print('Add hydrogens')
        t1 = time.time()
        print(c1.graph)
        print(c1.graph)
        print(c1.link_matrix)
        c1.add_hydrogens()
        print(c1.link_matrix)
        t2 = time.time()
        print(t2-t1)

        c2 = copy(c2)
        c1.atoms[0].link_with(c2.atoms[1])

        c1.build3d()

        print(len(c1.angles))
        print(len(c1.torsions))
        print(c1.torsion_degrees(*c1.torsions[30]))

    def test_internal_coordinates(self):
        reader = hp.MolReader('CCN(CC)P(OCCC)(OCCC)(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1', 'smi')
        mol = next(reader)
        mol.build3d()
        zmat = InternalCoordinates.calc_zmat(mol.link_matrix, mol.coordinates)
        print(zmat)

    def test_mol_modify(self):
        mol = next(hp.MolReader('O=P(OC(C)C)(c1nc(c2cccc(P(OC(C)C)(OC(C)C)=O)n2)ccc1)OC(C)C', 'smi'))

        self.assertEqual(32, len(mol.atoms))
        Sr = hp.Atom(symbol='Sr')
        Sr = mol.add_atom(Sr)
        self.assertEqual(33, len(mol.atoms))

        mol.build3d()
        mol.write('mol2', outdir.joinpath('cheminfo', 'modify_before.mol2'), overwrite=True)
        for a in mol.atoms:
            if a.symbol == 'N' and a.is_aromatic:
                print(mol.add_bond(Sr.idx, a.idx, 1))
        print(mol.smiles)

        mol.build3d('Ghemical', steps=500)
        mol.localopt('Ghemical', steps=500)
        clone = copy(mol)
        clone.write('mol', outdir.joinpath('cheminfo', 'modify_after.mol'), overwrite=True)
        clone.add_hydrogens()
        print(len(clone.atoms))
        clone.remove_hydrogens()
        print(len(clone.atoms))
        clone.write('mol', outdir.joinpath('cheminfo', 'rmh.mol'), overwrite=True)

        t1 = time.time()
        print(mol.longest_path())
        t2 = time.time()
        print(t2-t1)

        print(len(mol.components))

        def show_smiles():
            return mol.smiles

        print(mol.smiles)
        show_time(show_smiles)

    def test_mol_similarity(self):
        """"""
        mol1 = next(hp.MolReader('O=P(OC(C)C)(C1=NC(C2CC=CC(P(OC(C)C)(OC(C)C)=O)=N2)CC=C1)OC(C)C', 'smi'))
        mol2 = next(hp.MolReader('CC1(C)CCC(C)(C)C2=NC(C3=CC=CC(C4=CC=CC(C5=NC6=C(N=N5)C(C)(C)CCC6(C)C)=N4)=N3)=NN=C21', 'smi'))
        mol3 = next(hp.MolReader('O=P(OC(C)C)(C1=NC(C2CC=CC(P(OC(C)C)(OC(C)C)=O)=N2)CC=C1)OC(C)C', 'smi'))

        self.assertEqual(1, mol1.similarity(mol3))
        self.assertEqual(mol2.similarity(mol1), mol2.similarity(mol3))
        self.assertEqual(mol1.similarity(mol2), mol2.similarity(mol1))
        print(mol1.similarity(mol2))

    def test_conformer(self):
        mol = next(hp.MolReader('CCN(CC)C(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1', 'smi'))
        print(mol.conformer_index)
        print(mol.energy)
        print(mol.atoms_partial_charges)

        for a in mol.atoms:
            print(a, a.formal_charge, a.partial_charge)

        print(mol.conformers_count, mol.conformer_index)
        mol.store_current_conformer()
        print(mol.conformers_count, mol.conformer_index)
        mol.store_current_conformer()
        print(mol.conformers_count, mol.conformer_index)

    def test_cclib(self):
        path_log = Path(test.input_dir).joinpath('Am_BuPh-BPPhen.log')
        mol = hp.Molecule.read_by_cclib(path_log)

        print(mol.zero_point)
        print(mol.free_energy)  # - mol.energy - mol.zero_point  # Hartree to eV
        print(mol.entropy)
        print(mol.enthalpy)  # - mol.energy - mol.zero_point
        print(mol.temperature)
        print(mol.pressure)
        print(mol.thermal_energy)  # kcal to ev
        print(mol.capacity)  # cal to ev

    def test_missing_bonds(self):
        mol = next(hp.Molecule.read('CCC.CC.c1ccccc1', 'smi'))
        a = mol.atoms[0]
        spec = mol.graph_spectrum().spectrum

# if __name__ == '__main__':
#     mol = hp.Molecule()

