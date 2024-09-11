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
import hotpot as hp

import numpy as np

import networkx as nx
import time

import test


class TestChemInfo(ut.TestCase):

    def test_molecule(self):
        reader = hp.Molecule.read(
            'CCCC(CCCCCCC)CCCCN(CCCC(CC(CCCC(CCCC(CCCC(CC)(CC)(CC)CCC(CC)(CC)(CC)CC)CC(CCCCCC)CC(CC(CCCC(C)'
            '(C)(C)CCCC)C(C(C)(C)(C))CCC(CCC(C(C)(CC)(C))CCCCCCC)CCCCCC)CCCC(CCCCCCCCC)CCCC)CCC(CC)CCC)CCC)CCCCCCCC'
            'C)C(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CCC(CCC(CC)CCCCCCC)CCCCC)CCCCCCCC)C=C1.'
            'CCN(CC)P(OCCC)(OCCC)(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1',
            'smi'
        )

        m = next(reader)

        print(m.atoms)
        print(m.bonds)
        print(m.link_matrix)

        print(m.smiles)
        print(m.coordinates)
        m.add_hydrogens()
        print(m.coordinates)
        print(m.atoms)
        print(m.smiles)

        self.assertNotEqual(m.components, 2)
        c1 = m.components[1]

        self.assertFalse(m.have_normalized_labels)
        m.normalize_labels()
        self.assertTrue(m.have_normalized_labels)

        # build_3d
        self.assertFalse(c1.has_3d())
        c1.build_3d(steps=5000)
        self.assertTrue(c1.has_3d())
        c1.write_by_openbabel('/mnt/c/Users/zhang/OneDrive/Desktop/c1.mol2', 'mol2')

        self.assertTrue(c1.is_organic)

        for a in c1.atoms:
            a.simple_rings()

        for b in c1.bonds:
            b.rings()

        # Generate randomized SMILES
        self.assertNotEqual(c1.smiles, c1.smiles)

        # Generate canonical SMILES
        self.assertEquals(c1.canonical_smiles, c1.canonical_smiles)

        t1 = time.time()
        p = c1.paths
        t2 = time.time()
        lsp = c1.longest_simple_path()
        t3 = time.time()
        smi = c1.smiles
        t4 = time.time()
        can_smi = c1.canonical_smiles
        t5 = time.time()

        print(t2 - t1)
        print(t3 - t2)
        print(t4 - t3)
        print(t5 - t4)

        clone = c1.copy()
        self.assertIsNot(c1, clone)

        atoms_list = list(c1.atoms)
        atoms_array = np.array(c1.atoms)

        t6 = time.time()
        assert atoms_list[1] in atoms_list
        t7 = time.time()
        assert atoms_array[1] in atoms_array
        t8 = time.time()

        a = c1.atoms[5]
        print(a)
        print(a.neighbours)
        print(a.bonds)

    def test_mol_modify(self):
        mol = hp.Molecule.read_by_openbabel('CCN(CC)P(OCCC)(OCCC)(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1', 'smi')

        self.assertEqual(len(mol.atoms), 36)
        self.assertEqual(mol.atom_counts, 36)
        Sr = mol.add_atom('Sr', formal_charge=2)
        self.assertEqual(len(mol.atoms), 37)
        self.assertEqual(mol.atom_counts, 37)
        self.assertIn(Sr, mol)

        mol.build_3d()
        mol.remove_hydrogens()
        mol.write_by_openbabel('/mnt/c/Users/zhang/OneDrive/Desktop/c1.mol2', 'mol2')
        for a in mol.atoms:
            if a.symbol == 'N' and a.is_aromatic:
                print(mol.add_bond(Sr.idx, a.idx, 1))

        mol.build_3d()
        print(mol.metals)
        print(mol.canonical_smiles)
        clone = mol.copy()
        clone.write_by_openbabel('/mnt/c/Users/zhang/OneDrive/Desktop/zzzz.mol2', 'mol2')

        mol.remove_atom(Sr)
        self.assertEqual(len(mol.bonds), 78)
        self.assertIsNot(Sr.molecule, mol)
        self.assertNotIn(Sr, mol)
        self.assertEqual(mol.atom_counts, 76)
        self.assertEqual(len(mol.bonds), 78)

        t1 = time.time()
        add_mol = sum([mol] * 300)
        t2 = time.time()

        print(t2 - t1)

        print(add_mol.atom_counts)
        print(add_mol.components)
        print(mol.canonical_smiles)
        print(add_mol.canonical_smiles)

    def test_mol_similarity(self):
        """"""
        mol1 = hp.Molecule.read_by_openbabel('O=P(OC(C)C)(C1=NC(C2CC=CC(P(OC(C)C)(OC(C)C)=O)=N2)CC=C1)OC(C)C', 'smi')
        mol2 = hp.Molecule.read_by_openbabel('CC1(C)CCC(C)(C)C2=NC(C3=CC=CC(C4=CC=CC(C5=NC6=C(N=N5)C(C)(C)CCC6(C)C)=N4)=N3)=NN=C21', 'smi')

        t1 = time.time()
        print(t1)
        g1 = mol1.graph
        g2 = mol2.graph
        distance = nx.graph_edit_distance(mol1.graph, mol2.graph)
        t2 = time.time()

        print(distance)
        print(t2 - t1)

    def test_conformer(self):
        mol = hp.Molecule.read_by_openbabel('CCN(CC)C(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1', 'smi')
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

if __name__ == '__main__':
    mol = hp.Molecule()

