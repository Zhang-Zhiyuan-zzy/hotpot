"""
python v3.9.0
@Project: hotpot
@File   : test_cheminfo
@Auther : Zhiyuan Zhang
@Data   : 2023/7/16
@Time   : 22:21
Notes:
    Test `hotpot/cheminfo` module
"""
import logging
from pathlib import Path
import unittest as ut

import numpy as np

import hotpot as hp
import hotpot.cheminfo as ci
import test


logging.basicConfig(level=logging.INFO)


class TestMolecule(ut.TestCase):
    """ Test `hotpot/cheminfo/Molecule` class """

    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def test_read_g16log(self):
        """ test the `read_from` method """
        mol_path = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', 'abnormal_output.log')
        mol = hp.Molecule.read_from(mol_path, 'g16log', force=True)

        self.assertIsInstance(mol, hp.Molecule)
        self.assertTrue(mol.has_3d)
        self.assertGreater(mol.conformer_counts, 1)  # the read molecule should have multiply conformers

        # Test the accessibility of Molecule attributes
        self.assertIsInstance(mol.atoms[0], ci.Atom)
        self.assertIsInstance(mol.bonds[0], ci.Bond)

    def test_read_cif(self):
        """ test read a MOF from cif file """
        path_mil = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', 'MIL-101(Cr).cif')
        mil = hp.Molecule.read_from(path_mil)

        crystal = mil.crystal()
        pack_mil = crystal.pack_molecule

        self.assertTrue(mil is crystal.molecule, "Is the molecule of the crystal of a molecule the molecule itself ?")

        # Test to get the attributes of crystal
        self.assertEqual(pack_mil.weight, 259171.4350707429)
        self.assertEqual(crystal.lattice_params.tolist(), [[88.86899, 88.86899, 88.86899], [90.0, 90.0, 90.0]])
        self.assertTrue(np.all(crystal.vectors == crystal.matrix), "Is the vectors identical to the matrix?")
        for a, b in zip(np.dot(crystal.matrix, mil.frac_coordinates.T).T.flatten(), mil.coordinates.flatten()):
            self.assertAlmostEqual(a, b)

        self.assertLess(mil.atom_counts, pack_mil.atom_counts)

        self.assertEqual(crystal.lattice_type, 'Cubic')
        self.assertEqual(crystal.space_group, 'F d 3 m:2')
        self.assertEqual(crystal.volume, 701860.3898079607)

        # Reset the lattice
        matrix = crystal.matrix
        crystal.set_matrix(matrix)
        self.assertTrue(np.all(matrix == crystal.matrix), "the crystal matrix could be specified?")

    def test_graph_representation(self):
        """ test convert a molecule to graph representation """
        mol = hp.Molecule.read_from(test.test_root.joinpath("inputs/struct/Bi-ligand.mol2"))

        idt, feat, adj = mol.graph_representation()

        true_feat = np.array(
            [[8, 2, 2, 4, 0, 0],
             [8, 2, 2, 4, 0, 0],
             [6, 2, 2, 2, 0, 0],
             [6, 2, 2, 2, 0, 0],
             [8, 2, 2, 4, 0, 0],
             [8, 2, 2, 4, 0, 0],
             [83, 6, 2, 3, 10, 14]])

        true_adj = np.array([
            [0, 1, 2, 3, 3, 6, 0, 1, 2, 3, 3, 6],
            [2, 2, 3, 4, 5, 4, 2, 2, 3, 4, 5, 4]
        ])

        self.assertEqual(idt, "Bi-ligand")
        self.assertTrue(np.all(feat == true_feat), "the feature matrix can't match")
        self.assertTrue(np.all(true_adj == adj), "the adjacency can't match")

    def test_add_remove_atom_bonds(self):
        """"""
        mol = hp.Molecule.read_from('c1cnc(O)cc1', 'smi')
        mol.build_3d()
        # mol.unset_coordinates()

        sr = mol.add_atom('Sr')

        for o in [a for a in mol.atoms if a.symbol in ['O', 'N']]:
            mol.add_bond(sr, o, 1)

        mol.remove_hydrogens()
        mol.build_3d()
        mol.assign_bond_types()
        mol.normalize_labels()

        print(mol.smiles)
        print(mol.smarts())

        for a in mol.atoms:
            print(a, a.coordinates)

        for b in mol.bonds:
            print(b, b.length, b.ideal_length)

        mol.remove_atoms(sr)
        print(mol.smiles)
        print(mol.smarts())

    def test_shortest_path_search(self):
        """"""
        mol = hp.Molecule.read_from('C' * 10, 'smi')
        atom_indices = mol.shortest_path(0, 9)
        mol.shortest_path(0, 9, get_all=True)
        mol.shortest_path(0, 9, return_atoms=True)
        mol.shortest_path(0, 9, get_all=True, return_atoms=True)

        self.assertEqual(atom_indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_operators(self):
        """ test all operator method for all objects in cheminfo module """
        mol1 = hp.Molecule.read_from('c1ccccc1', 'smi')
        mol2 = hp.Molecule.read_from('c1ccccc1', 'smi')
        mol3 = hp.Molecule.read_from('c1ccccc1', 'smi')
        mol4 = hp.Molecule.read_from('c1cnccc1', 'smi')

        mols = [mol2, mol3]

        self.assertEqual(mol1, mol2)
        self.assertIn(mol1, mols)
        self.assertNotIn(mol4, mols)

    def test_rings(self):
        """ test the Ring class and other related code """
        mol = hp.Molecule.read_from('c1cccc2c1Cc3c(C2)cc[nH]3', 'smi')

        ring = mol.lssr[0]
        joint_rings, atoms = ring.joint_rings(expand=False)
        expand_rings, atoms = ring.joint_rings()
        aromatic_rings, atoms = ring.joint_rings(aromatic=True)

        self.assertEqual(len(joint_rings), 2)
        self.assertEqual(len(expand_rings), 3)
        self.assertEqual(len(aromatic_rings), 1)

        ring = mol.lssr[2]
        aromatic_rings, atoms = ring.joint_rings(aromatic=True)
        self.assertEqual(len(aromatic_rings), 0)


