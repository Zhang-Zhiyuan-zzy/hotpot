"""
python v3.9.0
@Project: hotpot
@File   : test_chem
@Auther : Zhiyuan Zhang
@Data   : 2023/11/21
@Time   : 20:56
"""
from pathlib import Path
from unittest import TestCase
import inspect

import numpy as np
import logging


import hotpot as hp
import test
from hotpot.cheminfo import *

logging.basicConfig(level=logging.INFO)


def find_out_cxx_error_attr(obj):
    for attr_name in [an for an in dir(obj) if not an.startswith('_')]:
        print(f'{obj}.{attr_name}:')
        try:
            print(getattr(obj, attr_name))
        except:
            print('Got a python Error')


class TestChem(TestCase):
    def test_operation(self):
        mol = Molecule.read_from('c1ccccc1.c1cccc(C(=O)O)c1', 'smi')
        sr = mol.add_atom('Sr')
        self.assertIn(sr, mol.atoms)

        for o in [a for a in mol.atoms if a.symbol == 'O']:
            mol.add_bond(sr, o, 1)

        mol.build_3d()
        mol.remove_hydrogens()
        mol.add_hydrogens()

        self.assertEqual(mol.atoms_dist_matrix.shape, (len(mol.atoms), len(mol.atoms)))

        for bond in mol.bonds:
            self.assertIsInstance(mol.bond(bond.atom1.idx, bond.atom2.idx), Bond)
        for angle in mol.angles:
            self.assertIsInstance(angle, Angle)
        for torsion in mol.torsions:
            self.assertIs(torsion.molecule, mol)
        for component in mol.components:
            self.assertNotEqual(component.refcode, mol.refcode)
        for ring in mol.lssr:
            self.assertIs(ring.molecule, mol)

        cryst = mol.compact_crystal()
        self.assertNotEqual(cryst.molecule.refcode, mol.refcode)

        cryst = mol.compact_crystal(inplace=True)
        self.assertEqual(cryst.molecule.refcode, mol.refcode)

        self.assertIs(mol.crystal, cryst)
        # if given OBMol is registered in the Molecule list, retrieve the registered one instead of create a new
        self.assertIs(Molecule(mol.ob_mol), mol)

        clone = mol.copy()
        clone.remove_atoms(*clone.metals)

        self.assertEqual(clone.smiles, 'c1ccccc1.[O]C(=O)c1ccccc1')
        self.assertFalse(clone.metals)
        self.assertTrue(mol.metals)
        self.assertNotEqual(len(mol.atoms), len(clone.atoms))
        self.assertNotEqual(len(mol.bonds), len(clone.bonds))

        clone.add_bond(clone.atom(6), clone.atom(7), 1)
        clone.remove_atoms()
        clone.add_hydrogens()
        self.assertEqual(clone.smiles, 'OC(=O)c1cccc(c1)c1ccccc1')

        clone.remove_atoms(clone.atoms[-1], clone.atoms[-2])
        for atom in clone.atoms:
            self.assertEqual(atom.idx, clone.atom(atom.idx).idx)
        for bond in clone.bonds:
            self.assertEqual(bond.idx, clone.bond(bond.atom1, bond.atom2).idx)
            self.assertEqual(bond.idx, clone.bond(bond.atom1.idx, bond.atom2.idx).idx)
            self.assertEqual(bond.idx, clone.bond(bond.atom1.idx, bond.atom2.idx).idx)

            self.assertEqual(bond.atom1.idx, clone.bond(bond.atom1, bond.atom2).atom1.idx)
            self.assertEqual(bond.atom2.idx, clone.bond(bond.atom1, bond.atom2).atom2.idx)

        comp1, comp2 = mol.components
        new_mol = comp1 + Molecule.read_from(comp2.smiles, 'smi')

        print(mol.smiles, comp1.smiles)

    def test_io(self):

        # test read from gjf
        mol_path = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', 'abnormal_output.log')
        mol = hp.Molecule.read_from(mol_path, 'g16', force=True)
        self.assertEqual(mol.energy, -380867.02680215525)
        self.assertTrue(mol.has_3d)

        # test read from cif
        path_mil = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', 'MIL-101(Cr).cif')
        mil = hp.Molecule.read_from(path_mil)

        crystal = mil.crystal

        self.assertTrue(mil is crystal.molecule, "Is the molecule of the crystal of a molecule the molecule itself ?")

        self.assertEqual(crystal.lattice_params.tolist(), [[88.86899, 88.86899, 88.86899], [90.0, 90.0, 90.0]])
        self.assertTrue(np.all(crystal.vectors == crystal.matrix), "Is the vectors identical to the matrix?")
        for a, b in zip(np.dot(crystal.matrix, mil.frac_coordinates.T).T.flatten(), mil.coordinates.flatten()):
            self.assertAlmostEqual(a, b)

        self.assertEqual(crystal.lattice_type, 'Cubic')
        self.assertEqual(crystal.space_group, 'F d 3 m:2')
        self.assertEqual(crystal.volume, 701860.3898079607)

        # Reset the lattice
        matrix = crystal.matrix
        crystal.set_matrix(matrix)
        self.assertTrue(np.all(matrix == crystal.matrix), "the crystal matrix could be specified?")

        pack_mil = crystal.pack_molecule
        self.assertLess(len(mol.atoms), len(pack_mil.atoms))
        self.assertLess(mol.weight, pack_mil.weight)

    def test_read_by_g16log_format(self):
        mol_path = Path(hp.hp_root).joinpath('..', 'test', 'inputs', 'struct', '100_99_C28H42N2NdO6P2.log')
        mol = hp.Molecule.read_from(mol_path, 'g16log')

        self.assertEqual(mol.energy, -64397.2805210937)
        self.assertEqual(mol.zero_point, 18.643010522078807)
        self.assertEqual(mol.free_energy, -2.2763656400493026)
        self.assertEqual(mol.enthalpy, 1.1911813092738477)
        self.assertEqual(mol.entropy, 0.011630209456066938)
        self.assertEqual(mol.thermal_energy, 19.81002274419889)
        self.assertEqual(mol.capacity, 0.0068090177987163505)

    def test_shortest_path_search(self):
        mol = hp.Molecule.read_from('C' * 10, 'smi')
        atom_paths = mol.shortest_paths(mol.atoms[0], mol.atoms[9])

        self.assertEqual(len(atom_paths), 1)
        self.assertEqual([a.idx for a in atom_paths[0]], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_thermo(self):
        mol = Molecule.read_from('c1ccccc1', 'smi')
        tmo = mol.get_thermo(T=298.15, P=101375)

    def test_solvents(self):
        mol = Molecule.read_from('CO.OC(=O)C', 'smi')
        sr = mol.add_atom('Sr')
        mol.normalize_labels()
        mol.add_bond(mol.atom('O1'), sr, 1)
        mol.add_bond(mol.atom('O2'), sr, 1)
        mol.build_3d()
        #
        find_out_cxx_error_attr(mol)
        find_out_cxx_error_attr(sr)

        # mol.remove_solvents()
        #
        # print(mol.atoms)
        # print(dict(hp.cheminfo._chem._molecule_dict))

    def test_remove_polar_hydrogens(self):
        mol = hp.Molecule.read_from("O=C(O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)O)C=C1", 'smi')
        mol.add_hydrogens()
        mol.remove_polar_hydrogens()

        self.assertEqual(mol.charge, 2)

    def test_build_3d(self):
        mol = Molecule.read_from("CC1=CC(=NC=C1)C2=NC=CC(=C2)C", 'smi')
        mol.build_3d()
        mol.localed_optimize(steps=500)

        mol.writefile('mol2', test.test_root.joinpath('output/build3d.mol2'))
        print(mol.smiles)

    def test_templates(self):
        """"""
        print(hp.Molecule.get_templates_list())
        print(hp.Molecule.get_templates_mol('VO2').charge)
        print(hp.Molecule.get_templates_mol('VO2').atoms)

    def test_calculate_molecule_charge(self):
        mol = Molecule.read_from('CO[Sr]OC(=O)', 'smi')
        self.assertRaises(AttributeError, mol.calculate_molecule_charges)
        print(mol.atoms)
        mol.add_hydrogens()
        print(mol.atoms)
        charges = mol.calculate_molecule_charges()
        print(charges)
        print(mol, mol.charge)

