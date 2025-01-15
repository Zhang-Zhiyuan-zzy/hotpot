"""
python v3.9.0
@Project: hp5
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/6/5
@Time   : 19:15
"""
from os.path import join as opj
from pathlib import Path
import unittest as ut
from copy import copy
from itertools import product

import hotpot as hp
from hotpot.cheminfo.core import InternalCoordinates
from hotpot.utils.tools import show_time

import numpy as np

import networkx as nx
import time

import test


outdir = Path(test.output_dir)


class TestChemInfo(ut.TestCase):

    @ut.skip
    def test_MolReader_iter(self):
        from tqdm import tqdm
        reader = hp.MolReader('/mnt/d/zhang/OneDrive/Papers/Gen3D/out.mol2')
        for m in tqdm(reader):
            # assert isinstance(m.formula, str)
            pass

    @ut.skip
    def test_pb_reader(self):
        from tqdm import tqdm
        from openbabel import pybel as pb
        from hotpot.cheminfo.obconvert import to_arrays
        from hotpot.cheminfo.core_ import Molecule
        reader = pb.readfile('mol2', '/mnt/d/zhang/OneDrive/Papers/Gen3D/out.mol2')
        for m in tqdm(reader):
            mol = Molecule(*to_arrays(m.OBMol)[:2])


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
        b = mol.bonds[20]
        self.assertIs(b.mol, mol)
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
        self.assertIs(clone.rings[0].mol, clone)

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

        c1.update_angles()
        c1.update_torsions()
        print(len(c1.angles))
        print(len(c1.torsions))
        print(c1.torsions[30].degrees)
        t1 = time.time()
        print(c1.simple_paths(cutoff=3))
        t2 = time.time()
        print(t2-t1)

        rings = c1.rings
        print(rings)
        print(rings[0].bonds)
        self.assertIn(rings[0].bonds[1], rings[0])
        self.assertNotIn(rings[0].bonds[2], rings[1])

        self.assertTrue(all(a.in_ring for r in rings for a in r.atoms))
        self.assertTrue(all(b.in_ring for r in rings for b in r.bonds))

        print(c1.to_rdmol())

    def test_atompair(self):
        import hotpot as hp
        mol = next(hp.MolReader( 'CCCC(CCCCCCC)CCCCN(CCCC(CC(CCCC(CCCC(CCCC(CC)(CC)(CC)CCC(CC)(CC)(CC)CC)CC(CCCCCC)CC(CC(CCCC(C)'
            '(C)(C)CCCC)C(C(C)(C)(C))CCC(CCC(C(C)(CC)(C))CCCCCCC)CCCCCC)CCCC(CCCCCCCCC)CCCC)CCC(CC)CCC)CCC)CCCCCCCC'
            'C)C(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CCC(CCC(CC)CCCCCCC)CCCCC)CCCCCCCC)C=C1', 'smi'))

        mol.build3d()
        print(mol.coordinates)
        t1 = time.time()
        pairs = mol.atom_pairs
        print(time.time() - t1)

        t1 = time.time()
        print(pairs.pair_distance)
        print(time.time() - t1)

        t1 = time.time()
        print(mol.pair_dist)
        print(time.time() - t1)


    def test_link_atoms(self):
        mol = next(hp.MolReader(opj(test.input_dir, 'Am_BuPh-BPPhen.log')))
        # print(mol.bonds)
        # mol.link_atoms()
        print(mol.bonds)

        self.assertEqual(mol.conformers._coordinates.shape, (54, 73, 3))
        mol.write(opj(test.output_dir, 'cheminfo', 'Am_BuPh-BPPhen.sdf'), overwrite=True)

    def test_molblock(self):
        mol = next(hp.MolReader('c1cncc3c1c2c(S3(=O)=O)c[nH]c2P(=O)(O)O', 'smi'))
        for b in mol.bonds:
            print(f"{b}: is aromatic? {b.is_aromatic}; is rotatable? {b.rotatable}")
        for t in mol.torsions:
            print(f"{t} is rotatable? {t.rotatable}")

        for r in mol.rings:
            print(f"{r} is aromatic? {r.perceive_aromatic()}")

        for a in mol.atoms:
            print(a.open_shell_electrons)

        for a in mol.atoms:
            print(a.missing_electrons_element)

        print("Hybridization:")
        for a in mol.atoms:
            print(a.hyb)

        print("Oxidation State:")
        for a in mol.atoms:
            print(a.symbol, a.oxidation_state)

        mol.build3d(steps=5000)
        mol.write(opj(test.output_dir, 'cheminfo', 'mol.mol2'), 'mol2', overwrite=True)

    def test_internal_coordinates(self):
        reader = hp.MolReader('CCN(CC)P(OCCC)(OCCC)(=O)C1=NC2=C(C=C1)C=CC1=C2N=C(C(=O)N(CC)CC)C=C1', 'smi')
        mol = next(reader)
        mol.build3d()
        zmat = InternalCoordinates.calc_zmat(mol.link_matrix, mol.coordinates)
        print(zmat)

    def test_build3d(self):
        mol = next(hp.MolReader(
            # 'O=P(OC(C)C)(c1nc(c2cccc(P(OC(C)C)(OC(C)C)=O)n2)ccc1)OC(C)C'
            # 'c1ccc(cc1)C(c1ccccc1)(c1ccccc1)N1CCN(CCN2CCN(CCN(CC2)C(c2ccccc2)(c2ccccc2)c2ccccc2)C(c2ccccc2)(c2ccccc2)c2ccccc2)CCN(CC1)C(c1ccccc1)(c1ccccc1)c1ccccc1',
            # 'C[C@H]1O[C@@H](O[C@@H]2[C@@H](COCc3ccccc3)O[C@@H]([C@@H]([C@@H]2OCc2ccccc2)OCc2ccccc2)O[C@@H]2[C@@H](OCc3ccccc3)CN([C@@H]2COCc2ccccc2)C(=O)OCc2ccccc2)[C@H]([C@H]([C@@H]1OCc1ccccc1)OCc1ccccc1)OCc1ccccc1',
            "BrCCCCCCOc1cccc(c1)C1=C2C=CC(=N2)C(=c2ccc(=C(C3=NC(=C(c4[nH]c1cc4)c1cccc(c1)OCCCCCCBr)C=C3)c1cccc(c1)OCCCCCCBr)[nH]2)c1cccc(c1)OCCCCCCBr",
            'smi'))

        Ga = mol.create_atom(symbol='Ga', formal_charge=3)

        coordination_atoms = [a for a in mol.atoms if a.symbol in ['N', 'O']]
        coordination_atoms = np.random.choice(coordination_atoms, min(6, len(coordination_atoms)), replace=False)
        for a in coordination_atoms:
            b = mol.add_bond(Ga, a, 1)
            self.assertTrue(b.has_metal)

        print(len(mol.bonds))
        assert isinstance(mol, hp.Molecule)

        mol.complexes_build_optimize_(save_screenshot=True)
        mol.write(opj(test.output_dir, 'cheminfo', 'built_mol.sdf'), 'sdf', overwrite=True)
        mol.write(opj(test.output_dir, 'cheminfo', 'built_mol_single.sdf'), 'sdf', overwrite=True, write_single=True)

    def test_judge_intersect(self):
        def _artificial_mol():
            mol = hp.Molecule()
            for i in range(6):
                mol.create_atom(atomic_number=6, coordinates=(2 * np.cos(i / 3 * np.pi), 2 * np.sin(i / 3 * np.pi), 0))
            for i in range(6):
                if i != 5:
                    mol.add_bond(i, i + 1, bond_order=2 if i % 2 == 0 else 1)
                else:
                    mol.add_bond(i, 0, bond_order=1)

            mol.create_atom(atomic_number=8, coordinates=(0, 0, -2))
            mol.create_atom(atomic_number=8, coordinates=(0, 0, 2))
            mol.add_bond(6, 7, 1)

            mol.write(opj(test.output_dir, 'cheminfo', f'mol.sdf'), overwrite=True)
            for i, (r, b) in enumerate(product(mol.rings, mol.bonds)):
                if r.is_bond_intersect_the_ring(b):
                    cycle = r.cycle_places
                    line = b.bond_line

                    center_point = cycle.center
                    intersect_points = cycle.line_intersect_points(line)

                    mr = r.to_mol()
                    mb = b.to_mol()

                    mol_ = mr + mb
                    mol_.create_atom(symbol='Cm', coordinates=center_point)
                    for p in intersect_points:
                        mol.create_atom(atomic_number=0, coordinates=p)

                    mol_.write(opj(test.output_dir, 'cheminfo', f'rb{i}.sdf'), overwrite=True)

        mol_intersect = next(hp.MolReader(Path(test.input_dir).joinpath('intersect.sdf')))
        mol_not = next(hp.MolReader(Path(test.input_dir).joinpath('not_intersect.sdf')))

        for i, (r, b) in enumerate(product(mol_intersect.rings, mol_intersect.bonds)):
            if r.is_bond_intersect_the_ring(b):
                cycle = r.cycle_places
                line = b.bond_line

                center_point = cycle.center
                intersect_points = cycle.line_intersect_points(line)

                mr = r.to_mol()
                mb = b.to_mol()

                mrb = mr + mb
                mrb.create_atom(symbol='Cm', coordinates=center_point)
                for p in intersect_points:
                    mrb.create_atom(atomic_number=0, coordinates=p)

                mrb.write(opj(test.output_dir, 'cheminfo', f'rb{i}.sdf'), overwrite=True)

    def test_mol_similarity(self):
        """"""
        mol1 = next(hp.MolReader('O=P(OC(C)C)(C1=NC(C2CC=CC(P(OC(C)C)(OC(C)C)=O)=N2)CC=C1)OC(C)C', 'smi'))
        mol2 = next(hp.MolReader('CC1(C)CCC(C)(C)C2=NC(C3=CC=CC(C4=CC=CC(C5=NC6=C(N=N5)C(C)(C)CCC6(C)C)=N4)=N3)=NN=C21', 'smi'))
        mol3 = next(hp.MolReader('O=P(OC(C)C)(C1=NC(C2CC=CC(P(OC(C)C)(OC(C)C)=O)=N2)CC=C1)OC(C)C', 'smi'))

        self.assertEqual(1, mol1.similarity(mol3))
        self.assertEqual(mol2.similarity(mol1), mol2.similarity(mol3))
        self.assertEqual(mol1.similarity(mol2), mol2.similarity(mol1))
        print(mol1.similarity(mol2))

    def test_read_g16log_file(self):
        mol = next(hp.MolReader(Path(test.input_dir).joinpath('Am_BuPh-BPPhen.log')))
        self.assertEqual(mol.conformers_number, 54)
        self.assertEqual(mol.coordinates.shape, (73, 3))
        self.assertEqual(mol.zero_point, 16.266286413195473)
        self.assertEqual(mol.energy, -60574.40216122108)
        self.assertEqual(mol.force.shape, (73, 3))
        self.assertEqual(mol.conformers._force.shape, (54, 73, 3))
        self.assertEqual(mol.gibbs, -60560.09243823103)
        self.assertEqual(mol.thermo, 17.2572589675573)
        self.assertEqual(mol.capacity, 0.00616935257190196)

        num_atoms = len(mol.atoms)
        mol.write(opj(test.output_dir, 'cheminfo', 'Am_BuPh-BPPhen_rmh.gjf'), overwrite=True)

        mol.remove_hydrogens()

        mol.add_hydrogens()
        self.assertEqual(len(mol.atoms), num_atoms)
        mol.optimize()
        mol.write(opj(test.output_dir, 'cheminfo', 'Am_BuPh-BPPhen.gjf'), overwrite=True)

    def test_export_gjf(self):
        mol = next(hp.MolReader(
            "BrCCCCCCOc1cccc(c1)C1=C2C=CC(=N2)C(=c2ccc(=C(C3=NC(=C(c4[nH]c1cc4)c1cccc"
            "(c1)OCCCCCCBr)C=C3)c1cccc(c1)OCCCCCCBr)[nH]2)c1cccc(c1)OCCCCCCBr",
            'smi'))

        Ga = mol.create_atom(symbol='Ga', formal_charge=3)

        coordination_atoms = [a for a in mol.atoms if a.symbol in ['N', 'O']]
        coordination_atoms = np.random.choice(coordination_atoms, min(6, len(coordination_atoms)), replace=False)
        for a in coordination_atoms:
            b = mol.add_bond(Ga, a, 1)
            self.assertTrue(b.has_metal)

        print(len(mol.bonds))
        assert isinstance(mol, hp.Molecule)

        mol.complexes_build_optimize_(save_screenshot=True)
        mol.write(opj(test.output_dir, 'cheminfo', 'built_mol_single.gjf'), overwrite=True, write_single=True)

    def test_add_hydrogen(self):
        mol = next(hp.MolReader(
            # "OC(=O)CN1[C@@H](CN(C2(C1=O)COCCOC2)C(=O)OC(C)(C)C)c1ccc(cc1)F",
            'CC(C)(C)OC(=O)N12CC(c3ccc(F)cc3)N34CC5=O[Ga]613([OH]5)O1CCO6CC2(C1)C4=O',
            'smi'))

        mol.build3d()
        mol.optimize()

        mol.write(opj(test.output_dir, 'cheminfo', 'addh_before.mol'), overwrite=True)
        t1 = time.time()
        mol.add_hydrogens()
        t2 = time.time()
        print(t2-t1)
        mol.write(opj(test.output_dir, 'cheminfo', 'addh_after.mol'), overwrite=True)
        mol.optimize(save_screenshot=True)
        mol.write(opj(test.output_dir, 'cheminfo', 'addh_opti.sdf'), overwrite=True)
        mol.write(opj(test.output_dir, 'cheminfo', 'addh_opti.gjf'), overwrite=True)

    @ut.skip('not implemented')
    def test_missing_bonds(self):
        mol = next(hp.Molecule.read('CCC.CC.c1ccccc1', 'smi'))
        a = mol.atoms[0]
        spec = mol.graph_spectrum().spectrum
