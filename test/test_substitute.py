"""
python v3.9.0
@Project: hotpot
@File   : test_substitute
@Auther : Zhiyuan Zhang
@Data   : 2023/9/20
@Time   : 11:17
"""
import logging
import unittest as ut

import openbabel.openbabel as ob

import test
import hotpot as hp
from hotpot import substitute


logging.basicConfig(level=logging.INFO)


class TestSubstitute(ut.TestCase):
    """"""
    @classmethod
    def setUpClass(cls) -> None:
        print('Test', cls.__class__)

    def setUp(self) -> None:
        print('running test:', self._testMethodName)

    def tearDown(self) -> None:
        print('Normalize terminate test!', self._testMethodName)

    def test_substitute(self):
        """"""
        _builder = ob.OBBuilder()
        _builder.SetKeepRings()

        subst = substitute.NodeEdgeJoin(
            "benzene",
            hp.Molecule.read_from('C1=CNC=C1', 'smi'),
            [0, 1], "[cH,CH,CH2,CH3]~[cH,CH,CH2,CH3]",
            unique_mols=True
        )

        benzene = hp.Molecule.read_from('OCCOc1ncccc1OCCOCC', "smi")
        # benzene = hp.Molecule.read_from("/home/zz1/mol0.mol2")

        benzene.normalize_labels()
        generate_mol = subst(benzene)
        print(generate_mol)

        for i, mol in enumerate(generate_mol):

            os = [atom for atom in mol.atoms if atom.symbol == 'O']
            sr = mol.add_atom(38)
            for o in os:
                mol.add_bond(sr, o, 1)

            mol.localed_optimize(to_optimal=True)

            print(mol.smiles)
            mol.remove_hydrogens()
            mol.add_hydrogens()
            mol.writefile('mol2', test.test_root.joinpath(f"output/substituted_{i}.mol2"))

    def test_new_substitute(self):
        import openbabel.openbabel as ob
        benzene = hp.Molecule.read_from("c1cnccc1", 'smi')
        pyrrol = hp.Molecule.read_from('c1cncc1', 'smi')

        be_mol = benzene.ob_mol

        py_mol = pyrrol.ob_mol

        atoms_mapping, bonds_mapping = benzene.add_component(pyrrol)

        for atom in benzene.atoms:
            if atom.ob_id < 6:
                atom.generations = 0
            else:
                atom.generations = 1

        for atom in benzene.atoms:
            print(atom.generations)

        benzene.remove_hydrogens()
        benzene.add_bond(5, 6, 1)
        benzene.add_hydrogens()
        benzene.build_3d()
        benzene.writefile('mol2', test.test_root.joinpath("output/substituted.mol2"))

    def test_read_write_substitute(self):
        """ Write a pyrrol12 substituent to hotpot.data.substituents.json, and rebuild a new by read the written info """
        benzene = hp.Molecule.read_from('c1ccccc1', 'smi')
        pyrrol = substitute.NodeEdgeJoin(
            "benzene",
            hp.Molecule.read_from('C1=CNC=C1', 'smi'),
            [0, 1], "[cH,CH,CH2,CH3]~[cH,CH,CH2,CH3]",
            unique_mols=True
        )

        pyrrol.writefile(substitute.substituent_root)

        gen_subst = substitute.Substituent.read_from()

        mols = []
        for subst in gen_subst:
            gen_mols = subst(benzene)
            mols.extend(gen_mols)

        for mol in mols:
            print(mol.smiles)

    def test_H_replace(self):
        """ Replace the hydrogens in a benzene to be phenyl """
        benzene = hp.Molecule.read_from('c1ccccc1', 'smi')
        phenyl_subst = substitute.NodeEdgeJoin(
            "phenyl", hp.Molecule.read_from('c1ccccc1', 'smi'),
            [0], '[H1]', unique_mols=True
        )
        benzene.normalize_labels()
        phenyl_subst.substituent.normalize_labels()

        gen_mols = phenyl_subst(benzene)

        for mol in gen_mols:
            print(mol.smiles)

        for atom in gen_mols[0].atoms:
            print(atom, atom.is_aromatic)

