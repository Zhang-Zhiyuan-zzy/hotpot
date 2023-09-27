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

        subst = substitute.EdgeJoinSubstituent(
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




