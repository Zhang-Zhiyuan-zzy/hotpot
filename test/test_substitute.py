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

    def test_EdgeSubst(self):
        """ Test the substituting operation by EdgeSubst """
        _builder = ob.OBBuilder()
        _builder.SetKeepRings()

        subst = substitute.EdgeSubst("benzene", 'c1ccc[nH]1', [1, 2],
                                     "[cH,CH,CH2,CH3]~[cH,CH,CH2,CH3]", unique_mols=True)

        benzene = hp.Molecule.read_from('OCCOc1ncccc1OCCOCC', "smi")

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

    def test_BondTypeSubst(self):
        """ test the substituting operation by BondTypeSubst """
        double_bond = substitute.BondTypeReplace("double_bond", 'C=C', [0, 1], 'CC')
        triple_bond = substitute.BondTypeReplace("triple_bond", 'C#C', [0, 1], 'CC')
        double_bond.writefile(substitute.substituent_root)
        triple_bond.writefile(substitute.substituent_root)

        octane = hp.Molecule.read_from('CCCCCCCC', 'smi')

        dmols = double_bond(octane)
        tmols = triple_bond(octane)

        print(dmols)
        print(tmols)

    def test_HydroSubst(self):
        """ test the substituting operation by HydroSubst """
        # Define phenyl
        phenyl = substitute.HydroSubst('phenyl', 'c1ccccc1', [0], '[*H,*H,*H2,*H3]')
        phenyl.writefile(substitute.substituent_root)

        frame_mol = hp.Molecule.read_from('c1ccccc1', 'smi')
        generate_mol = phenyl(frame_mol)

        mols = [mol for mol in generate_mol]

        for mol in mols:
            print(mols[0].similarity(mol))

        for atom in mols[0].atoms:
            print(atom.is_aromatic)

    def test_ElemReplace(self):
        """ test the substituting operation by ElemReplace """
        to_sulfur = substitute.ElemReplace('sulfur', 'S', [0], '[O,o,N,n]')
        to_sulfur.writefile(substitute.substituent_root)

        pyrrol = hp.Molecule.read_from('c1ccc[nH]1', 'smi')
        furan = hp.Molecule.read_from('C1CCCCO1', 'smi')

        pyrrol_mols = to_sulfur(pyrrol)
        print(pyrrol_mols)

        furan_mols = to_sulfur(furan)
        print(furan_mols)

    def test_write_read_substituent(self):
        """ test write and read hotpot/data/substituents.json file """
        benzene = hp.Molecule.read_from('c1ccccc1', 'smi')

        self.assertRaises(ValueError, substitute.EdgeSubst,
                          "thiophene01", 'c1ccsc1', [1, 2], "[cH,CH,CH2,CH3]~[cH,CH,CH2,CH3]", unique_mols=True)

        pyrrol01 = substitute.EdgeSubst("pyrrol01", 'c1ccc[nH]1', [1, 2],
                                        "[cH,CH,CH2,CH3]~[cH,CH,CH2,CH3]", unique_mols=True)
        thiophene01 = substitute.EdgeSubst("thiophene01", 'c1cccs1', [1, 2],
                                           "[cH,CH,CH2,CH3]~[cH,CH,CH2,CH3]", unique_mols=True)

        pyrrol01.writefile(substitute.substituent_root)
        thiophene01.writefile(substitute.substituent_root)

        gen_subst = substitute.Substituent.read_from()

        mols = []
        for subst in gen_subst:
            gen_mols = subst(benzene)
            mols.extend(gen_mols)

        for mol in mols:
            print(mol.smiles)
