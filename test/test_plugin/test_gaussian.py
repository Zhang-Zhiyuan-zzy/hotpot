"""
python v3.9.0
@Project: hp5
@File   : test_gaussian
@Auther : Zhiyuan Zhang
@Data   : 2024/6/28
@Time   : 10:16
"""
import os
from glob import glob
from os.path import join as opj
import unittest as ut
import hotpot as hp
from hotpot.plugins.qm.gaussian import run_gaussian, GaussOut, export_results

from test import test_dir, input_dir, output_dir


class TestGaussian(ut.TestCase):
    def test_run_gaussian(self):
        mol = next(hp.MolReader("c1ccccc1"))
        mol.build3d()
        mol.optimize()

        print(mol.coordinates)

        output = run_gaussian(mol)


    def test_GaussOut(self):
        # out = GaussOut.read_file(opj(input_dir, 'Am_BuPh-BPPhen.log'))
        # print(out.get_times('minute'))
        #
        # parse_dict = out.parse()
        # mol = out.export_mol()
        #
        # print(mol.coordinates)
        # print(mol.charge)
        # print(mol.gibbs)
        #
        # out.update_mol(mol)
        #
        # print(out.export_pandas_series())

        # df = export_results(
        #     *[p for p in glob('/mnt/d/zhang/OneDrive/Papers/dy/log/pair/normal/*.log')]
        # )
        # print(df)

        ligand_path = '/mnt/d/zhang/OneDrive/Papers/dy/gjf/ligand'
        for p in glob('/mnt/d/zhang/OneDrive/Papers/dy/log/pair/error/*.log'):
            try:
                out = GaussOut.read_file(p)
                mol = out.export_mol()
                mol.remove_metals()
                mol.charge = mol.charge - 3

                mol.write(
                    opj(ligand_path, os.path.basename(p).replace('.log', '.gjf')),
                    fmt='gjf',
                    # calc_mol_charge=True,
                    overwrite=True,
                    ob_opt={'b': None}
                )

            except ValueError:
                os.remove(p)
                print(os.path.basename(p))

