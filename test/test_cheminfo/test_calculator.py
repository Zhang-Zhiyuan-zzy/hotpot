# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : test_calculator
 Created   : 2025/5/19 17:09
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import os.path as osp
import unittest as ut

import hotpot as hp
from hotpot.cheminfo.calculator import MolChargeCalculator

class TestCalculator(ut.TestCase):
    def test_MolChargeCalculator(self):
        pair_dir = '/home/zz1/docker/proj/raws_ds/reduced_mono_ml_pair'

        error_list = {}
        for file in os.listdir(pair_dir):
            path_pair = osp.join(pair_dir, file)

            try:
                mol = hp.read_mol(path_pair)
            except StopIteration as e:
                print(file)
                raise e

            calc = MolChargeCalculator()
            try:
                print(f"{osp.splitext(file)[0]}, {mol.metals[0].symbol}: {calc(mol)}")
            except Exception as e:
                error_list[file] = mol.smiles

        for key, value in error_list.items():
            print(f"{key}: {value}")
        print(len(error_list))
