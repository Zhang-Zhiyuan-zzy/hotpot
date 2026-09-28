# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : test_mol_assemble
 Created   : 2025/5/20 9:59
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import unittest
import unittest as ut


import hotpot as hp
import hotpot.cheminfo.mol_assemble as mol_assemble



class TestMolAssemble(unittest.TestCase):
    def setUp(self):
        print(f"Staring test MolAssemble : {__file__}")

    def test_atom_link_atom(self):
        ...

    def test_bond_shoulder(self):
        ...

    def test_ring_wedge_is_public(self):
        self.assertIs(hp.RingWedge, mol_assemble.RingWedge)
