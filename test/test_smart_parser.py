# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : test_smart_parser
 Created   : 2025/7/7 21:21
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import unittest
from hotpot.cheminfo.search.smarts_parser import *


class TestSmartsParser(unittest.TestCase):
    # SMARTS examples for bracket validation
    valid_brackets = [
        "C1CCCCC1",
        "[CH3]C[CH2]C",
        "C[C@H](N)C(=O)O",
    ]
    invalid_brackets = [
        "CC]",           # missing opening bracket
        "[C@H]1CC[C@H](C(=O)O)C1[",  # extra '[' at end
        "[C[CH]]C",       # nested '[' inside brackets
        "[CH]C[CH]]C",   # stray ']' inside bracket content
    ]

    # SMARTS examples for numeric pairing
    valid_numeric_smarts = [
        "C[C@H](N)C(=O)O[C@@H]1CC[C@H](C(=O)O)CC1",
        "c1cc(ccc1)C[C@H]2CC[C@H](C(=O)O)C2C(=O)O",
        "[NH4+]-C(=O)C[C@H](N)C(=O)O[C@H]1CCC[C@H]1N",
        "[O-]C(=O)C[C@H]1CC[C@H](C(=O)O)C1N[C@@H](C)C",
        "[C@H]1CN([C@@H](C(=O)O)CC1)C(=O)C[C@H](N)C(=O)O",
        "C%12C[C@H](O)CC[C@@H](N)C(=O)C%12C[C@H](C)O",
        "c1nccc2[nH]c[nH]c2c1C[C@H](N)C(=O)O",
        "[C;R;X4]1CC[C@H]2C[C@@H](C(=O)O)NC12",
        "[#6,#7][C@H](N)C(=O)O[C@H]1CC[C@H](C)CC1C(=O)O",
        "c1c[nH]c2c1c([C@H]3[C@H](N)CCC3C(=O)O)c[nH]2C(=O)O",
        "[*:1]C[C@H](N)C(=O)O[C@H]1CC[C@H](C)C1C(=O)O",
        "[O;H1]C(=O)C[C@H](N)C(=O)O[C@@H]1CC[C@H](C)C1",
        "[NH3+]C[C@H](O)CCC[C@H](N)C(=O)O[C@H]1CC[C@H](C)C1",
        "c1ccc2c(c1)[nH]c3c2[C@@H]4CC[C@H](C(=O)O)N4c3C(=O)O",
        "[C@]3(C[C@H]2CC[C@H](C(=O)O)N2)C[C@H]4CCC4C3=O",
        "[CH2;X4;v2]C[C@H](N)C(=O)O[C@@H]1CC[C@H](C)C1",
        "C[N+](C)(C)CC[C@H](C(=O)O)C[C@H]1C[C@@H](O)CC1",
        "C[C@H](O)[C@@H]1CN(C(=O)O)CC[C@H](C)C1C(=O)O",
        "[O-]C(=O)[C@H]1CC[C@@H](C(=O)O)N1C[C@H](N)C(=O)O",
        "c1cc2cc[c@@]3(c2c(c1)C(=O)O)CC[C@H]3N[C@H](C)C(=O)O",
        "[C@@]1(CC[C@H]2CC[C@@H](C(=O)O)N2)C(=O)O[C@H]1C",
        "C[C@@H]1C[C@H](C(=O)O)N[C@H](C)CCC1C(=O)O",
        "[nH]1ccc2c(c1)C[C@H]3CC[C@@H](C(=O)O)N3C2=O",
        "C[C@H](N)[C@H](C(=O)O)CC[C@H]1CC[C@H](C)C1",
        "[C@H]1CC[C@@H](C(=O)O)N1C(=O)C[C@H](N)CCC",
        "[C;!R;r5]1CC[C@H](C(=O)O)N[C@H]2CC[C@@H]12C",
        "[#8;!D2]C(=O)[C@H]1CC[C@H](C(=O)O)N1C[C@H](N)C",
        "[C;R;H1]1CC[C@H](C(=O)O)OCC[C@H]2CC[C@@H]12N"
    ]
    invalid_numeric_smarts = [
        "C1CC",           # single '1'
        "C%12CC",         # single '%12'
        "C1CCC1C1",       # three '1's
    ]

    def test_validate_brackets_valid(self):
        """validate_brackets should accept matching, non-nested brackets."""
        for sm in self.valid_brackets:
            try:
                validate_brackets(sm)
            except ValueError as e:
                self.fail(f"validate_brackets raised on valid input {sm!r}: {e}")

    def test_validate_brackets_invalid(self):
        """validate_brackets should reject mismatched or nested brackets."""
        for sm in self.invalid_brackets:
            with self.assertRaises(ValueError, msg=f"{sm!r} did not raise ValueError"):
                validate_brackets(sm)

    def test_extract_numeric_pairs_valid(self):
        """extract_numeric_pairs should parse valid SMARTS without error."""
        for sm in self.valid_numeric_smarts:
            try:
                extract_numeric_pairs(sm, extract_brackets(sm))
            except Exception as e:
                self.fail(f"extract_numeric_pairs raised on valid input {sm!r}: {e}")

    def test_extract_numeric_pairs_unmatched_numeric(self):
        """extract_numeric_pairs should raise RuntimeError for unmatched numeric labels."""
        for sm in self.invalid_numeric_smarts:
            with self.assertRaises(RuntimeError, msg=f"{sm!r} did not raise RuntimeError"):
                extract_numeric_pairs(sm, extract_brackets(sm))
