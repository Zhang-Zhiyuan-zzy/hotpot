# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : test_elements
 Created   : 2025/5/21 16:35
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import unittest as ut

from numba.cuda import shared

import hotpot.cheminfo.elements as ele


class TestElements(ut.TestCase):
    def test_get_elements_properties_dict(self):
        ele_prop = ele.element_properties
        self.assertEqual(len(ele_prop.keys()), 118)

        for key in ele_prop.keys():
            self.assertIn(key, ele.elements.symbols)

        shared_props = set(ele_prop['H'].keys())
        for key, value in ele_prop.items():
            shared_props &= set(value.keys())
        print(f"Shared properties({len(shared_props)}): {shared_props}")

        unique_props = {key: set(value.keys())-shared_props for key, value in ele_prop.items()}
        print(f"Unique properties: {unique_props}")

    def test_elements_properties_has_value(self):
        ele_prop = ele.element_properties
        shared_props = set(key for key, values in ele_prop['H'].items() if isinstance(values, list) and values[0] is not None)

        for key, value in ele_prop.items():
            shared_props &= set(k for k, v in value.items() if isinstance(v, list) and v[0] is not None)

        print(f"Shared properties({len(shared_props)}) with values: {shared_props}")

    def test_check_null_elements(self):
        attr_name = "ionization_energies"
        ele_prop = getattr(ele.elements, attr_name)

        for e, value in ele_prop.items():
            if not isinstance(value, list) or v[0] is None:
                print(e)

