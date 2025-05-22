# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : parsed_smarts
 Created   : 2025/5/20 20:06
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""

parsed_info = {
    "atoms": [  # list of dict of atoms attributes, sorted by atom index

        # The dict could include any attributes to describe or represent an atom
        # The value of each item in the atom attr dict could be a single values or a set,
        # if the values is a set, which represent a multiply choice attrs for the atoms,
        # The key of atom attr dict may start with "not_...", such as "not_atomic_number".
        # if the key start with "not_.." represent an arbitrary choice except for the values in
        # its values.
        {"atomic_number": [6, 7, ...], "is_aromatic": False|True|None, "any_attrs": ...},  # Atom 0,
        {...}  # Atom1
    ],

    "bonds": {  # dict of dict to describe a bond, each item following the format: (atom1_id, atom2_id): {attr_dict}
        (0, 2): {"bond_orders": 1, ...: ...},  # Definition a single bond between Atom0 and Atom2 with certain attrs
        ...: ...,  # other bonds
    }
}
