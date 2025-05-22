# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : parse_smarts
 Created   : 2025/5/20 20:11
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 **Key features supported**:
- **Atoms**: element symbol, aromatic (lowercase), bracketed with atomic number (`[#6]`), atom list, negations (`!`), and extensibility for more attributes.
- **Bonds**: single, double, triple, aromatic, and stereochemistry (`/`, `\`).
- **Ring closures** and basic branch recognition.
- **Extensibility**: To add further SMARTS support, you can expand the `parse_atom` and `parse_bond` methods with more regex and logic.

**To add new attributes**, just expand the dictionaries and parsing logic. If you want recursive SMARTS, SMARTS properties like degree, charge, or more logic, you can hook into the areas noted in comments.

---

**Let me know if you want help extending this for advanced SMARTS features or have a particular test case to decode!**

TODO: BUG! BUG! BUG!
===========================================================
"""
import re
from hotpot.cheminfo.elements import elements

class SmartAtom:
    def __init__(self, index, **attrs):
        self.index = index
        self.attrs = attrs

    def to_dict(self):
        return self.attrs

class SmartBond:
    def __init__(self, atom1, atom2, **attrs):
        self.atom1 = atom1
        self.atom2 = atom2
        self.attrs = attrs

    def key(self):
        return tuple(sorted((self.atom1, self.atom2)))

    def to_dict(self):
        return self.attrs

class SmartsParser:
    ELEMENTS = {s: i for i, s in enumerate(elements.symbols) if s}
    ATOM_NUMBER_TO_SYMBOL = {v: k for k, v in ELEMENTS.items()}  # for reverse lookup

    BOND_MAP = {
        "-": {"bond_order": 1},
        "=": {"bond_order": 2},
        "#": {"bond_order": 3},
        ":": {"is_aromatic": True},
        "~": {"is_any": True},
        "/": {"stereo": "up"},
        "\\": {"stereo": "down"},
    }

    # Atom regex
    ATOM_REGEX = re.compile(
        r"""
        (\[[^\]]+\]) |        # [atom] brackets
        ([A-Z][a-z]?) |       # Atom symbols, e.g. C, Cl
        (c|n|o|s|p|b)         # Aromatic
        """, re.VERBOSE
    )
    BOND_REGEX = re.compile(r"(\-|\=|\#|\:|\/|\\|\~)")

    # For bracketed atom: split logic (comma = OR, & = AND, ; = sequence)
    SPLIT_TOKEN_PATTERN = re.compile(r'\s*,\s*|\s*&\s*|\s*;\s*')  # we only need , and & for now

    def __init__(self):
        self.parsed_info = {
            "atoms": [],
            "bonds": {}
        }
        self.atom_index = 0
        self.curr_bond = None

    def parse(self, smarts_str):
        self.__init__()
        i = 0
        length = len(smarts_str)
        prev_atom_idx = None
        ring_closures = {}
        atoms = []
        bonds = {}

        while i < length:
            m = self.BOND_REGEX.match(smarts_str, i)
            if m:
                bond = m.group(1)
                self.curr_bond = bond
                i += len(bond)
                continue

            m = self.ATOM_REGEX.match(smarts_str, i)
            if m:
                atom_str = m.group(0)
                atom_dict = self.parse_atom(atom_str)
                atom = SmartAtom(self.atom_index, **atom_dict)
                atoms.append(atom)
                if prev_atom_idx is not None:
                    if self.curr_bond is not None:
                        bond_attrs = self.parse_bond(self.curr_bond)
                    else:
                        bond_attrs = self.parse_bond('-')
                    bond = SmartBond(prev_atom_idx, self.atom_index, **bond_attrs)
                    bonds[bond.key()] = bond.to_dict()
                    self.curr_bond = None
                prev_atom_idx = self.atom_index
                self.atom_index += 1
                i += len(atom_str)
                continue

            ch = smarts_str[i]
            if ch.isdigit():
                ring_num = int(ch)
                if ring_num in ring_closures:
                    ring_atom = ring_closures[ring_num]
                    if self.curr_bond:
                        bond_attrs = self.parse_bond(self.curr_bond)
                    else:
                        bond_attrs = self.parse_bond('-')
                    bond = SmartBond(ring_atom, prev_atom_idx, **bond_attrs)
                    bonds[bond.key()] = bond.to_dict()
                    del ring_closures[ring_num]
                else:
                    ring_closures[ring_num] = prev_atom_idx
                self.curr_bond = None
                i += 1
                continue

            if ch == '(' or ch == ')':
                i += 1
                continue

            i += 1

        self.parsed_info["atoms"] = [a.to_dict() for a in atoms]
        self.parsed_info["bonds"] = bonds

        return self.parsed_info

    def parse_atom(self, atom_str):
        info = {}

        if atom_str.startswith('['):
            # Remove brackets and process logic (commas, ands, etc)
            core = atom_str[1:-1].strip()
            if not core:
                return info
            # token_split: inclusive of , (OR) and & (AND)—must treat precedence (AND binds tighter)
            and_parts = [p.strip() for p in core.split('&')]
            # If & is used, must be AND match.
            sets = []
            impossible = False
            for and_part in and_parts:
                # split by comma (OR)
                or_items = [item.strip() for item in and_part.split(',')]
                or_numbers = set()
                not_numbers = set()
                h_count = None
                chiral = None
                for item in or_items:
                    if not item:
                        continue
                    # Parse negation
                    if item.startswith('!'):
                        neg = item[1:]
                        if neg.startswith('#'):
                            # [!#6]
                            not_numbers.add(int(neg[1:]))
                        elif neg in self.ELEMENTS:
                            not_numbers.add(self.ELEMENTS[neg])
                        continue
                    # Atomic number
                    if item.startswith('#'):
                        or_numbers.add(int(item[1:]))
                        continue
                    # Elemental symbol
                    if re.fullmatch(r'[A-Z][a-z]?|\*', item):
                        if item != "*" and item in self.ELEMENTS:
                            or_numbers.add(self.ELEMENTS[item])
                        continue
                    # Hydrogen count
                    if item.startswith('H'):
                        hval = item[1:]
                        h_count = 1 if hval == "" else int(hval)
                        continue
                    # Chirality
                    if item.startswith('@'):
                        # SMARTS: @ (anticlockwise, S), @@ (clockwise, R)
                        chiral = {'@': 'anticlockwise', '@@': 'clockwise'}.get(item, item)
                        continue

                # After processing one AND block (each block may be a set of OR choices)
                if not_numbers:
                    # Remove forbidden numbers from allowed set if set, or note as not_allowed
                    or_numbers = {n for n in or_numbers if n not in not_numbers}
                    info['not_atomic_number'] = not_numbers
                if or_numbers:
                    sets.append(or_numbers)
                if h_count is not None:
                    info['h_count'] = h_count
                if chiral is not None:
                    info['chiral'] = chiral

            # Now, intersect sets from ANDs, or union if only OR
            if sets:
                intersection = set.intersection(*sets) if len(sets) > 1 else sets[0]
                if not intersection:
                    info['unsatisfiable'] = True
                elif len(intersection) == 1:
                    info['atomic_number'] = next(iter(intersection))
                else:
                    info['atomic_number'] = intersection
            elif 'not_atomic_number' in info and not sets:
                # e.g. [!#6]
                info['any_atomic_number_except'] = info.pop('not_atomic_number')

            if info.get('unsatisfiable', False):
                # indicate minimal info: return the error marker only
                return info

        else:  # Unbracketed: C, N, O, ...
            symb = atom_str
            if symb.islower():
                info['is_aromatic'] = True
            symb = symb.capitalize()
            if symb in self.ELEMENTS:
                info['atomic_number'] = self.ELEMENTS[symb]
        return info

    def parse_bond(self, bond_str):
        return self.BOND_MAP.get(bond_str, {"bond_order": 1}).copy()


# Example Usage:
if __name__ == "__main__":
    smarts_examples = [
        "C[C@H](O)C(=O)O",         # Chiral and implicit H
        "[C@@H2]",                 # Two hydrogens, reverse chirality
        "[N;H2]",                  # N, H count 2
        "[!#6,N,O]",               # Not C, or N or O
        "[!#6;N,O]",
        "[N&O&C]",
        "[N,O,C]",
        "c1ccccc1",                # Benzene, aromatic
        "O=C(O)C",                 # Carboxylic acid
    ]
    parser = SmartsParser()
    for s in smarts_examples:
        print(f"SMARTS: {s}")
        print("PARSED:")
        print(parser.parse(s))
        print("-" * 40)

