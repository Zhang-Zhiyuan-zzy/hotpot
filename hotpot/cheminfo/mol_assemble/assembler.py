# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : searcher_definition
 Created   : 2025/7/18 15:26
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import random
from typing import Union, Iterable, Literal, Optional

from hotpot import read_mol
from hotpot.cheminfo.core import Atom, Bond, Molecule
from hotpot.cheminfo.search import Searcher, Substructure, QueryAtom, QueryBond
from hotpot.cheminfo.mol_assemble.assembler import Fragment, shoulder_bond_action, atom_link_atom_action, bond_order_add


def has_hydrogen(atom: Atom) -> bool:
    return bool(atom.hydrogens) or atom.implicit_hydrogens > 0


class EdgeShoulder(Fragment):
    _subs = Substructure()
    _kw = dict(
        atomic_number={6, 7},
        has_hydrogen=has_hydrogen,
    )

    _subs.add_atom(QueryAtom(**_kw))
    _subs.add_atom(QueryAtom(**_kw))
    _subs.add_bond(0, 1)

    searcher = Searcher(_subs)

    def __init__(self, mol, action_points: tuple[int, int]):
        super().__init__(
            mol=mol,
            searcher=self.searcher,
            action_points=action_points,
            action_func=shoulder_bond_action
        )

class HydrogenGraft(Fragment):
    _subs = Substructure()
    _subs.add_atom(QueryAtom(
        atomic_numbers={6, 7, 8},
        has_hydrogen=has_hydrogen
    ))

    searcher = Searcher(_subs)
    def __init__(self, mol, action_points: tuple[int]):
        super().__init__(
            mol=mol,
            searcher=self.searcher,
            action_points=action_points,
            action_func=atom_link_atom_action
        )

class AlkylGraft(HydrogenGraft):
    def __init__(
            self,
            linker_length: Union[int, Iterable[int]],
            mode: Literal['random', 'permutations'] = 'Random',
            seed: Optional[int] = None,
            sample_weights: Optional[Iterable[float]] = None,
    ):
        if isinstance(linker_length, int):
            linker_length = [linker_length]
        elif isinstance(linker_length, tuple) and len(linker_length) == 2:
            linker_length = list(range(*linker_length))
        elif isinstance(linker_length, Iterable):
            linker_length = list(linker_length)
        else:
            raise TypeError('linker_length must be int or iterable of ints')

        self.linker_length = linker_length
        self.mode = mode
        self.seed = seed
        self.sample_weights = sample_weights

        if self.seed is not None:
            random.seed(self.seed)

        super().__init__(
            mol=read_mol('C'*linker_length[0]),
            action_points=(1,)
        )

    def graft(self, mol: Molecule) -> dict[str, Molecule]:
        if self.mode == 'random':
            self.frag = read_mol('C' * random.choices(self.linker_length, weights=self.sample_weights)[0])
            return super().graft(mol)
        elif self.mode == 'permutations':
            results = {}
            for ll in self.linker_length:
                self.frag = read_mol('C'*ll)
                results.update(super().graft(mol))
            return results
        else:
            raise NotImplementedError


class BondAdding(Fragment):
    _subs = Substructure()
    _kw = dict(
        atomic_number={6, 7, 8},
        has_hydrogen=has_hydrogen,
    )

    _subs.add_atom(QueryAtom(**_kw))
    _subs.add_atom(QueryAtom(**_kw))
    _subs.add_bond(0, 1)

    searcher = Searcher(_subs)
    def __init__(self, mol):
        super().__init__(
            mol=mol,
            searcher=self.searcher,
            action_points=[],
            action_func=bond_order_add
        )

class AtomReplace(Fragment):
    _subs = Substructure()
    _kw = dict()

if __name__ == '__main__':
    # frag = EdgeShoulder(read_mol('c1ccccc1'), (0, 1))
    # _mol = read_mol('c1ccccc1')
    #
    # # res = frag.graft(_mol)
    m = read_mol('CCC')
    print(m.atoms[0].implicit_hydrogens)
    m.atoms[1].symbol = 'N'
    print(m.smiles)
    print(m.atoms[1].implicit_hydrogens)
    m.atoms[1].calc_implicit_hydrogens()
    print(m.atoms[1].implicit_hydrogens)

