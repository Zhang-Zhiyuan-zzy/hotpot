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
import os
import os.path as osp
import json
import time
from collections import defaultdict
from typing import Iterable, Literal, Optional, Union
from itertools import product
import multiprocessing as mp

from tqdm import tqdm

import hotpot.cheminfo as ci
from hotpot.utils import fmt_print
from hotpot.cheminfo.core import Atom, Molecule
from hotpot.cheminfo.search import Searcher, Substructure, QueryAtom
from hotpot.cheminfo.mol_assemble.fragment import Fragment
from hotpot.cheminfo.mol_assemble.action_func import (
    shoulder_bond_action,
    atom_link_atom_action,
    bond_order_add,
    atom_replace,
    ring_wedge
)

__all__ = [
    "EdgeShoulder",
    "AtomLink",
    "AlkylGraft",
    "BondAdding",
    "AtomReplace",
    "AssembleFactory"
]

############## Definition of match function for Searcher ###############
def has_hydrogen(atom: Atom) -> bool:
    return bool(atom.hydrogens) or atom.implicit_hydrogens > 0

def max_heavy_bond_order(atom: Atom, _max_value: int) -> bool:
    return atom.sum_heavy_cov_orders < _max_value


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

class AtomLink(Fragment):
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

class AlkylGraft(AtomLink):
    def __init__(self, mol: Molecule):
        super().__init__(mol, action_points=(0,))

    @staticmethod
    def create_alkyl_collection(alkyl_length: Iterable[int]):
        return alkyl_generator(alkyl_length)

def alkyl_generator(lengths: Iterable[int]):
    lengths = list(lengths)
    max_length = max(lengths)
    alkyl = defaultdict(dict)
    alkyl[1]['CC'] = ci.read_mol('CC')
    for i in range(1, max_length):
        for mol in alkyl[i].values():
            open_site_atom = [a.idx for a in mol.atoms[1:] if len(a.neighbours) <= 4]
            for c in open_site_atom:
                clone = mol.copy()
                clone.atoms[c].add_atom(6)
                alkyl[i+1][clone.smiles] = clone

    results = defaultdict(list)
    for i in lengths:
        for mol in alkyl[i].values():
            mol.remove_atom(0)
            results[i].append(
                AlkylGraft(mol=mol)
            )

    return results


class BondAdding(Fragment):
    _subs = Substructure()
    _kw = dict(
        atomic_number={6, 7, 8},
        has_hydrogen=has_hydrogen,
        is_aromatic={False}
    )

    _subs.add_atom(QueryAtom(**_kw))
    _subs.add_atom(QueryAtom(**_kw))
    _subs.add_bond(0, 1)

    searcher = Searcher(_subs)
    def __init__(self):
        super().__init__(
            mol=Molecule(),
            searcher=self.searcher,
            action_points=[],
            action_func=bond_order_add
        )

class AtomReplace(Fragment):
    _symbol_to_heavy_cov_bond_order = {
        'N': lambda a: max_heavy_bond_order(a, _max_value=3),
        'O': lambda a: max_heavy_bond_order(a, _max_value=2),
        'Si': lambda a: max_heavy_bond_order(a, _max_value=4),
    }
    def _create_searcher(self, ele: str):
        if ele not in self._symbol_to_heavy_cov_bond_order:
            raise ValueError(f'The element {ele} is not supported, choose from {list(self._symbol_to_heavy_cov_bond_order.keys())}')

        qa = QueryAtom(
            atomic_number={6, 7, 8},
            match_sum_heavy_bo=self._symbol_to_heavy_cov_bond_order[ele],
        )
        sub = Substructure()
        sub.add_atom(qa)

        return Searcher(sub)

    def __init__(self, element: str):
        if element not in self._symbol_to_heavy_cov_bond_order:
            raise ValueError(f'The element {element} is not supported,'
                             f'choose from {list(self._symbol_to_heavy_cov_bond_order.keys())}')
        super().__init__(
            mol=ci.read_mol(element),
            searcher=self._create_searcher(element),
            action_points=[],
            action_func=atom_replace
        )

class RingWedge(Fragment):
    _sub = Substructure()
    _kw = dict(
        atomic_number={5, 6, 7, 14, 15},
        two_hydrogen=lambda a: a.implicit_hydrogens >= 2,  # at least two hydrogens
        is_end_heavy_atom=lambda a: len(a.heavy_neighbours) == 1
    )
    _sub.add_atom(QueryAtom(**_kw))
    searcher = Searcher(_sub)

    def __init__(self, mol: Molecule, action_points: tuple[int]):
        super().__init__(
            mol=mol,
            searcher=self.searcher,
            action_points=action_points,
            action_func=ring_wedge
        )


class MolBatch:
    def __init__(self, iter_smiles: Iterable[Molecule], batch_size: int = 100):
        self.list_smiles = list(iter_smiles)
        self.batch_size = batch_size
        self._len = (len(self.list_smiles) // self.batch_size) + 1
        self._idx = 0

    def __len__(self):
        return self._len

    def __next__(self):
        if self._idx < self._len:
            self._idx += 1
            return self.list_smiles[(self._idx-1)*self.batch_size:self._idx*self.batch_size]
        else:
            raise StopIteration

    def __iter__(self):
        for i in range(self._len):
            yield self.list_smiles[self._idx*self.batch_size:(self._idx+1)*self.batch_size]

    def refresh(self):
        self._idx = 0

class AssembleFactory:
    methods = {
        "EdgeShoulder": EdgeShoulder,
        "AtomLink": AtomLink,
        "AtomReplace": AtomReplace,
        "BondAdding": BondAdding,
    }

    def __init__(
            self,
            assembler: Iterable[Fragment],
            iter_step: int = 5,
            mode: Literal['random', 'permutations'] = 'random',
            seed: Optional[int] = None,
            sample_weights: Optional[Iterable[float]] = None,
            max_running: Optional[int] = 3000000,
            save_per_step: Optional[int] = 100000,
            catch_path: Optional[Union[str, os.PathLike]] = None
    ):
        self.assembler = list(assembler)
        self.max_step = iter_step
        self.mode = mode
        self.seed = seed
        self.sample_weights = sample_weights
        self.max_running = max_running
        self.save_per_step = save_per_step

        if catch_path is not None and not osp.exists(osp.dirname(catch_path)):
            raise IOError(f'The directory {osp.dirname(catch_path)} is not exist')
        self.catch_path = catch_path

    def get_desc(self, epoch, results):
        if isinstance(self.max_step, int):
            return f"Make Molecule({len(results)}/{self.max_running}) in {epoch} Epoch"
        else:
            return f"Make Molecule({len(results)}) in {epoch} Epoch"

    @staticmethod
    def _make_one(smi, assembler, q: mp.Queue) -> set[str]:
        q.put(set(assembler.graft(ci.read_mol(smi, fmt='smi'))))

    @staticmethod
    def _make_batch(list_smi, list_assembler, q: mp.Queue) -> set[str]:
        results = set(list_smi)
        for smi, assembler in product(list_smi, list_assembler):
            results.update(assembler.graft(ci.read_mol(smi, fmt='smi')))
        q.put(results)

    def make(self, mol_iter: Iterable[Molecule]):
        results = set(m.smiles for m in mol_iter)
        last_save_num = len(results)
        stop_generation = False

        for epoch in range(self.max_step):
            list_smi = list(results)
            total = len(list_smi) * len(self.assembler)
            p_bar = tqdm(desc=self.get_desc(epoch, results), total=total)
            for smi, assembler in product(list_smi, self.assembler):
                results.update(assembler.graft(ci.read_mol(smi, fmt='smi')))
                p_bar.desc = self.get_desc(epoch, results)
                p_bar.update()

                if len(results) > self.max_running:
                    stop_generation = True
                    break

                if (
                        isinstance(self.save_per_step, int) and
                        self.catch_path is not None and
                        len(results) - last_save_num > self.save_per_step
                ):
                    with open(self.catch_path, 'w') as writer:
                        writer.write('\n'.join(results))

            if self.catch_path is not None:
                with open(self.catch_path, 'w') as writer:
                    writer.write('\n'.join(results))

            if stop_generation:
                break

        return results

    def mp_make(
            self,
            mol_iter: Iterable[Molecule],
            nproc: Optional[int] = None,
            timeout: int = 100,
            batch_size: int = 10,
    ):
        """ Running the Assembler.make in a multiprocessing context. """
        if nproc is None:
            nproc = os.cpu_count()

        results = set(m.smiles for m in mol_iter)
        last_save_num = len(results)
        for epoch in range(self.max_step):
            list_smi = list(results)
            _iterator = MolBatch(list_smi, batch_size)

            total = len(_iterator)
            p_bar = tqdm(desc=self.get_desc(epoch, results), total=total)

            processes = {}
            time_stop = time.time()
            while True:
                # Harvest results
                if processes:
                    to_remove = []
                    for p, (q, t) in processes.items():
                        try:
                            results.update(q.get(block=False))
                            p.terminate()
                            to_remove.append(p)

                        except mp.queues.Empty:
                            if timeout and time.time() - t > timeout:
                                p.terminate()
                                to_remove.append(p)

                    for p in to_remove:
                        del processes[p]
                        p_bar.set_description(self.get_desc(epoch, results))
                        p_bar.update()

                if (    # Save temporary results
                        self.catch_path is not None and
                        isinstance(self.save_per_step, int) and
                        len(results) - last_save_num > self.save_per_step
                ):
                    with open(self.catch_path, 'w') as writer:
                        writer.write('\n'.join(results))
                    last_save_num = len(results)

                # Launch new Process
                if len(processes) < nproc:
                    try:
                        smi_iter = next(_iterator)

                        q = mp.Queue()
                        p = mp.Process(target=self._make_batch, args=(smi_iter, self.assembler, q))
                        p.start()

                        processes[p] = (q, time.time())

                    except StopIteration:
                        if not processes:
                            fmt_print.bold_magenta(f'Stop MolAssemble in {epoch} Epoch!!')
                            break
                        elif time.time() - time_stop > 10:
                            fmt_print.bold_magenta(f'StopIteration with {len(processes)} running processes!!')
                            time_stop = time.time()

            # Save results after a whole Epoch
            if self.catch_path is not None:
                with open(self.catch_path, 'w') as writer:
                    writer.write('\n'.join(results))

        return results

    @classmethod
    def load_default_assembler(
            cls,
            assembler: Optional[Iterable[Fragment]] = None,
            iter_step: int = 5,
            mode: Literal['random', 'permutations'] = 'random',
            seed: Optional[int] = None,
            sample_weights: Optional[Iterable[float]] = None,
            max_running: int = 3000000,
            save_per_step: Optional[int] = 100000,
            catch_path: Optional[Union[str, os.PathLike]] = None
    ) -> 'AssembleFactory':

        assembler = [] if assembler is None else list(assembler)
        assembler.extend(cls.load_assembler_file(osp.join(osp.dirname(osp.abspath(__file__)), "FragTemplete.json")))
        return AssembleFactory(
            assembler=assembler,
            iter_step=iter_step,
            mode=mode,
            seed=seed,
            sample_weights=sample_weights,
            max_running=max_running,
            save_per_step=save_per_step,
            catch_path=catch_path
        )

    @classmethod
    def load_assembler_file(cls, f: Union[str, os.PathLike]) -> list[Fragment]:
        return cls.load_assembler_contents(json.load(open(f)))

    @classmethod
    def load_assembler_contents(cls, contents: list[dict]) -> list[Fragment]:
        assembler = []
        for defined_dict in contents:
            if defined_dict['method'] == 'EdgeShoulder':
                assembler.extend(cls._define_edge_shoulder(defined_dict))
            elif defined_dict['method'] == 'AtomLink':
                assembler.extend(cls._define_atom_link(defined_dict))
            elif defined_dict['method'] == 'AtomReplace':
                assembler.extend(cls._define_atom_replace(defined_dict))
            elif defined_dict['method'] == 'BondAdding':
                assembler.extend(cls._define_bond_adding(defined_dict))
            elif defined_dict['method'] == 'AlkylGraft':
                assembler.extend(cls._define_alkyl(defined_dict))
            elif defined_dict['method'] == 'RingWedge':
                assembler.extend(cls._define_ring_wedge(defined_dict))
            else:
                raise NotImplementedError(f'Method {defined_dict["method"]} is not supported')
        return assembler

    @staticmethod
    def _define_ring_wedge(definition: dict):
        assembler = []
        for point in definition['points']:
            assert isinstance(point, list)
            assert len(point) == 1
            assert all(isinstance(p, int) for p in point)
            assembler.append(RingWedge(ci.read_mol(definition['smiles']), action_points=tuple(point)))

        return assembler

    @staticmethod
    def _define_edge_shoulder(definition: dict):
        assembler = []
        for point in definition['points']:
            assert isinstance(point, list)
            assert len(point) == 2
            assert all(isinstance(p, int) for p in point)
            assembler.append(EdgeShoulder(ci.read_mol(definition['smiles']), action_points=tuple(point)))

        return assembler

    @staticmethod
    def _define_atom_link(definition: dict):
        assembler = []
        for point in definition['points']:
            assert isinstance(point, list)
            assert len(point) == 1
            assert isinstance(point[0], int)
            assembler.append(AtomLink(ci.read_mol(definition['smiles']), tuple(point)))

        return assembler

    @staticmethod
    def _define_bond_adding(definition: dict):
        return [BondAdding()]

    @staticmethod
    def _define_atom_replace(definition: dict):
        return [AtomReplace(ele) for ele in definition['elements']]

    @staticmethod
    def _define_alkyl(definition: dict):
        alkyl_dict = AlkylGraft.create_alkyl_collection(definition['link_length'])
        return sum(map(lambda v: list(v), alkyl_dict.values()), start=[])



if __name__ == '__main__':
    phen_smi = 'n1c(C(=O)N)ccc2c1c3c(cc2)cccn3'
    rw_assembler = RingWedge(
        ci.read_mol('C1CCCC1'),
        action_points=(0,)
    )

    res = rw_assembler.graft(ci.read_mol(phen_smi, fmt='smi'))

