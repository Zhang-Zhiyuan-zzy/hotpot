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
import pickle
import traceback
from queue import Empty
from importlib.resources import files

from tqdm import tqdm

from hotpot.cheminfo.operations.io import read_mol
from hotpot.cheminfo.structure.atom import Atom
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.operations.search import Searcher, Substructure, QueryAtom
from .fragment import Fragment
from .action_func import (
    shoulder_bond_action,
    atom_link_atom_action,
    bond_order_add,
    atom_replace,
    ring_wedge,
    _assembly_hydrogen_total,
)

__all__ = [
    "EdgeShoulder",
    "AtomLink",
    "AlkylGraft",
    "BondAdding",
    "AtomReplace",
    "RingWedge",
    "AssembleFactory"
]

############## Definition of match function for Searcher ###############
def has_hydrogen(atom: Atom) -> bool:
    return bool(atom.hydrogens) or atom.implicit_hydrogens > 0

def max_heavy_bond_order(atom: Atom, _max_value: int) -> bool:
    return atom.sum_heavy_cov_orders <= _max_value


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
        atomic_number={6, 7},
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
    _subs = Substructure()
    _subs.add_atom(QueryAtom(
        atomic_number={6, 7, 8},
        has_hydrogen=has_hydrogen
    ))

    searcher = Searcher(_subs)
    def __init__(self, mol: Molecule):
        super().__init__(mol, action_points=(0,))

    @staticmethod
    def create_alkyl_collection(alkyl_length: Iterable[int]):
        return alkyl_generator(alkyl_length)

def _rooted_carbon_tree(atom, parent=None):
    return tuple(sorted(
        _rooted_carbon_tree(neighbour, atom)
        for neighbour in atom.neighbours if neighbour is not parent
    ))


def alkyl_generator(lengths: Iterable[int]):
    lengths = list(lengths)
    carbon = Molecule()
    carbon.create_atom(atomic_number=6)
    carbon.calc_implicit_hydrogens()
    alkyl = {1: {_rooted_carbon_tree(carbon.atoms[0]): carbon}}
    for length in range(1, max(lengths)):
        following = {}
        for mol in alkyl[length].values():
            for atom in mol.atoms:
                if len(atom.neighbours) < 4:
                    clone = mol.copy()
                    clone.atoms[atom.idx].add_atom(6)
                    clone.calc_implicit_hydrogens()
                    following[_rooted_carbon_tree(clone.atoms[0])] = clone
        alkyl[length + 1] = following

    results = defaultdict(list)
    for length in lengths:
        results[length].extend(AlkylGraft(mol) for mol in alkyl[length].values())
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
        'S': lambda a: max_heavy_bond_order(a, _max_value=2),
    }
    def _create_searcher(self, ele: str):
        if ele not in self._symbol_to_heavy_cov_bond_order:
            raise ValueError(f'The element {ele} is not supported, choose from {list(self._symbol_to_heavy_cov_bond_order.keys())}')

        qa = QueryAtom(
            atomic_number={6, 7, 8},
            match_sum_heavy_bo=self._symbol_to_heavy_cov_bond_order[ele],
            not_with_sp=lambda a: not any(na.atomic_number in {15, 16} for na in a.neighbours)
        )
        sub = Substructure()
        sub.add_atom(qa)

        return Searcher(sub)

    def __init__(self, element: str):
        if element not in self._symbol_to_heavy_cov_bond_order:
            raise ValueError(f'The element {element} is not supported,'
                             f'choose from {list(self._symbol_to_heavy_cov_bond_order.keys())}')
        super().__init__(
            mol=read_mol(element),
            searcher=self._create_searcher(element),
            action_points=[],
            action_func=atom_replace
        )

def not_amidoxime_nito(atom: Atom):
    if atom.atomic_number != 7:
        return True

    neighbours = atom.neighbours
    if len(neighbours) not in {1, 3}:
        return False

    if any(na.atomic_number not in {1, 6} for na in neighbours):
        return False

    alpha_carbon = [na for na in neighbours if na.atomic_number == 6]
    if len(alpha_carbon) != 1:
        return False

    alpha_carbon = alpha_carbon[0]

class RingWedge(Fragment):
    _sub = Substructure()
    _kw = dict(
        atomic_number={5, 6, 7, 14, 15},
        two_hydrogen=lambda a: _assembly_hydrogen_total(a) >= 2,
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
        self._len = (len(self.list_smiles) + self.batch_size - 1) // self.batch_size
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
        return self

    def refresh(self):
        self._idx = 0

class AssembleFactory:
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

        if catch_path is not None and not osp.isdir(osp.dirname(catch_path) or '.'):
            raise IOError(f'The directory {osp.dirname(catch_path)} is not exist')
        self.catch_path = catch_path

    def get_desc(self, epoch, results):
        if isinstance(self.max_step, int):
            return f"Make Molecule({len(results)}/{self.max_running}) in {epoch} Epoch"
        else:
            return f"Make Molecule({len(results)}) in {epoch} Epoch"

    @staticmethod
    def _make_batch(list_smi, list_assembler, q: mp.Queue) -> set[str]:
        try:
            results = set(list_smi)
            for smi, assembler in product(list_smi, list_assembler):
                results.update(assembler.graft(read_mol(smi, fmt='smi')))
        except BaseException as error:
            original_traceback = traceback.format_exc()
            try:
                payload = pickle.dumps((error, error.__cause__, error.__context__, error.__suppress_context__))
            except BaseException as transport_error:
                q.put(('transport_error', str(transport_error), original_traceback))
            else:
                q.put(('error', payload, original_traceback))
        else:
            q.put(('ok', results, None))
        finally:
            q.close()
            q.join_thread()

    def _save_results(self, results):
        if self.catch_path is not None:
            with open(self.catch_path, 'w') as writer:
                writer.write('\n'.join(results))

    def _save_failure(self, results, error):
        try:
            self._save_results(results)
        except OSError as save_error:
            error.__dict__.setdefault('__notes__', []).append(
                f'Could not persist completed assembly results: {save_error}'
            )

    @staticmethod
    def _close_worker(process, queue):
        try:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
        finally:
            queue.close()
            queue.join_thread()

    def make(self, mol_iter: Iterable[Molecule]):
        results = set(m.smiles for m in mol_iter)
        last_save_num = len(results)
        stop_generation = False
        try:
            for epoch in range(self.max_step):
                list_smi = list(results)
                total = len(list_smi) * len(self.assembler)
                with tqdm(desc=self.get_desc(epoch, results), total=total) as p_bar:
                    for smi, assembler in product(list_smi, self.assembler):
                        results.update(assembler.graft(read_mol(smi, fmt='smi')))
                        p_bar.set_description(self.get_desc(epoch, results))
                        p_bar.update()
                        if self.max_running is not None and len(results) > self.max_running:
                            stop_generation = True
                            break
                        if (
                            isinstance(self.save_per_step, int) and
                            self.catch_path is not None and
                            len(results) - last_save_num >= self.save_per_step
                        ):
                            self._save_results(results)
                            last_save_num = len(results)
                self._save_results(results)
                if stop_generation:
                    break
        except BaseException as error:
            self._save_failure(results, error)
            raise
        return results

    def mp_make(
            self,
            mol_iter: Iterable[Union[Molecule, str]],
            nproc: Optional[int] = None,
            timeout: int = 1800,
            batch_size: int = 10,
    ):
        """Expand new frontiers in owned workers; errors/timeouts fail visibly."""
        if nproc is None:
            nproc = os.cpu_count()
        results = {mol.smiles if isinstance(mol, Molecule) else mol for mol in mol_iter}
        current_results = set()
        last_save_num = len(results)
        completed_batches = 0
        completed_pairs = 0
        try:
            for epoch in range(self.max_step):
                batches = MolBatch(results - current_results, batch_size)
                iterator = iter(enumerate(batches, 1))
                current_results = set(results)
                processes = {}
                exhausted = False
                stop_generation = False
                with tqdm(desc=self.get_desc(epoch, results), total=len(batches), ncols=100) as p_bar:
                    try:
                        while not exhausted or processes:
                            if self.max_running is not None and len(results) > self.max_running:
                                stop_generation = True
                                break
                            while not exhausted and len(processes) < nproc:
                                try:
                                    batch_index, smiles = next(iterator)
                                except StopIteration:
                                    exhausted = True
                                    break
                                queue = mp.Queue()
                                process = mp.Process(
                                    target=self._make_batch,
                                    args=(smiles, self.assembler, queue),
                                )
                                processes[process] = (queue, time.monotonic(), batch_index, smiles)
                                process.start()
                            for process, (queue, started, batch_index, smiles) in list(processes.items()):
                                elapsed = time.monotonic() - started
                                subject = f'epoch={epoch}, batch={batch_index}, inputs={smiles!r}'
                                try:
                                    kind, payload, remote_traceback = queue.get_nowait()
                                except Empty:
                                    if process.exitcode is not None:
                                        # Exit follows the child's queue flush; drain once more
                                        # if completion arrived after the first empty poll.
                                        try:
                                            kind, payload, remote_traceback = queue.get_nowait()
                                        except Empty:
                                            raise RuntimeError(
                                                f'assembly.worker_exit: {subject}, exitcode={process.exitcode}; '
                                                'worker exited without completion'
                                            ) from None
                                    else:
                                        if timeout and elapsed >= timeout:
                                            raise TimeoutError(
                                                f'assembly.batch_timeout: {subject}, timeout={timeout}s, '
                                                f'elapsed={elapsed:.6f}s, completed_batches={completed_batches}, '
                                                f'completed_input_assembler_pairs={completed_pairs}, '
                                                f'known_results={len(results)}'
                                            )
                                        continue
                                if kind == 'error':
                                    try:
                                        error, cause, context, suppress_context = pickle.loads(payload)
                                        error.__cause__ = cause
                                        error.__context__ = context
                                        error.__suppress_context__ = suppress_context
                                    except Exception as transport_error:
                                        raise RuntimeError(
                                            f'assembly.error_transport: {subject}\n{remote_traceback}'
                                        ) from transport_error
                                    error.__dict__.setdefault('__notes__', []).append(
                                        f'Assembly worker {subject}\n{remote_traceback}'
                                    )
                                    raise error
                                if kind == 'transport_error':
                                    raise RuntimeError(
                                        f'assembly.error_transport: {subject}: {payload}\n{remote_traceback}'
                                    )
                                results.update(payload)
                                completed_batches += 1
                                completed_pairs += len(smiles) * len(self.assembler)
                                process.join(timeout=5)
                                if process.exitcode != 0:
                                    raise RuntimeError(
                                        f'assembly.worker_exit: {subject}, exitcode={process.exitcode}; '
                                        'worker did not exit normally after completion'
                                    )
                                self._close_worker(process, queue)
                                del processes[process]
                                p_bar.set_description(self.get_desc(epoch, results))
                                p_bar.update()
                            if (
                                self.catch_path is not None and
                                isinstance(self.save_per_step, int) and
                                len(results) - last_save_num >= self.save_per_step
                            ):
                                self._save_results(results)
                                last_save_num = len(results)
                            if processes:
                                time.sleep(0.01)
                    finally:
                        for process, (queue, _, _, _) in processes.items():
                            self._close_worker(process, queue)
                self._save_results(results)
                if stop_generation:
                    break
        except BaseException as error:
            self._save_failure(results, error)
            raise
        return results

    @classmethod
    def load_default_assembler(cls):
        """Load the packaged canonical assembly definitions."""
        with files(__package__).joinpath('data/fragment_template.json').open() as reader:
            return cls.load_assembler_contents(json.load(reader))

    @classmethod
    def init_with_default_assembler(
            cls,
            assembler: Optional[Iterable[Fragment]] = None,
            iter_step: int = 5,
            mode: Literal['random', 'permutations'] = 'random',
            seed: Optional[int] = None,
            sample_weights: Optional[Iterable[float]] = None,
            max_results: int = 3000000,
            save_per_step: Optional[int] = 100000,
            catch_path: Optional[Union[str, os.PathLike]] = None
    ) -> 'AssembleFactory':
        assembler = [] if assembler is None else list(assembler)
        assembler.extend(cls.load_default_assembler())
        return AssembleFactory(
            assembler=assembler,
            iter_step=iter_step,
            mode=mode,
            seed=seed,
            sample_weights=sample_weights,
            max_running=max_results,
            save_per_step=save_per_step,
            catch_path=catch_path
        )

    @classmethod
    def load_assembler_file(cls, f: Union[str, os.PathLike]) -> list[Fragment]:
        with open(f) as reader:
            return cls.load_assembler_contents(json.load(reader))

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
        assemblers = []
        smiles = definition.get('ExStrct', None)
        for point in definition['points']:
            assert isinstance(point, list)
            assert len(point) == 1
            assert all(isinstance(p, int) for p in point)

            assembler = RingWedge(read_mol(definition['smiles']), action_points=tuple(point))
            if isinstance(smiles, str):
                assembler.add_exclude_from_smi(smiles)

            assemblers.append(assembler)

        return assemblers

    @staticmethod
    def _define_edge_shoulder(definition: dict):
        assemblers = []
        smiles = definition.get('ExStrct', None)
        for point in definition['points']:
            assert isinstance(point, list)
            assert len(point) == 2
            assert all(isinstance(p, int) for p in point)

            assembler = EdgeShoulder(read_mol(definition['smiles']), action_points=tuple(point))
            if isinstance(smiles, str):
                assembler.add_exclude_from_smi(smiles)

            assemblers.append(assembler)

        return assemblers

    @staticmethod
    def _define_atom_link(definition: dict):
        assemblers = []
        for point in definition['points']:
            assert isinstance(point, list)
            assert len(point) == 1
            assert isinstance(point[0], int)
            assemblers.append(AtomLink(read_mol(definition['smiles']), tuple(point)))

        if smiles := definition.get('ExStrct', None):
            for assembler in assemblers:
                assembler.add_exclude_from_smi(smiles)

        return assemblers

    @staticmethod
    def _define_bond_adding(definition: dict):
        if smiles := definition.get('ExStrct', None):
            assembler = BondAdding()
            assembler.add_exclude_from_smi(smiles)
            return [assembler]
        else:
            return [BondAdding()]

    @staticmethod
    def _define_atom_replace(definition: dict):
        if smiles := definition.get('ExStrct', None):
            assemblers = []
            for ele in definition['elements']:
                assembler = AtomReplace(ele)
                assembler.add_exclude_from_smi(smiles)
                assemblers.append(assembler)
            return assemblers
        else:
            return [AtomReplace(ele) for ele in definition['elements']]

    @staticmethod
    def _define_alkyl(definition: dict):
        alkyl_dict = AlkylGraft.create_alkyl_collection(definition['link_length'])
        assemblers = sum(map(lambda v: list(v), alkyl_dict.values()), start=[])

        if smiles := definition.get('ExStrct', None):
            for assembler in assemblers:
                assembler.add_exclude_from_smi(smiles)

        return assemblers



if __name__ == '__main__':
    phen_smi = 'n1c(C(=O)N)ccc2c1c3c(cc2)cccn3'
    rw_assembler = RingWedge(
        read_mol('C1CCCC1'),
        action_points=(0,)
    )

    res = rw_assembler.graft(read_mol(phen_smi, fmt='smi'))
