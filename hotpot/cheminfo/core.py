"""
python v3.9.0
@Project: hotpot
@File   : core__
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 18:30
"""
import logging
import re
import time
from typing import Union, Literal, Iterable, Optional
from copy import copy
from collections import Counter
from itertools import combinations, product

import numpy as np
import networkx as nx
from openbabel import pybel as pb, openbabel as ob
import periodictable

from hotpot.utils import types
from hotpot.cheminfo.obconvert import write_by_pybel, mol2obmol
from .rdconvert import to_rdmol
from . import graph, forcefields as ff, _io
from . import geometry


class Molecule:
    def __init__(self):
        self._atoms = []
        self._bonds = []
        self._conformers = Conformers()
        self._conformers_index = 0

        self._angles = []
        self._torsions = []
        self._rings = []
        self._graph = nx.Graph()

        self._broken_metal_bonds = []

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            # Retrieve attributes from conformers
            return self._conformers.index_attr(item, self._conformers_index)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.formula})"

    def __add__(self, other):
        clone = copy(self)
        clone.add_component(other)
        return clone

    def __copy__(self):
        clone = Molecule()
        for atom in self._atoms:
            clone._create_atom(**atom.attr_dict)
        for bond in self._bonds:
            clone._add_bond(bond.a1idx, bond.a2idx, **bond.attr_dict)

        clone._update_graph()

        return clone

    def _retrieve_torsions(self):
        torsion = []
        t1 = time.time()
        for bond in self.bonds:
            a1_neigh = list(bond.atom1.neighbours)
            a2_neigh = list(bond.atom2.neighbours)
            a1_neigh.remove(bond.atom2)
            a2_neigh.remove(bond.atom1)

            for a, d in product(a1_neigh, a2_neigh):
                torsion.append(Torsion(a, bond.atom1, bond.atom2, d))
        t2 = time.time()
        print(f'Torsion calculation took {t2-t1} seconds')

        return torsion

    def simple_paths(self, source=None, target=None, cutoff=None):
        if source is None and target is None:
            st = combinations(self.graph.nodes, 2)
        elif isinstance(source, int) and isinstance(target, int):
            st = [(source, target)]
        elif isinstance(source, int):
            st = [(source, t) for t in self.graph.nodes if t!=source]
        else:
            st = [(s, target) for s in self.graph.nodes if s!=target]

        # if isinstance(cutoff, int):
        #     st = [(s, t) for s, t in st if len(nx.shortest_path(self.graph, s, t)) <= cutoff]

        return [p for s, t in st for p in nx.all_simple_paths(self.graph, s, t, cutoff=cutoff)]

    def _set_coordinates(self, coords: types.ArrayLike):
        coords = np.array(coords)
        if coords.shape != (len(self._atoms), 3):
            raise ValueError(f"The shape of coordinates should be {(len(self._atoms), 3)}, but got {coords.shape}")

        for atom, coord in zip(self._atoms, coords):
            atom.coordinates = coord

    def _update_graph(self):
        self._graph = nx.Graph()
        self._graph.add_edges_from(self._edge_with_attrs())
        self._graph.add_nodes_from(self._node_with_attrs())

        # clear the older AtomSeq
        self._angles = []
        self._torsions = []
        self._rings = []

    def _add_atom(self, atom):
        self._atoms.append(atom)
        atom.mol = self

        return atom

    def _add_bond(self, atom1: Union[int, "Atom"], atom2: Union[int, "Atom"], bond_order=1., **kwargs):

        atom1 = atom1 if isinstance(atom1, Atom) else self._atoms[atom1]
        atom2 = atom2 if isinstance(atom2, Atom) else self._atoms[atom2]

        if atom1.mol is not self or atom2.mol is not self:
            raise ValueError("at least one of atom1 and atom2 not on the molecule!")

        if (atom1.idx, atom2.idx) in self.graph.edges:
            raise ValueError(f"the bond between {atom1.idx} and {atom2.idx} is already exist!!")

        kwargs['bond_order'] = bond_order
        bond = Bond(atom1, atom2, **kwargs)
        self._bonds.append(bond)

        return bond

    def add_atom(self, atom):
        atom = self._add_atom(atom)
        self._update_graph()
        return atom

    def add_bond(self, atom1: Union[int, "Atom"], atom2: Union[int, "Atom"], bond_order=1., **kwargs):
        bond = self._add_bond(atom1, atom2, bond_order, **kwargs)
        self._update_graph()
        return bond

    def add_component(self, component: "Molecule"):
        component = copy(component)
        for atom in component.atoms:
            self._add_atom(atom)
        self._bonds.extend(component.bonds)
        del component
        self._update_graph()

    def add_hydrogens(
            self,
            recalc_implicit_hydrogens=False,
            remove_excess: bool = False
    ):
        """"""
        if recalc_implicit_hydrogens:
            self.calc_implicit_hydrogens()

        # Add hydrogens
        modified = False
        for atom in self.atoms:
            if not (atom.is_hydrogen or atom.is_metal):
                add_or_rm, hydrogens = atom._add_hydrogens(remove_excess=remove_excess)
                if add_or_rm:
                    modified = True

        if modified:
            self._update_graph()

        # Remove hydrogens on the atom beyond the maxiunm

    def break_metal_ligand_bonds(self) -> None:
        """ break all bonds link with metals """
        metal_bonds = [b for b in self.bonds if b.is_metal_ligand_bond]
        self._broken_metal_bonds.extend(metal_bonds)

        for b in metal_bonds:
            self._bonds.remove(b)

        self._update_graph()

    def clear_constraints(self) -> None:
        """ clear all set constraints """
        for atom in self.atoms:
            atom.constraint = False
        for bond in self.bonds:
            bond.constraint = False
        for angle in self.angles:
            angle.constraint = False
        for torsion in self.torsions:
            torsion.constraint = False

    def clear_metal_ligand_bonds(self) -> None:
        self.break_metal_ligand_bonds()
        self._broken_metal_bonds = []

    def recover_metal_ligand_bonds(self) -> None:
        self._bonds = list(set(self._bonds + self._broken_metal_bonds))
        self._update_graph()

    @property
    def angles(self) -> list["Angle"]:
        if not self._angles:
            self._angles = [Angle(n1, a, n2) for a in self.atoms for n1, n2 in combinations(a.neighbours, 2)]
        return copy(self._angles)

    @property
    def torsions(self) -> list["Torsion"]:
        if not self._torsions:
            self._torsions = self._retrieve_torsions()
        return copy(self._torsions)

    @property
    def atom_attr_matrix(self) -> np.ndarray:
        return np.array([a.attrs for a in self._atoms])

    @property
    def atoms(self):
        return copy(self._atoms)

    @property
    def bonds(self):
        return copy(self._bonds)

    def bond(self, a1idx: int, a2idx: int):
        return self.graph.edges[a1idx, a2idx]['bond']

    def build3d(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = 'UFF',
            steps: int = 500,
            sophisticated: bool = True,
            **kwargs
    ):
        if sophisticated and not self.is_organic:
            ff.complexes_build(self, **kwargs)

        else:
            self.add_hydrogens()
            ff.ob_build(self)
            ff.ob_optimize(self, forcefield, steps)

    @property
    def charge(self) -> int:
        return sum(a.formal_charge for a in self._atoms)

    @property
    def default_spin_mult(self) -> int:
        return (sum(a.atomic_number for a in self.atoms) - self.charge) % 2 + 1

    @property
    def conformers(self) -> "Conformers":
        return self._conformers

    def conformer_load(self, i: int):
        """ Load specific conformer to be current conformer """
        self.coordinates = self.conformers.index_attr('coordinates', i)
        self._conformers_index = i

    @property
    def conformers_number(self) -> int:
        return len(self._conformers)

    def conformer_add(
            self, coords: Optional[types.ArrayLike] = None,
            energy: Optional[Union[types.ArrayLike, float]] = None
    ):
        if coords is None:
            self._conformers.add(self.coordinates, energy)
        else:
            self._conformers.add(coords, energy)

    def conformer_clear(self):
        self._conformers.clear()

    def conformer_get(self, idx: Union[int, slice]) -> np.ndarray:
        """ Get specific conformer coordinates """
        return self._conformers[idx]

    def optimize_(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = None,
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            steps: Optional[int] = 100,
            step_size: int = 100,
            equilibrium: bool = False,
            equi_check_steps: int = 5,
            equi_max_displace: float = 1e-4,
            equi_max_energy: float = 1e-4,
            perturb_steps: Optional[int] = None,
            perturb_sigma: float = 0.5,
            save_screenshot: bool = False,
            increasing_Vdw: bool = False,
            Vdw_cutoff_start: float = 0.0,
            Vdw_cutoff_end: float = 12.5,
            print_energy: Optional[int] = None
    ):
        arguments = locals()
        del arguments["self"]
        del arguments["forcefield"]

        if forcefield is None:
            if self.has_metal:
                arguments['ff'] = 'UFF'
            else:
                arguments['ff'] = 'MMFF94s'
        else:
            arguments['ff'] = forcefield

        ff.OBFF(**arguments).optimize(self)

    def optimize(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = None,
            algorithm: Literal["steepest", "conjugate"] = "steepest",
            steps: Optional[int] = None,
            equilibrium: bool = False,
            equi_threshold: float = 1e-4,
            max_iter: int = 100,
            save_screenshot: bool = False,
            perturb_steps: Optional[int] = None,
            perturb_sigma: Optional[float] = 0.5
    ):
        if forcefield is None:
            if self.has_metal:
                forcefield = 'UFF'
            else:
                forcefield = 'MMFF94s'

        obff = ff.OBFF(
            ff=forcefield,
            algorithm=algorithm,
            steps=steps,
            equilibrium=equilibrium,
            equi_threshold=equi_threshold,
            max_iter=max_iter,
            save_screenshot=save_screenshot,
            perturb_steps=perturb_steps,
            perturb_sigma=perturb_sigma
        )
        # obff.ff.SetVDWCutOff(0.5)
        obff.optimize(self)

    def optimize_complexes(
            self,
            algorithm: Literal["steepest", "conjugate"] = "steepest",
            steps: Optional[int] = None,
            equilibrium: bool = True,
            equi_threshold: float = 1e-4,
            max_iter: int = 100,
            save_screenshot: bool = False
    ):
        """ A specific optimizing stratage for metal-ligand complexes """
        if not self.has_metal:
            print(UserWarning(
                "The `optimize_complexes()` is specified for metal-ligand complexes, \n"
                "it's much slower than `optimize()` method. For organic compounds, the \n"
                "`optimize() is more recommended."
            ))
            self.optimize('MMFF94s', algorithm, steps, equilibrium, equi_threshold, max_iter, save_screenshot)
            return


        self.refresh_atom_id()

        # Initialize optimizer
        obff = ff.OBFF(
            ff='UFF',
            algorithm=algorithm,
            steps=steps,
            equilibrium=equilibrium,
            equi_threshold=equi_threshold,
            max_iter=max_iter,
            save_screenshot=False
        )

        clone = copy(self)
        clone.break_metal_ligand_bonds()

        for component in clone.components:
            if component.is_organic:
                obff.optimize(component)
                clone.update_atoms_attrs_from_id_dict({a.id: {'coordinates': a.coordinates} for a in component.atoms})

        clone.recover_metal_ligand_bonds()
        # clone.constraint_bonds_angles()
        for a in clone.atoms:
            if not a.is_metal:
                a.constraint = True

        obff.ff = ob.OBForceField.FindType('UFF')
        obff.save_screenshot = save_screenshot
        # obff.optimize(clone)

        obff.equilibrium = True
        obff.perturb_steps = 30
        obff.perturb_sigma = 0.5
        self.coordinates = clone.coordinates
        self._conformers = clone.conformers
        obff.optimize(self)

    def complexes_build_optimize_(
            self,
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            steps: Optional[int] = 500,
            step_size: int = 100,
            equilibrium: bool = False,
            equi_check_steps: int = 5,
            equi_max_displace: float = 1e-4,
            equi_max_energy: float = 1e-4,
            perturb_steps: Optional[int] = 50,
            perturb_sigma: float = 0.5,
            save_screenshot: bool = True,
            increasing_Vdw: bool = False,
            Vdw_cutoff_start: float = 0.0,
            Vdw_cutoff_end: float = 12.5,
            print_energy: Optional[int] = 100,
            # parameter for complexes build
            build_times: int =5,
            init_opt_steps: int =500,
            second_opt_steps: int =1000,
            min_energy_opt_steps: int =3000,
            correct_hydrogens: bool = True
    ):
        arguments = locals()
        del arguments['self']

        # For organic compound
        if not self.has_metal:
            print(UserWarning(
                "The `optimize_complexes()` is specified for metal-ligand complexes, \n"
                "it's much slower than `optimize()` method. For organic compounds, the \n"
                "`optimize() is more recommended."
            ))
            arguments['ff'] = 'MMFF94s'
            self.optimize_(**arguments)
            return

        # build complex
        ff.complexes_build(
            self,
            build_times,
            init_opt_steps,
            second_opt_steps,
            min_energy_opt_steps,
            correct_hydrogens
        )

        # Initialize optimizer
        arguments['ff'] = 'UFF'
        obff = ff.OBFF_(**arguments)
        obff.ff.SetVDWCutOff(12.5)
        obff.optimize(self)

    def calc_implicit_hydrogens(self):
        for a in self.atoms:
            a.calc_implicit_hydrogens()

    @property
    def components(self) -> list['Molecule']:
        """"""
        graph = self.graph

        components = []
        for c_node_idx in nx.connected_components(graph):
            c_node_idx = tuple(c_node_idx)
            subgraph = graph.subgraph(c_node_idx)

            component = Molecule()

            for node_idx in c_node_idx:
                component._create_atom(**graph.nodes[node_idx])


            for edge_begin_idx, edge_end_index in subgraph.edges:
                component._add_bond(
                    c_node_idx.index(edge_begin_idx),
                    c_node_idx.index(edge_end_index),
                    **subgraph.edges[edge_begin_idx, edge_end_index]['bond'].attr_dict
                )

            component._update_graph()
            components.append(component)

        return components

    def constraint_bonds_angles(self, exclude_metal_bonds: bool = True):
        """ constraint all bond length and angles, this method ensure the molecule just to rotate the torsions. """
        self.update_angles()
        if exclude_metal_bonds:
            for bond in self.bonds:
                if not bond.has_metal:
                    bond.constraint = True
            for angle in self.angles:
                if not angle.has_metal:
                    angle.constraint = True

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([a.coordinates for a in self._atoms])

    @coordinates.setter
    def coordinates(self, value: np.ndarray) -> None:
        assert value.shape == (len(self.atoms), 3)
        for a, row in zip(self.atoms, value):
            a.coordinates = row

    def _create_atom(self, **kwargs):
        return Atom(self, **kwargs)

    def create_atom(self, **kwargs) -> 'Atom':
        atom = self._create_atom(**kwargs)
        self._update_graph()
        return atom

    @property
    def element_counts(self):
        return Counter([a.symbol for a in self.atoms])

    @property
    def simple_graph(self) -> nx.Graph:
        """ Return a networkx Graph without nodes and edges attrs """
        graph = nx.Graph()
        graph.add_edges_from(self.link_matrix)
        return graph

    def _node_with_attrs(self):
        return [(a.idx, {n:getattr(a, n) for n in a.attrs_enumerator}) for a in self._atoms]

    def _edge_with_attrs(self):
        # attrs = ('idx',) + Bond._attrs_enumerator
        return [(b.a1idx, b.a2idx, {'bond': b}) for b in self._bonds]

    @property
    def graph(self):
        """ Return networkx graph with nodes and edges attrs """
        return self._graph

    def graph_spectral(self, norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'):
        """ Return graph spectral matrix """
        clone = copy(self)
        clone.add_hydrogens()
        adj = graph.linkmat2adj(len(clone.atoms), clone.link_matrix)
        return graph.GraphSpectrum.from_adj_atoms(adj, np.array([a.atomic_number for a in clone.atoms]), norm=norm)

    @property
    def formula(self) -> str:
        formula = ''
        for ele, count in self.element_counts.items():
            formula += f'{ele}{count}'

        return formula

    @property
    def has_metal(self) -> bool:
        return any(a.is_metal for a in self.atoms)

    @property
    def has_bond_ring_intersection(self) -> bool:
        return any(r.is_bond_intersect_the_ring(b) for r, b in product(self.rings_small, self.bonds))

    @property
    def heavy_atoms(self) -> list["Atom"]:
        return [a for a in self._atoms if a.atomic_number != 1]

    @property
    def is_organic(self) -> bool:
        return all(not a.is_metal for a in self.atoms)

    @property
    def link_matrix(self) -> np.ndarray:
        return np.array([[b.atom1.idx, b.atom2.idx] for b in self.bonds], dtype=int)

    @property
    def metals(self) -> list['Atom']:
        return [a for a in self._atoms if a.is_metal]

    def refresh_atom_id(self):
        """ Refresh the Id of atoms """
        for i, atom in enumerate(self.atoms):
            atom.id = i

    @property
    def atom_id_dict(self):
        return {a.id: a for a in self.atoms}

    def _rm_atom(self, atom: "Atom"):
        if isinstance(atom, int):
            atom = self._atoms[atom]

        rm_bonds = atom.bonds
        for rmb in rm_bonds:
            self._bonds.remove(rmb)
        self._atoms.remove(atom)

    def _rm_atoms(self, atoms: Iterable[Union["Atom", int]]):
        atoms = [a if isinstance(a, Atom) else self._atoms[a] for a in atoms]

        rm_bonds = {b for a in atoms for b in a.bonds}
        for rmb in rm_bonds:
            self._bonds.remove(rmb)
        for rma in atoms:
            self._atoms.remove(rma)

    def remove_atom(self, atom: Union[int, "Atom"]) -> None:
        self._rm_atom(atom)
        self._update_graph()

    def remove_atoms(self, atoms: Iterable["Atom"]) -> None:
        self._rm_atoms(atoms)
        self._update_graph()

    def _rm_bond(self, bond: "Bond"):
        self._bonds.remove(bond)

    def _rm_bonds(self, bonds: Iterable["Bond"]):
        for bond in bonds:
            self._bonds.remove(bond)

    def remove_bond(self, bond: "Bond") -> None:
        self._bonds.remove(bond)
        self._update_graph()

    def remove_bonds(self, bonds: Iterable["Bond"]) -> None:
        self._rm_bonds(bonds)
        self._update_graph()

    def remove_hydrogens(self):
        self.remove_atoms([a for a in self._atoms if a.is_hydrogen])

    def set_default_valence(self):
        for atom in self._atoms:
            atom.set_default_valence()

    def similarity(
            self,
            other: "Molecule",
            method: str = "spectrum",
            norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'
    ):
        if method == "spectrum":
            return round(self.graph_spectral(norm) | other.graph_spectral(norm), 15)
        else:
            raise NotImplementedError

    @staticmethod
    def _longest_path(graph: nx.Graph, _start_node: int = None, _end_node: int = None):
        longest_path = []
        try:
            if isinstance(_start_node, int) and isinstance(_end_node, int):
                return max(nx.all_simple_paths(graph, _start_node, _end_node), key=len)

            elif isinstance(_start_node, int) and _end_node is None:
                for end_node in graph.nodes:
                    if end_node == _start_node:
                        longest_path.append([end_node])
                    else:
                        longest_path.append(max(nx.all_simple_paths(graph, _start_node, end_node), key=len))

            elif _start_node is None and isinstance(_end_node, int):
                for start_node in graph.nodes:
                    if start_node == _end_node:
                        longest_path.append([_end_node])
                    else:
                        longest_path.append(max(nx.all_simple_paths(graph, start_node, _end_node), key=len))

            else:
                for start_node, end_node in combinations(graph.nodes, 2):
                    longest_path.append(max(nx.all_simple_paths(graph, start_node, end_node), key=len))

        except ValueError as e:
            if e.args[0] == "max() arg is an empty sequence":
                path = []
                if isinstance(_start_node, int):
                    path.append(_start_node)
                if isinstance(_end_node, int):
                    path.append(_end_node)
                longest_path.append(path)

        return max(longest_path, key=len)

    def longest_path(self):
        return self._longest_path(self.graph)

    def canonical_tree(self):
        """ Return the canonical tree without cycles. """
        def _dfs(
                _graph: nx.Graph,
                _longest_path: tuple[int],
                _nested: dict
        ):
            branch_start = [(n, p) for p in _longest_path for n in _graph.neighbors(p) if n not in _longest_path]
            if branch_start:
                clone = _graph.copy()
                clone.remove_edges_from(branch_start)
            else:
                return

            for connected_nodes in nx.connected_components(clone):

                # Exclude the trunk path (parent longest path)
                if not connected_nodes - set(_longest_path):
                    continue

                sub_graph = clone.subgraph(connected_nodes)
                start_node, parent_node = next((n, p) for n, p in branch_start if n in connected_nodes)

                longest_path = self._longest_path(sub_graph, start_node)

                list_nested = _nested.setdefault(parent_node, [])
                _sub_nested = {}
                list_nested.append((longest_path, _sub_nested))

                _dfs(sub_graph, longest_path, _sub_nested)

        graph = self.graph
        nested_tree = (self._longest_path(graph), {})
        _dfs(graph, nested_tree[0], nested_tree[1])

        return nested_tree

    def super_acyclic_graph(self):
        raise NotImplementedError

    @property
    def smiles(self) -> str:
        """ Return smiles string. """
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        return pb.readstring('mol2', pb.Molecule(mol2obmol(self)[0]).write('mol2')).write('can').split()[0]
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        # return pb.Molecule(mol2obmol(self)[0]).write().strip()

    @property
    def rings(self) -> list["Ring"]:
        if not self._rings:
            self._rings = [Ring(*(self._atoms[i] for i in cycle)) for cycle in nx.cycle_basis(self.graph)]

        return copy(self._rings)

    @property
    def rings_small(self) -> list["Ring"]:
        return [r for r in self.rings if len(r) <= 8]

    @property
    def lssr(self):
        return None

    def to_obmol(self) -> ob.OBMol:
        return mol2obmol(self)[0]

    def to_rdmol(self):
        return to_rdmol(self)

    def to_pybel_mol(self) -> pb.Molecule:
        return pb.Molecule(mol2obmol(self)[0])

    def translation(self, vector: types.ArrayLike):
        vector = np.array(vector).flatten()
        assert len(vector) == 3

        coordinates = self.coordinates
        self.coordinates = coordinates + vector

    def update_atoms_attrs_from_id_dict(self, id_dict: dict[int, dict]):
        id_atoms = self.atom_id_dict
        for i, attr in id_dict.items():
            id_atoms[i].setattr(**attr)

    def update_angles(self):
        self._angles = [Angle(n1, a, n2) for a in self.atoms for n1, n2 in combinations(a.neighbours, 2)]

    def update_torsions(self):
        self._torsions = self._retrieve_torsions()

    @property
    def weight(self):
        return sum(a.mass for a in self._atoms)

    def write(
            self,
            filename=None,
            fmt: Optional[str] = None,
            overwrite=False,
            write_single: bool = False,
            ob_opt: dict = None,
            **kwargs
    ):
        # write_by_pybel(self, fmt, str(filename), overwrite, opt)
        writer = _io.MolWriter(filename, fmt, overwrite=overwrite, **kwargs)
        writer.write(self, write_single=write_single)


class MolBlock:
    _attrs_dict = {}
    _default_attrs = {}
    _attrs_enumerator = tuple(_attrs_dict.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label})"

    def __copy__(self):
        raise PermissionError(f"The {self.__class__.__name__} not allow to copy")

    def __getattr__(self, item):
        try:
            attr_idx = self._attrs_enumerator.index(item)
            return self._attrs_dict[item](self.attrs[attr_idx])
        except ValueError:
            # raise AttributeError(f"{item} is not an attribute of the {self.__class__.__name__}")
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        try:
            attr_idx = self._attrs_enumerator.index(key)
            self.attrs[attr_idx] = value
        except ValueError:
            super().__setattr__(key, value)

    @property
    def attrs_enumerator(self) -> tuple:
        return self._attrs_enumerator

    @property
    def attr_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.attrs_enumerator}

    @property
    def in_ring(self):
        return any(self in r for r in self.mol.rings)

    @property
    def rings(self):
        return [r for r in self.mol.rings if self in r]

    def setattr(self, *, add_defaults=False, **kwargs):
        _attrs = copy(self._default_attrs) if add_defaults else {}
        _attrs.update(kwargs)
        for name, value in _attrs.items():
            setattr(self, name, value)


class Atom(MolBlock):
    _symbols = (
        "0",
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "", ""
    )

    _attrs_dict = {
        # Name: datatype
        'atomic_number': int,
        'formal_charge': int,
        'partial_charge': float,
        'is_aromatic': bool,
        'x': float,
        'y': float,
        'z': float,
        'valence': int,
        'implicit_hydrogens': int,
        'id': int,
        'x_constraint': bool,
        'y_constraint': bool,
        'z_constraint': bool,
        # 'explicit_hydrogens'
    }

    _default_attrs = {
        "atomic_number": 0,
        "is_aromatic": False,
        "formal_charge": 0,
        "partial_charge": 0.,
        "coordinates": (0., 0., 0.),
        "x": 0.,
        "y": 0.,
        "z": 0.,
        "valence": 0,
        "implicit_hydrogens": 0,
    }

    _atomic_orbital = [       # Periodic
        [2],                  # 1: 1s
        [2, 6],               # 2: 2s, 2p
        [2, 6],               # 3: 3s, 3p
        [2, 10, 6],           # 4: 4s, 3d, 4p
        [2, 10, 6],           # 5: 5s, 4d, 5p
        [2, 14, 10, 6],       # 6: 6s, 4f, 5d, 6p
        [2, 14, 10, 6],       # 7: 7s, 5f, 6d, 7p
        [2, 18, 14, 10, 6]    # 8: 8s, 5g, 6f, 7d, 8p
    ]

    _default_valence = {
        1: 1,    # Hydrogen (H)
        2: 0,    # Helium (He) - inert
        3: 1,    # Lithium (Li)
        4: 2,    # Beryllium (Be)
        5: 3,    # Boron (B)
        6: 4,    # Carbon (C)
        7: 3,    # Nitrogen (N)
        8: 2,    # Oxygen (O)
        9: 1,    # Fluorine (F)
        10: 0,   # Neon (Ne) - inert
        11: 1,   # Sodium (Na)
        12: 2,   # Magnesium (Mg)
        13: 3,   # Aluminium (Al)
        14: 4,   # Silicon (Si)
        15: 3,   # Phosphorus (P)
        16: 2,   # Sulfur (S)
        17: 1,   # Chlorine (Cl)
        18: 0,   # Argon (Ar) - inert
        19: 1,   # Potassium (K)
        20: 2,   # Calcium (Ca)
        21: 3,   # Scandium (Sc)
        22: 4,   # Titanium (Ti)
        23: 5,   # Vanadium (V)
        24: 3,   # Chromium (Cr)
        25: 2,   # Manganese (Mn)
        26: 2,   # Iron (Fe)
        27: 3,   # Cobalt (Co)
        28: 2,   # Nickel (Ni)
        29: 2,   # Copper (Cu)
        30: 2,   # Zinc (Zn)
        31: 3,   # Gallium (Ga)
        32: 4,   # Germanium (Ge)
        33: 3,   # Arsenic (As)
        34: 2,   # Selenium (Se)
        35: 1,   # Bromine (Br)
        36: 0,   # Krypton (Kr) - inert
        37: 1,   # Rubidium (Rb)
        38: 2,   # Strontium (Sr)
        39: 3,   # Yttrium (Y)
        40: 4,   # Zirconium (Zr)
        41: 5,   # Niobium (Nb)
        42: 6,   # Molybdenum (Mo)
        43: 7,   # Technetium (Tc)
        44: 4,   # Ruthenium (Ru)
        45: 3,   # Rhodium (Rh)
        46: 2,   # Palladium (Pd)
        47: 1,   # Silver (Ag)
        48: 2,   # Cadmium (Cd)
        49: 3,   # Indium (In)
        50: 4,   # Tin (Sn)
        51: 3,   # Antimony (Sb)
        52: 2,   # Tellurium (Te)
        53: 1,   # Iodine (I)
        54: 0,   # Xenon (Xe) - inert
        55: 1,   # Cesium (Cs)
        56: 2,   # Barium (Ba)
        57: 3,   # Lanthanum (La)
        58: 3,   # Cerium (Ce)
        59: 3,   # Praseodymium (Pr)
        60: 3,   # Neodymium (Nd)
        61: 3,   # Promethium (Pm)
        62: 3,   # Samarium (Sm)
        63: 3,   # Europium (Eu)
        64: 3,   # Gadolinium (Gd)
        65: 3,   # Terbium (Tb)
        66: 3,   # Dysprosium (Dy)
        67: 3,   # Holmium (Ho)
        68: 3,   # Erbium (Er)
        69: 3,   # Thulium (Tm)
        70: 3,   # Ytterbium (Yb)
        71: 3,   # Lutetium (Lu)
        72: 4,   # Hafnium (Hf)
        73: 5,   # Tantalum (Ta)
        74: 6,   # Tungsten (W)
        75: 5,   # Rhenium (Re)
        76: 4,   # Osmium (Os)
        77: 3,   # Iridium (Ir)
        78: 2,   # Platinum (Pt)
        79: 1,   # Gold (Au)
        80: 2,   # Mercury (Hg)
        81: 3,   # Thallium (Tl)
        82: 4,   # Lead (Pb)
        83: 3,   # Bismuth (Bi)
        84: 2,   # Polonium (Po)
        85: 1,   # Astatine (At)
        86: 0,   # Radon (Rn) - inert
        87: 1,   # Francium (Fr)
        88: 2,   # Radium (Ra)
        89: 3,   # Actinium (Ac)
        90: 4,   # Thorium (Th)
        91: 5,   # Protactinium (Pa)
        92: 6,   # Uranium (U)
        93: 5,   # Neptunium (Np)
        94: 6,   # Plutonium (Pu)
        95: 3,   # Americium (Am)
        96: 3,   # Curium (Cm)
        97: 3,   # Berkelium (Bk)
        98: 3,   # Californium (Cf)
        99: 3,   # Einsteinium (Es)
        100: 3,  # Fermium (Fm)
        101: 3,  # Mendelevium (Md)
        102: 3,  # Nobelium (No)
        103: 3,  # Lawrencium (Lr)
        104: 4,  # Rutherfordium (Rf)
        105: 5,  # Dubnium (Db)
        106: 6,  # Seaborgium (Sg)
        107: 7,  # Bohrium (Bh)
        108: 4,  # Hassium (Hs)
        109: 3,  # Meitnerium (Mt)
        110: 4,  # Darmstadtium (Ds)
        111: 1,  # Roentgenium (Rg)
        112: 2,  # Copernicium (Cn)
        113: 3,  # Nihonium (Nh)
        114: 4,  # Flerovium (Fl)
        115: 3,  # Moscovium (Mc)
        116: 2,  # Livermorium (Lv)
        117: 1,  # Tennessine (Ts)
        118: 0,  # Oganesson (Og) - inert
    }

    _electronegativity = {
        1: 2.20,  # Hydrogen (H)
        2: None,  # Helium (He)
        3: 0.98,  # Lithium (Li)
        4: 1.57,  # Beryllium (Be)
        5: 2.04,  # Boron (B)
        6: 2.55,  # Carbon (C)
        7: 3.04,  # Nitrogen (N)
        8: 3.44,  # Oxygen (O)
        9: 3.98,  # Fluorine (F)
        10: None,  # Neon (Ne)
        11: 0.93,  # Sodium (Na)
        12: 1.31,  # Magnesium (Mg)
        13: 1.61,  # Aluminum (Al)
        14: 1.90,  # Silicon (Si)
        15: 2.19,  # Phosphorus (P)
        16: 2.58,  # Sulfur (S)
        17: 3.16,  # Chlorine (Cl)
        18: None,  # Argon (Ar)
        19: 0.82,  # Potassium (K)
        20: 1.00,  # Calcium (Ca)
        21: 1.36,  # Scandium (Sc)
        22: 1.54,  # Titanium (Ti)
        23: 1.63,  # Vanadium (V)
        24: 1.66,  # Chromium (Cr)
        25: 1.55,  # Manganese (Mn)
        26: 1.83,  # Iron (Fe)
        27: 1.88,  # Cobalt (Co)
        28: 1.91,  # Nickel (Ni)
        29: 1.90,  # Copper (Cu)
        30: 1.65,  # Zinc (Zn)
        31: 1.81,  # Gallium (Ga)
        32: 2.01,  # Germanium (Ge)
        33: 2.18,  # Arsenic (As)
        34: 2.55,  # Selenium (Se)
        35: 2.96,  # Bromine (Br)
        36: 3.00,  # Krypton (Kr)
        37: 0.82,  # Rubidium (Rb)
        38: 0.95,  # Strontium (Sr)
        39: 1.22,  # Yttrium (Y)
        40: 1.33,  # Zirconium (Zr)
        41: 1.60,  # Niobium (Nb)
        42: 2.16,  # Molybdenum (Mo)
        43: 1.90,  # Technetium (Tc)
        44: 2.20,  # Ruthenium (Ru)
        45: 2.28,  # Rhodium (Rh)
        46: 2.20,  # Palladium (Pd)
        47: 1.93,  # Silver (Ag)
        48: 1.69,  # Cadmium (Cd)
        49: 1.78,  # Indium (In)
        50: 1.96,  # Tin (Sn)
        51: 2.05,  # Antimony (Sb)
        52: 2.10,  # Tellurium (Te)
        53: 2.66,  # Iodine (I)
        54: 2.60,  # Xenon (Xe)
        55: 0.79,  # Cesium (Cs)
        56: 0.89,  # Barium (Ba)
        57: 1.10,  # Lanthanum (La)
        58: 1.12,  # Cerium (Ce)
        59: 1.13,  # Praseodymium (Pr)
        60: 1.14,  # Neodymium (Nd)
        61: 1.13,  # Promethium (Pm)
        62: 1.17,  # Samarium (Sm)
        63: 1.20,  # Europium (Eu)
        64: 1.20,  # Gadolinium (Gd)
        65: 1.22,  # Terbium (Tb)
        66: 1.23,  # Dysprosium (Dy)
        67: 1.24,  # Holmium (Ho)
        68: 1.24,  # Erbium (Er)
        69: 1.25,  # Thulium (Tm)
        70: 1.10,  # Ytterbium (Yb)
        71: 1.27,  # Lutetium (Lu)
        72: 1.30,  # Hafnium (Hf)
        73: 1.50,  # Tantalum (Ta)
        74: 2.36,  # Tungsten (W)
        75: 1.90,  # Rhenium (Re)
        76: 2.20,  # Osmium (Os)
        77: 2.20,  # Iridium (Ir)
        78: 2.28,  # Platinum (Pt)
        79: 2.54,  # Gold (Au)
        80: 2.00,  # Mercury (Hg)
        81: 1.62,  # Thallium (Tl)
        82: 2.33,  # Lead (Pb)
        83: 2.02,  # Bismuth (Bi)
        84: 2.00,  # Polonium (Po)
        85: 2.20,  # Astatine (At)
        86: None,  # Radon (Rn)
        87: 0.70,  # Francium (Fr)
        88: 0.89,  # Radium (Ra)
        89: 1.10,  # Actinium (Ac)
        90: 1.30,  # Thorium (Th)
        91: 1.50,  # Protactinium (Pa)
        92: 1.38,  # Uranium (U)
        93: 1.36,  # Neptunium (Np)
        94: 1.28,  # Plutonium (Pu)
        95: 1.30,  # Americium (Am)
        96: 1.30,  # Curium (Cm)
        97: 1.30,  # Berkelium (Bk)
        98: 1.30,  # Californium (Cf)
        99: 1.30,  # Einsteinium (Es)
        100: 1.30,  # Fermium (Fm)
        101: 1.30,  # Mendelevium (Md)
        102: 1.30,  # Nobelium (No)
        103: None,  # Lawrencium (Lr)
        104: None,  # Rutherfordium (Rf)
        105: None,  # Dubnium (Db)
        106: None,  # Seaborgium (Sg)
        107: None,  # Bohrium (Bh)
        108: None,  # Hassium (Hs)
        109: None,  # Meitnerium (Mt)
        110: None,  # Darmstadtium (Ds)
        111: None,  # Roentgenium (Rg)
        112: None,  # Copernicium (Cn)
        113: None,  # Nihonium (Nh)
        114: None,  # Flerovium (Fl)
        115: None,  # Moscovium (Mc)
        116: None,  # Livermorium (Lv)
        117: None,  # Tennessine (Ts)
        118: None  # Oganesson (Og)
    }

    _attrs_enumerator = tuple(_attrs_dict.keys())

    # Element categorize in periodic tabel
    _alkali_metals = [3, 11, 19, 37, 55, 87]  # Group 1
    _alkaline_earth_metals = [4, 12, 20, 38, 56, 88]  # Group 2
    _transition_metals = list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)) + list(range(104, 113))
    _post_transition_metals = [13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116]
    _lanthanides = list(range(57, 72))
    _actinides = list(range(89, 104))
    metal_ = set(_alkali_metals + _alkaline_earth_metals + _transition_metals
                 + _post_transition_metals + _lanthanides + _actinides)

    _nonmetals = [1, 6, 7, 8, 15, 16, 34]
    _metalloids = [5, 14, 32, 33, 51, 52, 84]
    _noble_gases = [2, 10, 18, 36, 54, 86, 118]
    _halogens = [9, 17, 35, 53, 85, 117]

    covalent_radii = np.array([0.] + [getattr(periodictable, ob.GetSymbol(i)).covalent_radius or 0. for i in range(1, 119)])

    def __init__(self, mol: Molecule = None, **kwargs):
        self.mol = mol or Molecule()
        getattr(self.mol, '_atoms').append(self)

        self.attrs = np.zeros(len(self._attrs_enumerator))

        self.setattr(add_defaults=True, **kwargs)

    @classmethod
    def _get_atom_attr_dict(cls, atomic_number: int) -> dict:
        oba = ob.OBAtom()
        oba.SetAtomicNum(atomic_number)
        return dict(
            atomic_number=oba.GetAtomicNum(),
            formal_charge=oba.GetFormalCharge(),
            partial_charge=oba.GetPartialCharge(),
            valence=oba.GetTotalValence(),
            implicit_hydrogens=oba.GetImplicitHCount()
        )

    def _add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        if isinstance(atom, str):
            atom = ob.GetAtomicNum(atom)

        if isinstance(atom, int):
            atom_attrs_ = self._get_atom_attr_dict(atom)
        else:
            atom_attrs_ = atom.attr_dict

        atom_attrs_.update(atom_attrs or {})
        atom = self.mol._create_atom(**atom_attrs_)
        getattr(self.mol, '_add_bond')(self, atom, bond_order, **(bond_attrs or {}))

        return atom

    def add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        atom = self._add_atom(atom, bond_order, atom_attrs, bond_attrs)
        getattr(self.mol, '_update_graph')()
        return atom

    def _add_hydrogens(self, num: int = None, remove_excess: bool = False) -> (int, list["Atom"]):
        hs_metals = [a for a in self.neighbours if a.atomic_number == 1 or a.is_metal]
        if num is None:
            num = self.implicit_hydrogens - len(hs_metals)

        if num > 0:
            return 1, [self._add_atom() for _ in range(num)]
        elif num < 0 and remove_excess:
            to_remove = [a for a in hs_metals if a.atomic_number == 1][:abs(num)]
            if to_remove:
                self.mol._rm_atoms(to_remove)
                return -1, to_remove
            else:
                return 0, []
        else:
            return 0, []

    def add_hydrogen(self, num: int = None) -> list["Atom"]:
        add_or_rm, hydrogens = self._add_hydrogens(num)
        getattr(self.mol, '_update_graph')()
        return hydrogens

    @property
    def bonds(self) -> list["Bond"]:
        # return [self.mol.bonds[i] for i in self.bonds_idx]
        edge_viewer = self.mol.graph.edges
        return [edge_viewer[u, v]['bond'] for u, v in edge_viewer(self.idx)]

    def calc_implicit_hydrogens(self):
        self.implicit_hydrogens = self.valence - self.sum_bond_orders

    @property
    def constraint(self) -> bool:
        return all([self.x_constraint, self.y_constraint, self.z_constraint])

    @constraint.setter
    def constraint(self, value: bool):
        self.x_constraint = self.y_constraint = self.z_constraint = value

    @property
    def coordinates(self):
        return self.x, self.y, self.z

    @coordinates.setter
    def coordinates(self, value: types.ArrayLike):
        self.x, self.y, self.z = value

    @property
    def exact_mass(self):
        return ob.GetExactMass(self.atomic_number)

    @property
    def electronegativity(self) -> Optional[float]:
        return self._electronegativity[self.atomic_number]

    @property
    def hyb(self) -> int:
        if self.is_metal:
            return 0

        return 3 - (self.missing_electrons_element - len(self.bonds) - self.implicit_hydrogens)

    @property
    def idx(self):
        return self.mol.atoms.index(self)

    @property
    def is_error_electron_configure(self) -> bool:
        if self.is_metal:
           return False

        if self.missing_electrons_element != 0:
            return True
        return False

    @property
    def is_hydrogen(self) -> bool:
        return self.atomic_number == 1

    @property
    def is_metal(self) -> bool:
        return self.atomic_number in self.metal_

    @property
    def label(self) -> str:
        return f"{self.symbol}{self.idx}"

    def link_with(self, other: "Atom"):
        assert self.mol is not other.mol

        other_clone_idx = other.idx + len(self.mol.atoms)
        self.mol.add_component(other.mol)

        other_clone = self.mol.atoms[other_clone_idx]
        self.mol.add_bond(self, other_clone)

    @property
    def mass(self):
        return ob.GetMass(self.atomic_number)

    @property
    def molecule(self):
        return self.mol

    @property
    def neigh_idx(self) -> np.ndarray:
        return np.array(list(self.mol.graph.neighbors(self.idx)), dtype=int)

    @property
    def neighbours(self) -> list['Atom']:
        return np.take(self.mol.atoms, self.neigh_idx).tolist()

    @property
    def missing_electrons_element(self):
        """ missing electrons in open shell for isolated element. """
        n, l, conf = self.calc_electron_config()
        return sum(self._atomic_orbital[n]) - sum(conf)

    @property
    def missing_electrons(self):
        """ missing electrons in open shell for this atom. """
        return self.missing_electrons_element - self.sum_bond_orders

    @property
    def open_shell_electrons(self) -> int:
        n, l, conf = self.calc_electron_config()
        return sum(conf)

    @property
    def l(self) -> int:
        n, l, conf = self.calc_electron_config()
        return l

    @property
    def n(self) -> int:
        n, l, conf = self.calc_electron_config()
        return n

    def calc_electron_config(self) -> (int, int, list[int]):
        shells = self._atomic_orbital
        #       s  p  d  f, g
        conf = [0, 0, 0, 0, 0]
        _atomic_number = self.atomic_number

        n = 0
        l = 0
        while _atomic_number > 0:
            if l >= len(shells[n]):
                n += 1
                l = 0
                conf = [0, 0, 0, 0, 0]

            if _atomic_number - shells[n][l] > 0:
                conf[l] = shells[n][l]
            else:
                conf[l] = _atomic_number

            _atomic_number -= shells[n][l]
            l += 1

        return n, l, conf

    @property
    def oxidation_state(self) -> int:
        if self.is_metal:
            return self.formal_charge
        elif self.atomic_number == 6:
            return 4
        elif self.atomic_number == 7:
            return 3
        elif self.atomic_number == 8:
            return 2
        elif self.atomic_number in [1, 9, 17]:  # H, F, Cl
            return 1

        _state = 0
        _temp_state = 0
        for bond in self.bonds:
            other_atom = bond.another_end(self)
            if other_atom.is_metal or other_atom.atomic_number == 1:
                continue

            if other_atom.atomic_number == 6:
                _temp_state += 1

            elif self.electronegativity > other_atom.electronegativity:
                _state -= int(bond.bond_order)
            # elif self.electronegativity < other_atom.electronegativity:
            else:
                _state += int(bond.bond_order)

        _state += self.implicit_hydrogens

        if _state > 0:
            _state += _temp_state
        elif _state < 0:
            _state -= _temp_state

        return _state

    def set_valence_to_default(self):
        self.valence = Atom._default_valence[self.atomic_number]

    def setattr(self, *, add_defaults=False, **kwargs):
        coords = kwargs.get("coordinates", None)
        symbol = kwargs.get("symbol", None)

        if coords is not None:
            del kwargs["coordinates"]
            kwargs.update({'x': coords[0], 'y': coords[1], 'z': coords[2]})
        if isinstance(symbol, str):
            del kwargs["symbol"]
            kwargs['atomic_number'] = Atom._symbols.index(symbol)

        super().setattr(add_defaults=add_defaults, **kwargs)

    @property
    def sum_bond_orders(self) -> int:
        return int(sum(b.bond_order for b in self.bonds))

    @property
    def symbol(self):
        return self._symbols[self.atomic_number]

    @symbol.setter
    def symbol(self, value: str):
        self.atomic_number = self._symbols.index(value)

    @property
    def vector(self):
        return np.array([self.coordinates])


class AtomSeq:
    """ Represent instances assembled by a sequence of atoms, like Bond, Angles, Torison, and Rings. """
    def __init__(self, *atoms: Atom):
        self._check_is_same_mol(atoms)
        self._atoms = atoms

        if isinstance(self, Bond):
            self._bonds = [self]
        else:
            self._bonds = [self.mol.bond(atoms[i].idx, atoms[i+1].idx) for i in range(len(atoms) - 1)]

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __getattr__(self, item):
        if re.match(r"atom\d+", item):
            idx = int(item[4:]) - 1
            return self._atoms[idx]

        elif re.match(r"bond\d+", item):
            idx = int(item[4:]) - 1
            atom_start = self._atoms[idx]
            atom_end = self._atoms[idx+1]
            return list(set(atom_start.bonds) & set(atom_end.bonds))[0]

        elif re.match(r"a\d+idx", item):
            len_idx = len(item) - 4
            idx = int(item[1:1+len_idx]) - 1
            return self._atoms[idx].idx

        else:
            super().__getattribute__(item)

    def __len__(self):
        return len(self._atoms)

    def __contains__(self, item: Union[Atom, "Bond"]):
        if isinstance(item, Atom):
            return item in self._atoms
        elif isinstance(item, Bond):
            return item in self._bonds

    @staticmethod
    def _check_is_same_mol(*atoms):
        if any(atoms[0].mol is not a.mol for a in atoms[1:]):
            raise ValueError('All atoms must belong to same mol.')

    @property
    def atoms(self):
        return copy(self._atoms)

    @property
    def bonds(self):
        return copy(self._bonds)

    @property
    def mol(self) -> Molecule:
        return self._atoms[0].mol

    @property
    def has_metal(self):
        return any(a.is_metal for a in self._atoms)

    def to_mol(self):
        mol = Molecule()
        for atom in self._atoms:
            mol.create_atom(**atom.attr_dict)

        mol_atoms = mol.atoms
        for bond in self._bonds:
            a1 = mol_atoms[self._atoms.index(bond.atom1)]
            a2 = mol_atoms[self._atoms.index(bond.atom2)]
            mol.add_bond(a1, a2, **bond.attr_dict)

        return mol

class Bond(AtomSeq, MolBlock):
    _attrs_dict = {
        'bond_order': float,
        'constraint': bool
    }
    _attrs_enumerator = tuple(_attrs_dict.keys())
    _bond_order_symbol = {
        0.: '?',
        1.: '-',
        1.5: '@',
        2.: '=',
        3.: '#'
    }
    _bond_order_names = {
        0.: 'Unknown',
        1.: 'Single',
        1.5: 'Aromatic',
        2.: 'Double',
        3.: 'Triple'
    }

    def __init__(self, atom1: Atom, atom2: Atom, **kwargs):
        super().__init__(atom1, atom2)
        self.attrs = np.zeros(len(self._attrs_enumerator))

        self.setattr(add_defaults=True, **kwargs)

    def __repr__(self):
        return MolBlock.__repr__(self)

    def __eq__(self, other):
        return (other.atom1 == self.atom1 and other.atom2 == self.atom2) or (other.atom1 == self.atom2 and other.atom2 == self.atom1)

    def __hash__(self):
        return hash((self.atom1, self.atom2)) + hash((self.atom2, self.atom1))

    def __getattr__(self, item):
        try:
            return AtomSeq.__getattr__(self, item)
        except AttributeError:
            return MolBlock.__getattr__(self, item)

    def another_end(self, this_end: Atom):
        try:
            return self.atoms[abs(self.atoms.index(this_end) - 1)]
        except ValueError:
            raise ValueError("The given atom is in neither ends of the bond!")

    @property
    def bond_line(self) -> geometry.Line:
        return geometry.Line(self.atom1.coordinates, self.atom2.coordinates)

    @property
    def idx(self) -> int:
        return self.mol.bonds.index(self)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.atom1.vector - self.atom2.vector))

    @property
    def label(self):
        return f"{self.atom1.label}{self._bond_order_symbol[self.bond_order]}{self.atom2.label}"

    @property
    def is_aromatic(self) -> bool:
        return any(r.is_aromatic for r in self.rings)

    # @is_aromatic.setter
    # def is_aromatic(self, value: bool):
    #     if not self.in_ring:
    #         raise AttributeError("Can't assign a bond outside an ring to be ")
    #
    #     self.atom1.is_aromatic = value
    #     self.atom2.is_aromatic = value

    @property
    def is_metal_ligand_bond(self) -> bool:
        return (self.atom1.is_metal and not self.atom2.is_metal) or (not self.atom1.is_metal and self.atom2.is_metal)

    @property
    def rotatable(self) -> bool:
        return (self.bond_order == 1) and (not self.is_aromatic) and (not self.in_ring)


class Angle(AtomSeq):

    def __init__(self, a1: Atom, a2: Atom, a3: Atom):
        super().__init__(a1, a2, a3)
        self.constraint = False

    @property
    def degrees(self) -> float:
        return InternalCoordinates.calc_angle(
            self.atom1.coordinates,
            self.atom2.coordinates,
            self.atom3.coordinates
        )


class Torsion(AtomSeq):
    """"""
    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom):
        super().__init__(a, b, c, d)
        self.constraint = False

    @property
    def degrees(self) -> float:
        return InternalCoordinates.calc_dehedral(
            self.atom1.coordinates,
            self.atom2.coordinates,
            self.atom3.coordinates,
            self.atom4.coordinates
        )

    @property
    def rotatable(self) -> bool:
        return self.bond2.rotatable


class Ring(AtomSeq):
    def __init__(self, *atoms: Atom):
        super().__init__(*atoms)
        self._bonds = self._bonds + [self.mol.bond(self._atoms[0].idx, self._atoms[-1].idx)]

    @property
    def is_aromatic(self) -> bool:
        return all(a.is_aromatic for a in self._atoms)

    def is_bond_intersect_the_ring(self, bond: Bond) -> bool:
        if bond in self._bonds:
            return False

        return self.cycle_places.is_line_intersect_the_cycle(bond.bond_line)

    @property
    def cycle_places(self) -> geometry.CyclePlanes:
        return geometry.CyclePlanes(*[a.coordinates for a in self.atoms])

    def perceive_aromatic(self):
        pi_electrons = 0

        for atom in self._atoms:
            if atom.atomic_number == 6 and atom.hyb == 2:  # C
                pi_electrons += 1
            elif atom.atomic_number == 7 and atom.hyb == 2:  # N
                if atom.sum_bond_orders == 2:
                    pi_electrons += 2
                elif atom.sum_bond_orders == 3:
                    pi_electrons += 1
            elif atom.atomic_number in [8, 15, 16]:  # O P S
                pi_electrons += 2
            else:
                return False

        return (pi_electrons - 2) % 4 == 0


class Conformers:
    """ Representing an ensemble of conformers of a Molecule """
    _attrs = (
        'coordinates',
        'energy',
        'partial_charge',
        'gibbs',
        'force',
        'zero_point',
        'spin_mult',
        'thermo',
        'capacity'
    )

    def __init__(self):
        self._coordinates = None
        self._energy = None
        self._partial_charges = None
        self._spin_mult = None
        self._gibbs = None
        self._force = None
        self._zero_point = None
        self._thermo = None
        self._capacity = None

    def __len__(self):
        if self._coordinates is None:
            return 0
        return len(self._coordinates)

    def __getitem__(self, idx):
        info = {}
        for attr in self._attrs:
            try:
                info[attr] = self.index_attr(attr, idx)
            except (IndexError, TypeError):
                continue

        return info

    def __iter__(self):
        return iter(self._coordinates)

    def index_attr(self, name, i):
        try:
            return getattr(self, f"_{name}")[i]
        except IndexError:
            raise IndexError(f'The number of conformers is {len(self._coordinates)}, but attempt to access the index {i}')
        except TypeError:
            try:
                return getattr(self, f"_{name}")
            except AttributeError:
                raise AttributeError(f"The conformers did not store {name} information")

    def add(self, coords: types.ArrayLike, energy: Union[types.ArrayLike, float] = None):
        coords = np.array(coords)

        if coords.ndim == 2:
            coords = coords.reshape((1,) + coords.shape)

        assert coords.ndim == 3
        assert coords.shape[-1] == 3

        if isinstance(self._coordinates, np.ndarray) and self._coordinates.shape[1] == coords.shape[1]:
            coords = np.vstack((self._coordinates, coords))

        self._coordinates = coords

        if energy is not None:
            if isinstance(energy, float):
                energy = np.array([energy])
            else:
                energy = np.array(energy).flatten()

            if self._energy is not None:
                energy = np.vstack((self._coordinates, coords))

            if len(energy) != len(self._coordinates):
                raise ValueError("The length of input energy in conformer nust matches with the coordinates in conformer")

            self._energy = energy

    def clear(self):
        self._coordinates = None
        self._energy = None

    def coordinates(self, i):
        return self.index_attr('coordinates', i)

    def energy(self, i):
        return self.index_attr('energy', i)

    def gibbs(self, i):
        return self.index_attr('gibbs', i)

    def partial_charge(self, i):
        return self.index_attr('partial_charge', i)

    def force(self, i):
        return self.index_attr('force', i)




class InternalCoordinates:
    """"""
    def __init__(self, coordinates: np.ndarray, link_matrix: np.ndarray):
        self.coordinates = coordinates
        self.link_matrix = link_matrix
        self._graph = nx.Graph()
        self._graph.add_edges_from(link_matrix)

    @staticmethod
    def calc_zmat_idx(graph_or_link: Union[nx.Graph, np.ndarray]) -> np.ndarray:
        """ Calculate the z-matrix of a graph represented by a link matrix """
        if isinstance(graph_or_link, np.ndarray):
            assert graph_or_link.shape[1] == 2
            _graph = nx.Graph()
            _graph.add_edges_from(graph_or_link)
        elif isinstance(graph_or_link, nx.Graph):
            _graph = graph_or_link
        else:
            raise TypeError(f"The graph_or_link argument should be a nx.Graph or np.ndarray")

        # TODO: check the nodes id, the nodes key should from 0 to n-1

        # Create z-matrix index
        zmat_idx = -1 * np.ones((_graph.number_of_nodes(), 4), dtype=int)
        visited_nodes = set()
        num_nodes = len(_graph)
        node = 0
        while len(visited_nodes) < num_nodes:
            row = len(visited_nodes)
            visited_nodes.add(node)

            path = graph.graph_dfs_path(_graph, node, scope_nodes=visited_nodes, max_deep=min(row + 1, 4))
            zmat_idx[row, :row+1] = path

            try:
                node = min(n for vn in visited_nodes for n in _graph.neighbors(vn) if n not in visited_nodes)
            except ValueError:
                if not len(visited_nodes) == num_nodes:
                    raise RuntimeError("Get an incorrect Z-Matirx !!")

        return zmat_idx

    @staticmethod
    def calc_zmat(graph_or_link: Union[nx.Graph, np.ndarray], coords: np.ndarray) -> np.ndarray:
        """ Calculate the z-matrix of a graph represented by a link matrix """
        zmat_idx = InternalCoordinates.calc_zmat_idx(graph_or_link)
        zmat = -1 * np.ones((zmat_idx.shape[0], 3))

        for i, index in enumerate(zmat_idx):
            zmat[i, :i] = InternalCoordinates._calc_zmat_line(index, coords)

        return zmat

    @staticmethod
    def _calc_zmat_line(index: types.ArrayLike, coords: np.ndarray):
        assert len(index) == 4

        vectors = [coords[index[i]] - coords[index[i-1]] for i in range(1, 4) if index[i] >= 0]

        results = []
        if len(vectors) >= 1:
            r = np.linalg.norm(vectors[0])
            results.append(r)

        if len(vectors) >= 2:
            r1 = np.linalg.norm(vectors[1])
            cos_a = np.dot(vectors[0], vectors[1]) / (r*r1)
            a = np.degrees(np.arccos(cos_a))
            results.append(a)

        if len(vectors) == 3:
            n1 = np.cross(vectors[0], vectors[1])
            n2 = np.cross(vectors[1], vectors[2])

            norm_n1 = np.linalg.norm(n1)
            norm_n2 = np.linalg.norm(n2)

            if norm_n1 == 0 or norm_n2 == 0:
                cos_dehedral = -1
            else:
                cos_dehedral = round(np.dot(n1, n2) / (norm_n1 * norm_n2), 8)

            dehedral = np.degrees(np.arccos(cos_dehedral))
            results.append(dehedral)

        return results

    @staticmethod
    def calc_angle(a: types.ArrayLike, b: types.ArrayLike, c: types.ArrayLike):
        return InternalCoordinates._calc_angle(np.array(a), np.array(b), np.array(c))

    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, return_degree: bool = True):
        # Calculate vectors ab and bc
        ab = b - a
        bc = c - b

        # Calculate the dot product and magnitudes
        abc_dot = np.dot(ab, bc)
        norm_ab = np.linalg.norm(ab)
        norm_bc = np.linalg.norm(bc)

        # Calculate the angle in radians
        cos_theta = abc_dot / (norm_ab * norm_bc)
        angle_radians = np.arccos(cos_theta)

        if return_degree:
            return np.degrees(angle_radians) # Convert to degrees
        else:
            return angle_radians

    @staticmethod
    def calc_dehedral(
            a: types.ArrayLike,
            b: types.ArrayLike,
            c: types.ArrayLike,
            d: types.ArrayLike,
            return_degree: bool = True
    ):
        return InternalCoordinates._calc_dehedral(np.array(a), np.array(b), np.array(c), np.array(d), return_degree)

    @staticmethod
    def _calc_dehedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, return_degree: bool = True):
        # Calculate vectors ab, bc, and cd
        ab = b - a
        bc = c - b
        cd = d - c

        n1 = np.cross(ab, bc)
        n2 = np.cross(bc, cd)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)

        if norm_n1 == 0 or norm_n2 == 0:
            cos_dehedral = -1
        else:
            cos_dehedral = np.dot(n1, n2) / (norm_n1*norm_n2)

        dehedral_radians = np.arccos(cos_dehedral)
        if return_degree:
            return np.degrees(dehedral_radians) # Convert to degrees
        else:
            return dehedral_radians


if __name__ == "__main__":
    pass
