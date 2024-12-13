"""
python v3.9.0
@Project: hotpot
@File   : core__
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 18:30
"""
from typing import Union, Literal, Callable, Iterable, Optional
from copy import copy
from collections import Counter
from itertools import combinations, product

import numpy as np
import networkx as nx
from openbabel import pybel as pb, openbabel as ob
import periodictable

from hotpot.utils import types
from hotpot.cheminfo.obconvert import write_by_pybel, mol2obmol, obmol2mol
from . import graph


# def _update_mol_graph(modifier: Callable):
#     def wrapper(self: _MolCore, *args, **kwargs):
#         modifier(self, *args, **kwargs)
#         self._graph = self._update_graph()
#
#     return wrapper
#
# class _MolCore:
#     def __init__(self, wrapper: "Molecule"):
#         self.wrapper = wrapper
#         self._atoms = []
#         self._bonds = []
#         self._graph = nx.Graph()
#
#     def _node_with_attrs(self):
#         return [(a.idx, {n:getattr(a, n) for n in a.attrs_enumerator}) for a in self._atoms]
#
#     def _edge_with_attrs(self):
#         return [(b.a1idx, b.a2idx, {n:getattr(b, n) for n in b.attrs_enumerator}) for b in self._bonds]
#
#     def _update_graph(self):
#         """ Return networkx _graph with nodes and edges attrs """
#         _graph = nx.Graph()
#         _graph.add_edges_from(self._edge_with_attrs())
#         _graph.add_nodes_from(self._node_with_attrs())
#
#         return _graph
#
#     @_update_mol_graph
#     def add_atom(self, atom):
#         self._atoms.append(atom)
#         atom.mol = self.wrapper
#
#     def add_bond(self, atom1: Union[int, "Atom"], atom2: Union[int, "Atom"], bond_order=1., **kwargs):
#
#         atom1 = atom1 if isinstance(atom1, Atom) else self._atoms[atom1]
#         atom2 = atom2 if isinstance(atom2, Atom) else self._atoms[atom2]
#
#         if atom1.mol is not self or atom2.mol is not self:
#             raise ValueError("at least one of atom1 and atom2 not on the molecule!")
#
#         if (atom1.idx, atom2.idx) in self.graph.edges:
#             raise ValueError(f"the bond between {atom1.idx} and {atom2.idx} is already exist!!")
#
#         kwargs['bond_order'] = bond_order
#         bond = Bond(atom1, atom2, **kwargs)
#         self._bonds.append(bond)
#
#         return bond
#
#     @property
#     def atoms(self):
#         return copy(self._atoms)
#
#     @property
#     def bonds(self):
#         return copy(self._bonds)
#
#     @property
#     def link_matrix(self):
#         return np.array([[b.atom1.idx, b.atom2.idx] for b in self.bonds], dtype=int)
#
#
# class MetaMol(types):
#     """"""
#     _modifier = {'add_atom', 'add_bond', 'rm_atom', 'rm_bond'}
#
#     @classmethod
#     def __prepare__(metacls, name, bases):
#         """"""
#         return {'__core': _MolCore}
#
#     def __new__(cls, name, bases, namespace):
#         """"""

class Molecule:
    def __init__(self):
        self._atoms = []
        # self._bonds = set()
        self._bonds = []
        self._graph = nx.Graph()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.formula})"

    def __copy__(self):
        clone = Molecule()
        for atom in self._atoms:
            clone._create_atom(**atom.attr_dict)
        for bond in self._bonds:
            clone._add_bond(bond.a1idx, bond.a2idx, **bond.attr_dict)

        clone._update_graph()

        return clone

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

    def add_hydrogens(self, recalc_implicit_hydrogens=False):
        """"""
        if recalc_implicit_hydrogens:
            self.calc_implicit_hydrogens()

        for atom in self.atoms:
            if not (atom.is_hydrogen or atom.is_metal):
                atom._add_hydrogens()

        self._update_graph()

    @property
    def angles(self) -> list[tuple[int]]:
        return [(a.idx, n1, n2) for a in self.atoms for n1, n2 in combinations(a.neigh_idx, 2)]

    def angle_degrees(self, center: int, a: int, b: int) -> float:
        return InternalCoordinates.calc_angle(
            self.atoms[a].coordinates,
            self.atoms[center].coordinates,
            self.atoms[b].coordinates
        )

    @property
    def torsions(self) -> list[tuple[int, int]]:
        torsion = []
        for bond in self.bonds:
            a1_neigh = list(bond.atom1.neigh_idx)
            a2_neigh = list(bond.atom2.neigh_idx)
            a1_neigh.remove(bond.atom2.idx)
            a2_neigh.remove(bond.atom1.idx)

            for a, d in product(a1_neigh, a2_neigh):
                torsion.append((a, bond.atom1.idx, bond.atom2.idx, d))

        return torsion

    def torsion_degrees(self, a: int, b: int, c: int, d: int) -> float:
        return InternalCoordinates.calc_dehedral(
            self.atoms[a].coordinates,
            self.atoms[b].coordinates,
            self.atoms[c].coordinates,
            self.atoms[d].coordinates
        )

    @property
    def atom_attr_matrix(self) -> np.ndarray:
        return np.array([a.attrs for a in self._atoms])

    @property
    def atoms(self):
        return copy(self._atoms)

    @property
    def bonds(self):
        return copy(self._bonds)
        # return [edge['bond'] for edge in self._graph.edges]

    def build3d(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = None,
            steps: int = 500
    ):

        if forcefield is None:
            if self.has_metal:
                forcefield = 'UFF'
            else:
                forcefield = 'MMFF94'

        pmol = self.to_pybel_mol()
        pmol.make3D(forcefield, steps=steps)
        for atom, pa in zip(self.atoms, pmol.atoms):
            atom.coordinates = pa.coords

    def localopt(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = None,
            steps: int = 500
    ):

        if forcefield is None:
            if self.has_metal:
                forcefield = 'UFF'
            else:
                forcefield = 'MMFF'

        pmol = self.to_pybel_mol()
        pmol.localopt(forcefield, steps=steps)
        for atom, pa in zip(self.atoms, pmol.atoms):
            atom.coordinates = pa.coords

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

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([a.coordinates for a in self._atoms])

    @coordinates.setter
    def coordinates(self, value: np.ndarray) -> None:
        assert value.shape == (len(self.atoms), 3)
        for a, row in zip(self.atoms, value):
            a.coordinates = row

    def _create_atom(self, **kwargs): return Atom(self, **kwargs)

    def create_atoms(self, **kwargs) -> 'Atom':
        atom = self._create_atom(**kwargs)
        self._update_graph()
        return atom

    @property
    def element_counts(self):
        return Counter([a.symbol for a in self.atoms])

    @property
    def has_metal(self) -> bool:
        return bool([a for a in self._atoms if a.is_metal])

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
    def heavy_atoms(self) -> list["Atom"]:
        return [a for a in self._atoms if a.atomic_number != 1]

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

    def remove_hydrogens(self):
        self.remove_atoms([a for a in self._atoms if a.is_hydrogen])

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
    def weight(self):
        return sum(a.mass for a in self._atoms)

    @property
    def smiles(self) -> str:
        """ Return smiles string. """
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        return pb.readstring('mol2', pb.Molecule(mol2obmol(self)[0]).write('mol2')).write('can').split()[0]
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        # return pb.Molecule(mol2obmol(self)[0]).write().strip()

    @property
    def simple_cycles(self) -> list[list[int]]:
        graph = self.graph
        return nx.cycle_basis(graph)

    @property
    def sssr(self) -> list["Ring"]:
        return [Ring([self._atoms[i] for i in cycle]) for cycle in nx.cycle_basis(self.simple_graph)]

    @property
    def lssr(self):
        return None

    def to_pybel_mol(self) -> pb.Molecule:
        return pb.Molecule(mol2obmol(self)[0])

    def write(self, fmt='smi', filename=None, overwrite=False, opt=None):
        write_by_pybel(self, fmt, str(filename), overwrite, opt)


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

    def setattr(self, **kwargs):
        _attrs = copy(self._default_attrs)
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

        self.setattr(**kwargs)

    # def __getattr__(self, item):
    #     try:
    #         attr_idx = self._attrs_enumerator.index(item)
    #         return self._attrs_dict[item](self.attrs[attr_idx])
    #     except ValueError:
    #         # raise AttributeError(f"{item} is not an attribute of the {self.__class__.__name__}")
    #         return super().__getattribute__(item)
    #
    # def __setattr__(self, key, value):
    #     try:
    #         attr_idx = self._attrs_enumerator.index(key)
    #         self.attrs[attr_idx] = value
    #     except ValueError:
    #         super().__setattr__(key, value)

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.label})"

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

    def _add_hydrogens(self, num: int = None) -> list["Atom"]:
        if num is None:
            num = self.implicit_hydrogens - len([a for a in self.neighbours if a.atomic_number == 1])
        return [self._add_atom() for _ in range(num)]

    def add_hydrogen(self, num: int = None) -> list["Atom"]:
        hydrogens = self._add_hydrogens(num)
        getattr(self.mol, '_update_graph')()
        return hydrogens

    @property
    def bonds(self) -> list["Bond"]:
        # return [self.mol.bonds[i] for i in self.bonds_idx]
        edge_viewer = self.mol.graph.edges
        return [edge_viewer[u, v]['bond'] for u, v in edge_viewer(self.idx)]

    # @property
    # def bonds_idx(self) -> np.ndarray:
    #     # return np.array([self.mol.graph.edges[e]['idx'] for e in self.mol.graph.edges])
    #     if self.molecule.link_matrix.size > 0:
    #         return np.where(np.any(self.molecule.link_matrix == self.idx, axis=1))[0]
    #     else:
    #         return np.array([])

    def calc_implicit_hydrogens(self):
        self.implicit_hydrogens = self.valence - self.sum_bond_orders

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
    def idx(self):
        return self.mol.atoms.index(self)

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

    def setattr(self, **kwargs):
        coords = kwargs.get("coordinates", None)
        symbol = kwargs.get("symbol", None)

        if coords is not None:
            del kwargs["coordinates"]
            kwargs.update({'x': coords[0], 'y': coords[1], 'z': coords[2]})
        if isinstance(symbol, str):
            del kwargs["symbol"]
            kwargs['atomic_number'] = Atom._symbols.index(symbol)

        super().setattr(**kwargs)

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


class Bond(MolBlock):
    _attrs_dict = {
        'bond_order': float
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
        self.atom1 = atom1
        self.atom2 = atom2
        self.attrs = np.zeros(len(self._attrs_enumerator))

        self.setattr(**kwargs)

    def __eq__(self, other):
        return (other.atom1 == self.atom1 and other.atom2 == self.atom2) or (other.atom1 == self.atom2 and other.atom2 == self.atom1)

    def __hash__(self):
        return hash((self.atom1, self.atom2)) + hash((self.atom2, self.atom1))

    # def setattr(self, **kwargs):
    #     if kwargs.get('idx', None) is not None:
    #         kwargs.pop('idx')
    #     super().setattr(**kwargs)

    # def __getattr__(self, item):
    #     try:
    #         attr_idx = self._attrs_enumerator.index(item)
    #         return self._attrs_dict[item](self.attrs[attr_idx])
    #     except ValueError:
    #         # raise AttributeError(f"{item} is not an attribute of the {self.__class__.__name__}")
    #         return super().__getattribute__(item)

    # def __setattr__(self, key, value):
    #     try:
    #         attr_idx = self._attrs_enumerator.index(key)
    #         self.attrs[attr_idx] = value
    #     except ValueError:
    #         super().__setattr__(key, value)
    #
    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.label})"

    @property
    def mol(self) -> Molecule:
        return self.atom1.mol

    @property
    def idx(self) -> int:
        return self.mol.bonds.index(self)

    @property
    def a1idx(self) -> int:
        return self.atom1.idx

    @property
    def a2idx(self) -> int:
        return self.atom2.idx

    @property
    def bond_length(self) -> float:
        return float(np.linalg.norm(self.atom1.vector - self.atom2.vector))

    @property
    def label(self):
        return f"{self.atom1.label}{self._bond_order_symbol[self.bond_order]}{self.atom2.label}"

    @property
    def is_aromatic(self) -> bool:
        return self.atom1.is_aromatic and self.atom2.is_aromatic

    @is_aromatic.setter
    def is_aromatic(self, value: bool):
        self.atom1.is_aromatic = value
        self.atom2.is_aromatic = value


class Angle:
    def __init__(self, mol: Molecule, c: int, a: int, b: int):
        self.mol = mol
        self.c = c
        self.a = a
        self.b = b

    @property
    def degrees(self) -> float:
        return 0


class Ring:
    def __init__(self, atoms: list[Atom]):
        self.atoms = atoms

    def _check_ring(self):
        """ Check whether all atoms are in a same molecule """
        assert len({a.mol for a in self.atoms}) == 1

    @property
    def mol(self) -> Molecule:
        return self.atoms[0].mol


class Conformers:
    """ Representing an ensemble of conformers of a Molecule """

class Coordinates:
    """"""


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
    # a1, a2 = Atom(), Atom()
    # # a.molecule = Molecule()
    # print(a1.idx)
    # a1.symbol = "Am"
    # a2.symbol = "Eu"
    # print(a1.atomic_number)
    # print(a2.atomic_number)
    #
    # print(a1.mol is a2.mol)
    # a1.mol.add_atom(a2)
    # print(a1.mol is a2.mol)
    # b = a1.mol.add_bond(a1, a2, 2)
    #
    # print(b.idx)
    # print(b)
    # print(a1.coordinates)
    # print(a2.mol.weight)
    pass
