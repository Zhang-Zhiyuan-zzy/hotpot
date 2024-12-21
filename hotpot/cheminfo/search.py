"""
python v3.9.0
@Project: hotpot
@File   : search
@Auther : Zhiyuan Zhang
@Data   : 2024/12/13
@Time   : 16:27
"""
from abc import abstractproperty
from typing import Union, Sequence, Literal
import networkx as nx
from core import Molecule, Atom, Bond

def raise_not_implemented(self): raise NotImplemented(f"{self.__class__.__name__} not implemented")

class Query:
    _match_class = abstractproperty(raise_not_implemented)

    def __init__(self, **kwargs: set):
        self.kwargs = kwargs

    def __eq__(self, other: dict):
        if isinstance(other, self._match_class):
            other = {attr: getattr(other, attr) for attr in self.kwargs}

        return all(other[attr] in self.kwargs[attr] for attr in self.kwargs)


class QueryAtom(Query):
    _match_class = Atom

    def __init__(self, mol: "Substructure" = None, **attrs):
        self.mol = mol
        super().__init__(**attrs)

    @property
    def idx(self) -> int:
        return self.mol.query_atoms.index(self)

    @classmethod
    def from_atom(
            cls, atom: Atom,
            include_attrs: Sequence[str] = None,
            exclude_attrs: Sequence[Literal['atomic_number', 'is_aromatic']] = None
    ):
        kwargs = {
            "atomic_number": set(atom.atomic_number),
            "is_aromatic": set(atom.is_aromatic)
        }

        if include_attrs:
            for attr in include_attrs:
                kwargs[attr] = set(getattr(atom, attr))

        if exclude_attrs:
            for attr in exclude_attrs:
                kwargs.pop(attr)

        return cls(**kwargs)


class QueryBond(Query):

    _match_class = Bond

    def __init__(self, atom1: QueryAtom, atom2: QueryAtom, **attrs):
        self.atom1 = atom1
        self.atom2 = atom2
        super().__init__(**attrs)

    @property
    def a1idx(self) -> int:
        return self.atom1.idx

    @property
    def a2idx(self) -> int:
        return self.atom2.idx


class Substructure:
    """"""
    def __init__(self):
        self.query_atoms = []
        self.query_bonds = []
        self.query_graph = None

    def add_atom(self, atom: Union[Atom, QueryAtom]):
        """"""
        if isinstance(atom, Atom):
            atom = QueryAtom.from_atom(atom)

        self.query_atoms.append(atom)
        atom.mol = self

        return atom

    def add_bond(self, atom1: Union[int, QueryAtom], atom2: Union[int, QueryAtom], **bond_attrs):
        if isinstance(atom1, int):
            atom1 = self.query_atoms[atom1]
        if isinstance(atom2, int):
            atom2 = self.query_atoms[atom2]

        bond = QueryBond(atom1, atom2, **bond_attrs)
        self.query_bonds.append(bond)

        return bond

    def construct_graph(self):
        """"""
        self.query_graph = nx.Graph()
        self.query_graph.add_nodes_from([(a.idx, {'qa': a}) for a in self.query_atoms])
        self.query_graph.add_edges_from([(b.a1idx, b.a2idx, {'qb': b}) for b in self.query_bonds])


class Searcher:
    def __init__(self, substructure: Substructure):
        self.substructure = substructure
        self.substructure.construct_graph()
        self._graph_matcher = nx.Graph

    def search(self, mol: Molecule):
        """"""

    @staticmethod
    def _node_match(n1attr: dict, n2attr: dict):
        return
