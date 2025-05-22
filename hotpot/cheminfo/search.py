"""
python v3.9.0
@Project: hotpot
@File   : search
@Auther : Zhiyuan Zhang
@Data   : 2024/12/13
@Time   : 16:27
"""
from abc import abstractmethod
from typing import Union, Sequence, Literal, Container, Any, Iterable
import networkx as nx
from networkx.algorithms import isomorphism

from hotpot.cheminfo.core import Molecule, Atom, Bond

def raise_not_implemented(self): raise NotImplemented(f"{self.__class__.__name__} not implemented")

class Query:
    """
    Query serves as an abstract base class for defining query objects to perform attribute-based
    matching against other objects. Each derived class must implement the abstract `label` property
    and specify `_match_class` to define a compatible class for the `match` method's comparison.

    Query objects are instantiated with keyword arguments representing constraints on attributes,
    stored in the `kwargs` dictionary. These constraints are verified for proper types. The `match`
    method compares the Query instance against a compatible object to check if all constraints
    are satisfied based on their attributes.

    Attributes:
        kwargs (Dict[str, Set[Any]]): A mapping of attribute names to sets of acceptable values.
        _match_class (Type): Specifies the class with which this Query instance can perform
            attribute-based comparison.

    Methods:
        label: Abstract property, to be implemented by derived classes.
        match: Determines whether the given object meets all constraints defined in the Query instance.
        _check_kwargs_types: Ensures that all constraints in kwargs are valid containers, converting
            them to sets if necessary.

    Args:
        kwargs (Dict[str, Union[Container, Any]]): Keyword arguments representing constraints on
            attributes. Keys are attribute names, and values are containers of acceptable values.

    Raises:
        TypeError: If `match` is called with an incompatible object type, or if constraints in `kwargs`
            are not valid containers.
    """
    _match_class = None

    def __init__(self, **kwargs: Union[Container, Any]):
        self.kwargs = {n: set(v) for n, v in kwargs.items()}
        self._check_kwargs_types()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label}, {self.kwargs})"

    @property
    @abstractmethod
    def label(self):
        raise NotImplemented(f"{self.__class__.__name__} not implemented")

    def match(self, other):
        """
        Compares the current object to another object of a specific class to determine if
        all specified attributes meet the given conditions.

        Attributes:
            _match_class: The class type that the other object should match.
            kwargs: A dictionary where keys are attribute names and values are
                conditions those attributes must satisfy for a match.

        Parameters:
            self: Refers to the current instance of the class.
            other (self._match_class): An object of the expected type to compare with.

        Raises:
            TypeError: If the other object is not of the expected class.

        Returns:
            bool: True if all specified conditions for attributes are met, or if no
            conditions are specified. False if any attribute condition is not satisfied.
        """
        if not isinstance(other, self._match_class):
            raise TypeError(f"The {self.__class__.__name__} object should compare with "
                            f"{self._match_class.__name__} object, but got {type(other)} instead.")

        if not self.kwargs:
            return True

        try:
            return all(getattr(other, attr) in self.kwargs[attr] for attr in self.kwargs)
        except KeyError:
            return False

    def _check_kwargs_types(self):
        """
        Checks the types of the keyword arguments provided to the Query instance.
        Ensures that all attributes are of the type `Container` and that they
        are converted to sets if not already of type `set`.

        Raises:
            TypeError: If any attribute of the keyword arguments is not of
                       the type `Container`.
        """
        for attr, value in self.kwargs.items():
            if not isinstance(value, Container):
                raise TypeError("The attrs of Query should be Container")

            if not isinstance(value, set):
                self.kwargs[attr] = set(value)


class QueryAtom(Query):
    """
    Represents a query atom in a substructure search.

    This class is designed to represent an atom within a query substructure
    used for substructure searching in molecular structures. It extends the
    Query class and provides specialized methods and attributes to represent
    and work with query atoms.

    Attributes:
        _match_class: Internal class attribute representing the matching
        entity type, set to Atom.
        sub: The parent Substructure object to which this query atom belongs.

    Methods:
        label: Returns a string representation of the query atom index.
        idx: Returns the integer index of the query atom within the parent
        Substructure object's query_atoms list.
        from_atom: Creates and returns a QueryAtom instance from a given
        Atom object with optional filtering for attributes.
    """
    _match_class = Atom

    def __init__(self, sub: "Substructure" = None, **attrs):
        self.sub = sub
        super().__init__(**attrs)

    @property
    def label(self):
        return str(self.idx)

    @property
    def idx(self) -> int:
        return self.sub.query_atoms.index(self)

    @classmethod
    def from_atom(
            cls, atom: Atom,
            include_attrs: Sequence[str] = None,
            exclude_attrs: Sequence[Literal['atomic_number', 'is_aromatic']] = None
    ):
        """
        This method is a factory method that creates an instance of the class from the
        given Atom object. It extracts attributes from the Atom object and allows
        certain attributes to be included or excluded during the instance creation.

        Args:
            atom (Atom): The Atom object from which the instance attributes are derived.
            include_attrs (Sequence[str], optional): A sequence of attribute names
                to be included during the instance creation. If specified, these
                attributes will be retrieved from the Atom object.
            exclude_attrs (Sequence[Literal['atomic_number', 'is_aromatic']], optional):
                A sequence of attribute names to be excluded during the instance
                creation. These attributes, if present, will not be included in the
                resulting instance.

        Returns:
            object: Returns an instance of the class populated with the specified
            attributes derived from the Atom object.
        """
        attrs = {n: set(getattr(atom, n)) for n in Atom._attrs_enumerator}
        attrs.update({n: set(getattr(atom, n)) for n in include_attrs})
        if exclude_attrs:
            for attr in exclude_attrs:
                attrs.pop(attr)
        return cls(**attrs)
    

class QueryBond(Query):
    """
    Represents a query bond between two atoms in a molecular structure.

    A QueryBond is used to define a connection between two QueryAtoms in a molecular
    substructure. It ensures the atoms belong to the same substructure and provides
    access to their indices and the substructure itself.

    Attributes:
        atom1 (QueryAtom): The first atom in the bond.
        atom2 (QueryAtom): The second atom in the bond.

    Methods:
        label: Returns the bond label as a formatted string.
        a1idx: Returns the index of the first atom.
        a2idx: Returns the index of the second atom.
        sub: Returns the substructure the bond belongs to.
    """
    _match_class = Bond

    def __init__(self, atom1: QueryAtom, atom2: QueryAtom, **attrs):
        assert atom1.sub is atom2.sub
        self.atom1 = atom1
        self.atom2 = atom2
        super().__init__(**attrs)

    @property
    def label(self):
        return f"{self.atom1.idx}-{self.atom2.idx}"

    @property
    def a1idx(self) -> int:
        return self.atom1.idx

    @property
    def a2idx(self) -> int:
        return self.atom2.idx

    @property
    def sub(self):
        return self.atom1.sub


class Substructure:
    """
    Establishes and manages a chemical substructure composed of atoms, bonds,
    and their topological representation as a graph.

    The Substructure class serves to define a chemical substructure containing atoms and bonds,
    with methods to build and interact with its components. It facilitates the creation of
    substructure queries, enabling operations such as adding atoms and bonds, importing from
    SMARTS format, and constructing a graph representation of the substructure.

    Attributes:
    query_atoms: List of query atoms belonging to this substructure.
    query_bonds: List of query bonds that define connections between atoms.
    query_graph: Graph representation of the substructure for topological analysis.
    """
    def __init__(self):
        self.query_atoms = []
        self.query_bonds = []
        self.query_graph = None  # 确保这里初始化图对象

    @classmethod
    def from_SMARTS(cls, smarts: str):
        # TODO: Wu 将SMILES转化为Substructure的结构
        ...

    def add_atom(self, atom_query: Union[Atom, QueryAtom]):
        """
        Adds an atom or query atom to the collection of query atoms. Converts a given
        Atom instance to a QueryAtom if required and associates it with the collection.

        Parameters:
            atom_query (Union[Atom, QueryAtom]): The atom or query atom to be added. If
                an Atom is provided, it will be converted into a QueryAtom.

        Returns:
            QueryAtom: The added or converted QueryAtom.
        """
        if isinstance(atom_query, Atom):
            atom_query = QueryAtom.from_atom(atom_query)

        self.query_atoms.append(atom_query)
        atom_query.sub = self

        return atom_query

    def add_bond(self, atom1: Union[int, QueryAtom], atom2: Union[int, QueryAtom], **bond_attrs):
        """
        Adds a bond between two QueryAtom objects or their indices within the context
        of a molecular query. This method creates a new QueryBond instance
        representing the bond and appends it to the query's list of bonds.

        Parameters:
            atom1 (int | QueryAtom): A QueryAtom object or the index of a QueryAtom
                                     in the `query_atoms` list.
            atom2 (int | QueryAtom): A QueryAtom object or the index of a QueryAtom
                                     in the `query_atoms` list.
            bond_attrs: Additional keyword attributes for the bond.

        Returns:
            QueryBond: The created bond object that was added to the `query_bonds`.

        Raises:
            AssertionError: If the two specified atoms are not within the same
                            molecular query context as the current object.
        """
        if isinstance(atom1, int):
            atom1 = self.query_atoms[atom1]
        if isinstance(atom2, int):
            atom2 = self.query_atoms[atom2]

        assert atom1.sub is atom2.sub is self

        bond = QueryBond(atom1, atom2, **bond_attrs)
        self.query_bonds.append(bond)

        return bond

    def construct_graph(self):
        """
        Constructs and returns a graph representation of the query atoms and bonds.

        The graph is created using the NetworkX library. Nodes in the graph
        represent query atoms, while edges in the graph represent query bonds.
        Each node is associated with its corresponding query atom, and each
        edge is associated with its corresponding query bond.

        Returns:
            Graph: A NetworkX Graph object representing the query atoms and
            bonds.
        """
        self.query_graph = nx.Graph()
        self.query_graph.add_nodes_from([(a.idx, {'qa': a}) for a in self.query_atoms])
        self.query_graph.add_edges_from([(b.a1idx, b.a2idx, {'qb': b}) for b in self.query_bonds])
        return self.query_graph


class Searcher:
    """
    Searcher is a utility class designed to identify substructures within molecular
    graphs.

    The main purpose of this class is to locate occurrences of a specific substructure
    within a given molecular graph by leveraging graph isomorphism techniques. Users can
    utilize this class to perform substructure search tasks in chemical informatics or
    related fields. Substructures are detected based on atom and bond properties, and
    strict matching ensures reliability. The results are returned as a Hits object, which
    encapsulates substructure matches.

    Attributes:
        substructure (Substructure): A predefined substructure pattern that will be
        searched for within molecular graphs.
    """
    def __init__(self, substructure: "Substructure"):
        self.substructure = substructure

    def search(self, mol: Molecule) -> "Hits":
        """
        Search for substructures within a given molecular graph.

        This method takes a molecular graph and identifies occurrences of a pre-defined
        substructure pattern within it. It uses graph isomorphism for matching,
        ensuring accurate detection of substructure instances. Substructures are
        detected based on properties of atoms and bonds defined in the molecular graph.

        Args:
            mol (Molecule): The molecular graph in which to search for the
            substructure. It must include an atom-bond graph representation.

        Returns:
            Hits: An object representing the matched substructure occurrences,
            including the input molecule and the corresponding substructure
            graph matches.
        """
        return Hits(
            mol, self.substructure,
            isomorphism.GraphMatcher(
                mol.atom_bond_graph,
                self.substructure.construct_graph(),
                node_match=self._node_match,
                edge_match=self._edge_match
            )
        )

    @staticmethod
    def _node_match(mol_node: dict, query_node: dict) -> bool:
        query_atom: QueryAtom = query_node.get('qa')  # QueryAtom in substructure.
        atom: Atom = mol_node.get('atom')   # Atom in Molecule.

        # 防止 sub_atom 或 mol_atom 为空
        if not query_atom or not atom:
            raise AttributeError('Not get QueryAtom in substructure or Atom in molecule!')

        return query_atom.match(atom)

    @staticmethod
    def _edge_match(mol_edge: dict, query_edge: dict) -> bool:
        query_bond = query_edge.get('qb')  # 子结构中的 QueryBond
        bond = mol_edge.get('bond')

        if not query_bond or bond is None:
            raise AttributeError('Not get QueryBond in substructure or Bond in molecule!')

        return query_bond.match(bond)


class Hits:
    """
    Represents a collection of matching substructures (hits) found in a molecular graph.

    This class provides a way to store and interact with matches of a substructure
    (sub) within a molecular graph (mol) based on a specified graph matcher. It uses
    the subgraph monomorphisms provided by the graph matcher to identify and store
    hits, which are represented by `Hit` objects.

    Attributes:
        sub: The substructure being searched for in the molecular graph.
        mol: The molecular graph being analyzed for substructure matches.
        graph_matcher: A graph matcher object responsible for finding subgraph
            isomorphisms.
        hits: A list of `Hit` objects representing all subgraph matches found.

    Raises:
        None
    """
    def __init__(self, sub, mol, graph_matcher, get_hit: bool = True):
        self.sub = sub
        self.mol = mol
        self.graph_matcher = graph_matcher
        self.get_hit = get_hit

        self._nodes_indices = list(self._get_nodes_set())
        self._hits = None

    @property
    def hits(self) -> list["Hit"]:
        if self._hits is None:
            self._hits = [Hit(self.sub, self.mol, ai) for ai in self._get_nodes_set()]

        return self._hits

    def _get_nodes_set(self):
        return set(frozenset(ai.keys()) for ai in self.graph_matcher.subgraph_monomorphisms_iter())

    def __iter__(self):
        return iter(self.hits) if self.get_hit else iter(self._nodes_indices)

    def __getitem__(self, item: int) -> Union[frozenset[int], "Hit"]:
        return self.hits[item] if self.get_hit else self._nodes_indices[item]

    def __len__(self):
        return len(self._nodes_indices)

    def __bool__(self):
        return bool(self._nodes_indices)

    def __contains__(self, item: Union[Iterable[int], "Hit"]):
        if isinstance(item, Iterable):
            return frozenset(item) in self._nodes_indices
        else:
            return item in self.hits


class Hit:
    """
    Represent a match or "hit" within a molecule based on specific substructure search.

    The Hit class encapsulates the results of a substructure search within a molecule.
    It stores the matched substructure atoms and bonds. This class also provides an
    interface to access the atoms and bonds that form the substructure match. It is
    particularly useful for cheminformatics tasks that involve molecule comparisons or
    pattern recognition.

    Attributes:
        mol: The molecule object where the substructure is found.
        sub: The substructure object that was matched.
        atom_indices: The list of indices corresponding to the atoms in the molecule
            that participate in the substructure match, preserving their order.
        atoms: The list of Atom objects derived from `mol` corresponding to the
            matched atom_indices.
        bonds: The list of Bond objects within the matched substructure, determined
            by considering the atoms connected and filtering bonds in the molecule.
    """
    def __init__(self, mol, sub, atom_indices):
        self.mol = mol
        self.sub = sub
        self.atom_indices = atom_indices

        self.atoms = [self.mol.atoms[i] for i in self.atom_indices]
        self.bonds = [b for b in self.mol.bonds if b.atom1 in self.atoms and b.atom2 in self.atoms]

