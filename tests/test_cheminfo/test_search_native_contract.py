"""Frozen public and behavioral contracts for the native-search migration."""

import inspect
import types
import typing
from types import MappingProxyType

import pytest

import hotpot as hp
import hotpot.cheminfo as cheminfo
import hotpot.cheminfo.search as search_package
from hotpot.cheminfo.search import smarts
from hotpot.cheminfo.search import search as search_module
from hotpot.cheminfo.search._smarts_syntax import TokenType
from hotpot.cheminfo.search.errors import (
    SmartsSyntaxError,
    UnsupportedSmartsError,
)
from hotpot.cheminfo.search.search import (
    Hit,
    Hits,
    Query,
    QueryAtom,
    QueryBond,
    Searcher,
    Substructure,
)
from hotpot.cheminfo.search.semantics import SmartsSemantics


PUBLIC_SEARCH_NAMES = (
    "Query",
    "QueryAtom",
    "QueryBond",
    "Substructure",
    "Searcher",
    "Hits",
    "Hit",
    "SmartsSemantics",
    "SmartsSyntaxError",
    "UnsupportedSmartsError",
)


def _clean_annotation_text(annotation):
    """Keep public types visible without coupling to defining module paths."""
    for prefix in (
        "hotpot.cheminfo.search.search.",
        "hotpot.cheminfo.search.semantics.",
        "hotpot.cheminfo.core.",
        "typing.",
    ):
        annotation = annotation.replace(prefix, "")
    return annotation


def _annotation_text(annotation):
    if isinstance(annotation, str):
        return repr(annotation)
    if annotation is Ellipsis:
        return "..."

    origin = typing.get_origin(annotation)
    union_type = getattr(types, "UnionType", object())
    if origin in (typing.Union, union_type):
        arguments = ", ".join(
            _annotation_text(argument) for argument in typing.get_args(annotation)
        )
        return f"Union[{arguments}]"

    if origin is not None:
        formatted = _clean_annotation_text(inspect.formatannotation(annotation))
        annotation_arguments = typing.get_args(annotation)
        if not annotation_arguments:
            return formatted
        base = formatted.split("[", 1)[0]
        arguments = ", ".join(
            _annotation_text(argument) for argument in annotation_arguments
        )
        return f"{base}[{arguments}]"

    return _clean_annotation_text(inspect.formatannotation(annotation))


def _signature_text(obj):
    """Render signatures canonically across supported Python versions."""
    signature = inspect.signature(obj)
    parts = []
    positional_only = 0
    has_varargs = False
    keyword_marker_added = False

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only += 1
        elif positional_only:
            parts.insert(positional_only, "/")
            positional_only = 0

        if (
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not has_varargs
            and not keyword_marker_added
        ):
            parts.append("*")
            keyword_marker_added = True

        prefix = ""
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
            has_varargs = True
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            prefix = "**"

        text = f"{prefix}{parameter.name}"
        has_annotation = parameter.annotation is not inspect.Parameter.empty
        if has_annotation:
            text += f": {_annotation_text(parameter.annotation)}"
        if parameter.default is not inspect.Parameter.empty:
            separator = " = " if has_annotation else "="
            text += f"{separator}{parameter.default!r}"
        parts.append(text)

    if positional_only:
        parts.append("/")

    rendered = f"({', '.join(parts)})"
    if signature.return_annotation is not inspect.Signature.empty:
        rendered += f" -> {_annotation_text(signature.return_annotation)}"
    return rendered


SIGNATURES = (
    (Query, "(**kwargs: Union[Container, Any])"),
    (Query.match, "(self, obj)"),
    (Query.__repr__, "(self)"),
    (Query.label.fget, "(self)"),
    (
        QueryAtom,
        "(sub: 'Substructure' = None, map_number: int = None, **attrs)",
    ),
    (
        QueryAtom.from_atom,
        "(atom: Atom, include_attrs: Sequence[str] = None, "
        "exclude_attrs: Sequence[Literal['atomic_number', 'is_aromatic']] = None)",
    ),
    (QueryAtom.map_number.fget, "(self)"),
    (QueryAtom.label.fget, "(self)"),
    (QueryAtom.idx.fget, "(self) -> int"),
    (QueryBond, "(atom1: QueryAtom, atom2: QueryAtom, **attrs)"),
    (QueryBond.match, "(self, obj)"),
    (QueryBond.label.fget, "(self)"),
    (QueryBond.a1idx.fget, "(self) -> int"),
    (QueryBond.a2idx.fget, "(self) -> int"),
    (QueryBond.sub.fget, "(self)"),
    (Substructure, "()"),
    (
        Substructure.from_mol,
        "(mol: Molecule, addition_atom_attr: dict[int, dict[str, set]] = None, "
        "addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]] = None) "
        "-> 'Substructure'",
    ),
    (
        Substructure.from_smarts,
        "(smarts: str, *, semantics: SmartsSemantics = "
        "<SmartsSemantics.FULL_GRAPH: 'full_graph'>) -> 'Substructure'",
    ),
    (
        Substructure.from_smiles,
        "(smiles: str, addition_atom_attr: dict[int, dict[str, set]] = None, "
        "addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]] = None)",
    ),
    (Substructure.add_atom, "(self, atom_query: Union[Atom, QueryAtom])"),
    (
        Substructure.add_bond,
        "(self, atom1: Union[int, QueryAtom], atom2: Union[int, QueryAtom], "
        "**bond_attrs)",
    ),
    (Substructure.construct_graph, "(self)"),
    (Substructure.__repr__, "(self)"),
    (Searcher, "(substructure: 'Substructure')"),
    (Searcher.has_match, "(self, mol: Molecule) -> bool"),
    (
        Searcher.iter_mappings,
        "(self, mol: Molecule, max_matches: int = None) -> Iterator[mappingproxy]",
    ),
    (
        Searcher.search,
        "(self, mol: Molecule, max_matches: int = None) -> 'Hits'",
    ),
    (
        Hits,
        "(sub, mol, graph_matcher, get_hit: bool = True, max_matches: int = None)",
    ),
    (Hits.__iter__, "(self)"),
    (
        Hits.__getitem__,
        "(self, item: int) -> Union[frozenset[int], ForwardRef('Hit')]",
    ),
    (Hits.__len__, "(self)"),
    (Hits.__bool__, "(self)"),
    (
        Hits.__contains__,
        "(self, item: Union[Iterable[int], ForwardRef('Hit')])",
    ),
    (Hits.hits.fget, "(self) -> list['Hit']"),
    (Hit, "(mol, sub, atom_indices, mappings=())"),
    (Hit.mapped_bonds, "(self, mapping_index: int = 0)"),
    (
        Hit.mapped_atom_indices,
        "(self, query_idx: int) -> tuple[int, ...]",
    ),
    (Hit.mapped_atoms, "(self, query_idx: int) -> tuple[Atom, ...]"),
    (Hit.induced_bonds.fget, "(self)"),
    (Hit.mappings.fget, "(self)"),
    (SmartsSemantics.rings_for, "(self, atom)"),
    (SmartsSemantics.bonds_for, "(self, atom)"),
    (SmartsSemantics.degree, "(self, atom) -> 'int'"),
    (SmartsSemantics.connectivity, "(self, atom) -> 'int'"),
    (SmartsSemantics.valence, "(self, atom) -> 'float'"),
    (
        smarts.substructure_from_smarts,
        "(smarts: 'str', *, semantics: 'SmartsSemantics' = "
        "<SmartsSemantics.FULL_GRAPH: 'full_graph'>) -> 'Substructure'",
    ),
    (
        smarts.parse_bracket_atom,
        "(expr_text: 'str', *, semantics: 'SmartsSemantics' = "
        "<SmartsSemantics.FULL_GRAPH: 'full_graph'>) -> 'Dict[str, object]'",
    ),
    (smarts.tokenize, "(smarts: 'str') -> 'List[Tuple[TokenType, str]]'"),
    (
        hp.Molecule.search_substructure,
        "(self, smarts: str, *, semantics='full_graph')",
    ),
)


@pytest.mark.parametrize(("obj", "expected"), SIGNATURES)
def test_public_search_signatures(obj, expected):
    assert _signature_text(obj) == expected


def test_public_exports_descriptors_enums_and_errors():
    assert search_module.__all__ == list(PUBLIC_SEARCH_NAMES[:7])

    for name in PUBLIC_SEARCH_NAMES:
        canonical = getattr(search_package, name)
        assert getattr(cheminfo, name) is canonical
        assert getattr(hp, name) is canonical

    for cls, method_name in (
        (QueryAtom, "from_atom"),
        (Substructure, "from_mol"),
        (Substructure, "from_smarts"),
        (Substructure, "from_smiles"),
    ):
        assert isinstance(inspect.getattr_static(cls, method_name), classmethod)

    for cls, property_name in (
        (Query, "label"),
        (QueryAtom, "map_number"),
        (QueryAtom, "label"),
        (QueryAtom, "idx"),
        (QueryBond, "label"),
        (QueryBond, "a1idx"),
        (QueryBond, "a2idx"),
        (QueryBond, "sub"),
        (Hits, "hits"),
        (Hit, "induced_bonds"),
        (Hit, "mappings"),
    ):
        descriptor = inspect.getattr_static(cls, property_name)
        assert isinstance(descriptor, property)
        assert descriptor.fset is None

    assert Query.label.fget.__isabstractmethod__
    query = Query()
    with pytest.raises(NotImplementedError):
        _ = query.label

    assert [(member.name, member.value) for member in SmartsSemantics] == [
        ("FULL_GRAPH", "full_graph"),
        ("LIGAND_SKELETON", "ligand_skeleton"),
    ]
    assert [(member.name, member.value) for member in TokenType] == [
        ("ATOM", "ATOM"),
        ("BRACKET", "BRACKET"),
        ("BOND", "BOND"),
        ("BRANCH_L", "BRANCH_L"),
        ("BRANCH_R", "BRANCH_R"),
        ("RING", "RING"),
        ("DOT", "DOT"),
    ]
    assert smarts.TokenType is TokenType
    assert SmartsSyntaxError.__bases__ == (ValueError,)
    assert UnsupportedSmartsError.__bases__ == (NotImplementedError,)


@pytest.mark.parametrize("molecule", [hp.Molecule(), hp.read_mol("C", "smi")])
def test_empty_query_has_one_empty_embedding(molecule):
    searcher = Searcher(Substructure())

    assert searcher.has_match(molecule)
    mappings = tuple(searcher.iter_mappings(molecule))
    assert len(mappings) == 1
    assert isinstance(mappings[0], MappingProxyType)
    assert dict(mappings[0]) == {}

    hits = searcher.search(molecule)
    assert not hits.truncated
    assert len(hits) == 1
    assert hits[0].atom_indices == frozenset()
    assert tuple(map(dict, hits[0].mappings)) == ({},)

    bounded_mappings = searcher.iter_mappings(molecule, max_matches=0)
    assert tuple(bounded_mappings) == ()
    assert bounded_mappings.yielded == 0
    assert bounded_mappings.truncated

    bounded_hits = searcher.search(molecule, max_matches=0)
    assert not bounded_hits
    assert bounded_hits.truncated


def test_duplicate_query_atom_retains_sparse_query_mapping_keys():
    query = Substructure()
    duplicate = QueryAtom(atomic_number={6})
    query.add_atom(duplicate)
    query.add_atom(duplicate)
    tail = query.add_atom(QueryAtom(atomic_number={6}))
    query.add_bond(duplicate, tail, bond_order={1})

    assert [atom.idx for atom in query.query_atoms] == [0, 0, 2]
    assert set(query.construct_graph()) == {0, 2}

    mappings = tuple(Searcher(query).iter_mappings(hp.read_mol("CC", "smi")))
    assert {tuple(mapping) for mapping in mappings} == {(0, 2)}
    assert {tuple(mapping.items()) for mapping in mappings} == {
        ((0, 0), (2, 1)),
        ((0, 1), (2, 0)),
    }
    with pytest.raises(TypeError):
        mappings[0][0] = 99


def test_duplicate_query_edge_uses_last_predicate_but_hit_maps_all_edges():
    query = Substructure()
    first_atom = query.add_atom(QueryAtom(atomic_number={6}))
    second_atom = query.add_atom(QueryAtom(atomic_number={6}))
    ignored_edge = query.add_bond(first_atom, second_atom, bond_order={2})
    effective_edge = query.add_bond(first_atom, second_atom, bond_order={1})

    graph = query.construct_graph()
    assert graph.edges[0, 1]["qb"] is effective_edge
    assert graph.edges[0, 1]["qb"] is not ignored_edge

    single_bond_molecule = hp.read_mol("CC", "smi")
    hit = Searcher(query).search(single_bond_molecule)[0]
    target_bond = single_bond_molecule.bond(0, 1)
    assert hit.mapped_bonds() == [target_bond, target_bond]
    assert hit.bonds == [target_bond, target_bond]
    assert not Searcher(query).has_match(hp.read_mol("C=C", "smi"))


def test_programmatic_self_loop_requires_a_target_self_loop():
    target = hp.Molecule()
    target.create_atom(atomic_number=6, implicit_hydrogens=0)
    target_loop = target.add_bond(0, 0, 1.0)

    query = Substructure()
    query_atom = query.add_atom(QueryAtom(atomic_number={6}))
    query.add_bond(query_atom, query_atom, bond_order={1.0})
    searcher = Searcher(query)

    assert set(query.construct_graph().edges) == {(0, 0)}
    assert searcher.has_match(target)
    assert {tuple(mapping.items()) for mapping in searcher.iter_mappings(target)} == {
        ((0, 0),)
    }
    assert searcher.search(target)[0].mapped_bonds() == [target_loop]

    target_without_loop = hp.Molecule()
    target_without_loop.create_atom(atomic_number=6, implicit_hydrogens=0)
    assert not searcher.has_match(target_without_loop)


def test_max_matches_bounds_and_iterator_state_without_order_contract():
    molecule = hp.read_mol("CCC", "smi")
    searcher = Searcher(Substructure.from_smarts("C"))

    zero = searcher.iter_mappings(molecule, max_matches=0)
    assert zero.max_matches == 0
    assert zero.yielded == 0
    assert not zero.truncated
    with pytest.raises(StopIteration):
        next(zero)
    assert zero.yielded == 0
    assert zero.truncated

    below_available = searcher.iter_mappings(molecule, max_matches=2)
    assert below_available.max_matches == 2
    first = next(below_available)
    assert below_available.yielded == 1
    assert not below_available.truncated
    second = next(below_available)
    assert below_available.yielded == 2
    assert below_available.truncated
    assert {first[0], second[0]} <= {0, 1, 2}
    assert first[0] != second[0]
    with pytest.raises(StopIteration):
        next(below_available)
    assert below_available.truncated

    exact = searcher.iter_mappings(molecule, max_matches=3)
    assert {mapping[0] for mapping in exact} == {0, 1, 2}
    assert exact.yielded == 3
    assert not exact.truncated

    above_available = searcher.iter_mappings(molecule, max_matches=4)
    assert {mapping[0] for mapping in above_available} == {0, 1, 2}
    assert above_available.yielded == 3
    assert not above_available.truncated

    for limit, expected_count, expected_truncated in (
        (0, 0, True),
        (2, 2, True),
        (3, 3, False),
        (4, 3, False),
    ):
        hits = searcher.search(molecule, max_matches=limit)
        assert len(hits) == expected_count
        assert hits.truncated is expected_truncated


@pytest.mark.parametrize(
    ("invalid_limit", "error"),
    ((True, TypeError), ("1", TypeError), (1.0, TypeError), (-1, ValueError)),
)
@pytest.mark.parametrize("method_name", ("iter_mappings", "search"))
def test_max_matches_rejects_invalid_limits(invalid_limit, error, method_name):
    searcher = Searcher(Substructure.from_smarts("C"))
    method = getattr(searcher, method_name)
    with pytest.raises(error):
        method(hp.read_mol("C", "smi"), max_matches=invalid_limit)


class _OneShotMappingProvider:
    def __init__(self, mappings):
        self.mappings = mappings
        self.calls = 0
        self.emitted = 0

    def subgraph_monomorphisms_iter(self):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("mapping provider was consumed more than once")
        for mapping in self.mappings:
            self.emitted += 1
            yield mapping


def test_hits_consumes_one_shot_target_to_query_provider_once():
    molecule = hp.read_mol("CCC", "smi")
    query = Substructure.from_smarts("CC")
    provider = _OneShotMappingProvider(
        (
            {2: 0, 1: 1},
            {1: 0, 0: 1},
            {0: 0, 1: 1},
        )
    )

    hits = Hits(query, molecule, provider, max_matches=1)

    assert provider.calls == 1
    assert provider.emitted == 2
    assert hits.truncated
    assert len(hits) == 1
    assert hits[0].atom_indices == frozenset({1, 2})
    assert tuple(map(dict, hits[0].mappings)) == ({0: 2, 1: 1},)

    assert list(hits) == list(hits)
    assert provider.calls == 1
    assert provider.emitted == 2
