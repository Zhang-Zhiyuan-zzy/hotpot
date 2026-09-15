"""Thin, test-only adapter around Hotpot's public SMARTS search API.

The adapter deliberately has no broad exception boundary.  It translates only
the exceptions which the active parser/reader use for expected rejection;
unexpected exceptions propagate to pytest as infrastructure or product bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import hotpot as hp
from hotpot.cheminfo.search.smarts import tokenize


@dataclass(frozen=True)
class QuerySummary:
    atom_count: int
    bond_count: int
    component_count: int
    atom_map_numbers: Tuple[Optional[int], ...]


@dataclass(frozen=True)
class ParseResult:
    accepted: bool
    phase: str
    error_code: Optional[str]
    error_position: Optional[int]
    diagnostic: Optional[str]
    ast_or_query_summary: Optional[QuerySummary]
    query: Optional[Any] = None


@dataclass(frozen=True)
class TargetResult:
    """Prepared target with an adapter-local to current-Hotpot-index map.

    ``atom_identity_map`` is not a cross-engine or persistent atom identity.
    """

    accepted: bool
    phase: str
    error_code: Optional[str]
    diagnostic: Optional[str]
    atom_identity_map: Tuple[int, ...]
    molecule: Optional[Any] = None


@dataclass(frozen=True)
class MatchResult:
    query_accepted: bool
    target_accepted: Optional[bool]
    matched: bool
    phase: str
    raw_embedding_count: int
    embeddings: Tuple[Tuple[int, ...], ...]
    unique_target_atom_sets: Tuple[Tuple[int, ...], ...]
    truncated: Optional[bool]
    error_code: Optional[str]
    diagnostic: Optional[str]


@dataclass(frozen=True)
class SearchResult:
    matched_record_ids: Tuple[str, ...]
    per_record_matches: Mapping[str, MatchResult]
    truncated: Optional[bool]
    errors: Mapping[str, str]
    strategy: str = "linear_scan"


def _reject_options(options: Optional[Mapping[str, Any]]) -> None:
    if options:
        names = ", ".join(sorted(options))
        raise NotImplementedError(
            "Hotpot's active SMARTS API does not expose adapter options: " + names
        )


def _error_position(diagnostic: str) -> Optional[int]:
    """Extract a parser-reported position, or preserve its absence as ``None``."""

    match = re.search(r"\bposition\s+(\d+)\b", diagnostic)
    return int(match.group(1)) if match else None


def _query_summary(query: Any) -> QuerySummary:
    graph = query.construct_graph()
    component_count = 0
    if graph.number_of_nodes():
        import networkx as nx

        component_count = nx.number_connected_components(graph)
    return QuerySummary(
        atom_count=len(query.query_atoms),
        bond_count=len(query.query_bonds),
        component_count=component_count,
        atom_map_numbers=tuple(atom.map_number for atom in query.query_atoms),
    )


def parse_smarts(text: str, options: Optional[Mapping[str, Any]] = None) -> ParseResult:
    """Tokenize and compile one SMARTS query through the active Hotpot API."""
    _reject_options(options)
    try:
        tokenize(text)
    except ValueError as exc:
        return ParseResult(
            False,
            "tokenize",
            "invalid_syntax",
            _error_position(str(exc)),
            str(exc),
            None,
        )

    try:
        query = hp.Substructure.from_smarts(text)
    except ValueError as exc:
        return ParseResult(
            False,
            "query_compile",
            "invalid_syntax",
            _error_position(str(exc)),
            str(exc),
            None,
        )
    except NotImplementedError as exc:
        return ParseResult(
            False,
            "query_compile",
            "unsupported_feature",
            _error_position(str(exc)),
            str(exc),
            None,
        )

    return ParseResult(
        True,
        "query_compile",
        None,
        None,
        None,
        _query_summary(query),
        query,
    )


def prepare_target(
    smiles: str, options: Optional[Mapping[str, Any]] = None
) -> TargetResult:
    """Load a SMILES using the same Open Babel-backed path as Hotpot users."""
    _reject_options(options)
    try:
        molecule = hp.read_mol(smiles, "smi")
    except (ValueError, StopIteration, OSError) as exc:
        return TargetResult(
            False, "target_prepare", "invalid_target", str(exc), (), None
        )
    return TargetResult(
        True,
        "target_prepare",
        None,
        None,
        tuple(range(len(molecule.atoms))),
        molecule,
    )


def _match_compiled_query(query: Any, molecule: Any) -> MatchResult:
    hits = hp.Searcher(query).search(molecule)
    embeddings = tuple(
        sorted(
            tuple(mapping[index] for index in range(len(query.query_atoms)))
            for hit in hits.hits
            for mapping in hit.mappings
        )
    )
    unique_target_atom_sets = tuple(
        sorted(tuple(sorted(hit.atom_indices)) for hit in hits.hits)
    )
    return MatchResult(
        True,
        True,
        bool(unique_target_atom_sets),
        "match",
        len(embeddings),
        embeddings,
        unique_target_atom_sets,
        None,
        None,
        None,
    )


def match_smarts(
    smarts: str,
    smiles: str,
    options: Optional[Mapping[str, Any]] = None,
) -> MatchResult:
    """Compile, prepare, and match while retaining all query-order embeddings."""
    _reject_options(options)
    parsed = parse_smarts(smarts)
    if not parsed.accepted:
        return MatchResult(
            False,
            None,
            False,
            parsed.phase,
            0,
            (),
            (),
            None,
            parsed.error_code,
            parsed.diagnostic,
        )

    target = prepare_target(smiles)
    if not target.accepted:
        return MatchResult(
            True,
            False,
            False,
            target.phase,
            0,
            (),
            (),
            None,
            target.error_code,
            target.diagnostic,
        )
    return _match_compiled_query(parsed.query, target.molecule)


def search_smarts(
    smarts: str,
    molecule_records: Iterable[Mapping[str, str]],
    options: Optional[Mapping[str, Any]] = None,
) -> SearchResult:
    """Run an explicit non-indexed, single-process linear-scan baseline.

    Records have ``id`` and ``smiles`` fields.  The query is compiled once and
    every target is prepared and matched independently.  This is intentionally
    not presented as an index, cache, parallel search, or optimized batch API.
    """
    _reject_options(options)
    parsed = parse_smarts(smarts)
    if not parsed.accepted:
        return SearchResult((), {}, None, {"<query>": parsed.error_code})

    matched_ids = []
    per_record: Dict[str, MatchResult] = {}
    errors: Dict[str, str] = {}
    for record in molecule_records:
        record_id = record["id"]
        target = prepare_target(record["smiles"])
        if target.accepted:
            result = _match_compiled_query(parsed.query, target.molecule)
        else:
            result = MatchResult(
                True,
                False,
                False,
                target.phase,
                0,
                (),
                (),
                None,
                target.error_code,
                target.diagnostic,
            )
            errors[record_id] = target.error_code
        per_record[record_id] = result
        if result.matched:
            matched_ids.append(record_id)
    return SearchResult(tuple(matched_ids), per_record, None, errors)


__all__ = [
    "MatchResult",
    "ParseResult",
    "QuerySummary",
    "SearchResult",
    "TargetResult",
    "match_smarts",
    "parse_smarts",
    "prepare_target",
    "search_smarts",
]
