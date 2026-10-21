"""Private adapters for Hotpot's native topology-search executor."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Iterable, Sequence

from .query_ir import (
    _compile_query,
    _query_derived_requirements,
    _query_object_requirements,
)
from .semantics import SmartsSemantics


__all__ = ()


@dataclass(frozen=True)
class _PreparedQuery:
    """Immutable ownership record for one compiled native query snapshot."""

    query_spec: dict
    needs_atom_objects: bool
    needs_bond_objects: bool
    derived_requirements: dict


def _native_module() -> ModuleType:
    try:
        return import_module("hotpot.cheminfo.search._search_native")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.search native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc


def _synthetic_problem(
    query_keys: Sequence[int],
    query_edges: Sequence[tuple[int, int]],
    target_keys: Sequence[int],
    target_edges: Sequence[tuple[int, int]],
    node_compatible: Sequence[Sequence[bool]],
    edge_compatible: Sequence[Sequence[bool]],
):
    return _native_module().SyntheticProblem(
        query_keys,
        query_edges,
        target_keys,
        target_edges,
        node_compatible,
        edge_compatible,
    )


def _compile_problem(substructure, mol):
    source = next(_iter_mapping_sources((_prepare_query(substructure),), mol))
    return source._problem


def _prepare_query(substructure) -> _PreparedQuery:
    """Snapshot one mutable ``Substructure`` as reusable private query IR."""
    query_spec = _compile_query(substructure)
    atom_objects, bond_objects = _query_object_requirements(query_spec)
    return _PreparedQuery(
        query_spec=query_spec,
        needs_atom_objects=atom_objects,
        needs_bond_objects=bond_objects,
        derived_requirements=_query_derived_requirements(query_spec),
    )


def _iter_mapping_sources(queries, mol):
    """Run ordered prepared queries against one shared target snapshot."""
    queries = tuple(queries)
    if not queries:
        return

    target_spec = _pack_target_snapshot(
        mol,
        needs_atom_objects=any(query.needs_atom_objects for query in queries),
        needs_bond_objects=any(query.needs_bond_objects for query in queries),
        derived_requirements=_merge_derived_requirements(
            tuple(query.derived_requirements for query in queries)
        ),
    )
    native_module = _native_module()
    target = native_module.NativeTarget(target_spec)
    for query in queries:
        yield _NativeMappingSource(
            native_module.NativeProblem(query.query_spec, target)
        )


def _merge_derived_requirements(requirements):
    return {
        name: tuple(
            dict.fromkeys(
                value
                for requirement in requirements
                for value in requirement[name]
            )
        )
        for name in ("full", "ligand", "atom_fields")
    } | {
        "bond_aromatic": any(
            requirement["bond_aromatic"] for requirement in requirements
        ),
        "total_h": any(requirement["total_h"] for requirement in requirements),
    }


def _pack_target(mol, *, needs_atom_objects, needs_bond_objects):
    """Snapshot a molecule in the native executor's target schema."""
    return _pack_target_snapshot(
        mol,
        needs_atom_objects=needs_atom_objects,
        needs_bond_objects=needs_bond_objects,
        derived_requirements={
            "full": (
                "degree",
                "connectivity",
                "valence",
                "ring_count",
                "ring_sizes",
            ),
            "ligand": (
                "degree",
                "connectivity",
                "valence",
                "ring_count",
                "ring_sizes",
            ),
            "atom_fields": (
                "atomic_number",
                "formal_charge",
                "is_aromatic",
                "group",
                "is_lanthanide",
                "is_actinide",
                "is_metal",
            ),
            "bond_aromatic": True,
            "total_h": True,
        },
    )


def _pack_target_snapshot(
    mol, *, needs_atom_objects, needs_bond_objects, derived_requirements
):
    atoms_by_key = {}
    for atom in mol.atoms:
        atoms_by_key[atom.idx] = atom

    bonds_by_edge = {}
    for bond in mol.bonds:
        edge = tuple(sorted((bond.a1idx, bond.a2idx)))
        bonds_by_edge[edge] = bond

    keys = tuple(sorted(atoms_by_key))
    edges = tuple(sorted(bonds_by_edge))
    needs_full_rings = (
        _needs_rings(derived_requirements["full"])
        or derived_requirements["bond_aromatic"]
    )
    full_rings = mol.rings if needs_full_rings else ()
    ligand_rings = (
        mol.ligand_rings
        if _needs_rings(derived_requirements["ligand"])
        else ()
    )
    full_ring_sizes = (
        _pack_ring_sizes(full_rings, keys)
        if _needs_rings(derived_requirements["full"])
        else {}
    )
    ligand_ring_sizes = (
        _pack_ring_sizes(ligand_rings, keys)
        if _needs_rings(derived_requirements["ligand"])
        else {}
    )
    aromatic_edges = (
        _pack_aromatic_edges(full_rings)
        if derived_requirements["bond_aromatic"]
        else set()
    )
    return {
        "keys": keys,
        "edges": edges,
        "atoms": tuple(
            _pack_atom(
                atoms_by_key[key],
                needs_atom_objects,
                derived_requirements,
                full_ring_sizes.get(key, ()),
                ligand_ring_sizes.get(key, ()),
            )
            for key in keys
        ),
        "bonds": tuple(
            _pack_bond(
                bonds_by_edge[edge],
                needs_bond_objects,
                edge in aromatic_edges,
            )
            for edge in edges
        ),
    }


def _pack_atom(
    atom,
    needs_object,
    derived_requirements,
    full_ring_sizes,
    ligand_ring_sizes,
):
    fields = {
        "atomic_number": int(atom.atomic_number),
        "formal_charge": int(atom.formal_charge),
        "is_aromatic": bool(atom.is_aromatic),
        "group": (
            int(atom.group)
            if "group" in derived_requirements["atom_fields"]
            else 0
        ),
        "is_lanthanide": (
            bool(atom.is_lanthanide)
            if "is_lanthanide" in derived_requirements["atom_fields"]
            else False
        ),
        "is_actinide": (
            bool(atom.is_actinide)
            if "is_actinide" in derived_requirements["atom_fields"]
            else False
        ),
        "is_metal": (
            bool(atom.is_metal)
            if "is_metal" in derived_requirements["atom_fields"]
            else False
        ),
        "implicit_hydrogens": int(atom.implicit_hydrogens),
        "explicit_hydrogens": (
            int(atom.explicit_hydrogens)
            if derived_requirements["total_h"]
            else 0
        ),
    }
    return {
        "fields": fields,
        "full": _pack_semantics(
            atom,
            SmartsSemantics.FULL_GRAPH,
            derived_requirements["full"],
            full_ring_sizes,
        ),
        "ligand": _pack_semantics(
            atom,
            SmartsSemantics.LIGAND_SKELETON,
            derived_requirements["ligand"],
            ligand_ring_sizes,
        ),
        "object": atom if needs_object else None,
    }


def _pack_semantics(atom, semantics, required_fields, ring_sizes):
    return {
        "degree": (
            int(semantics.degree(atom)) if "degree" in required_fields else 0
        ),
        "connectivity": (
            int(semantics.connectivity(atom))
            if "connectivity" in required_fields
            else 0
        ),
        "valence": (
            _native_number(semantics.valence(atom))
            if "valence" in required_fields
            else 0.0
        ),
        "ring_count": (
            len(ring_sizes) if "ring_count" in required_fields else 0
        ),
        "ring_sizes": ring_sizes if "ring_sizes" in required_fields else (),
    }


def _pack_bond(bond, needs_object, is_aromatic):
    return {
        "fields": {
            "bond_order": float(bond.bond_order),
            "is_aromatic": bool(is_aromatic),
        },
        "bond_kind": bond.bond_kind.value,
        "is_aromatic": bool(is_aromatic),
        "object": bond if needs_object else None,
    }


def _needs_rings(required_fields):
    return bool({"ring_count", "ring_sizes"}.intersection(required_fields))


def _native_number(value):
    return int(value) if isinstance(value, int) else float(value)


def _pack_ring_sizes(rings, keys):
    sizes = {key: [] for key in keys}
    for ring in rings:
        size = len(ring)
        for atom in ring.atoms:
            sizes[atom.idx].append(size)
    return {key: tuple(values) for key, values in sizes.items()}


def _pack_aromatic_edges(rings):
    return {
        tuple(sorted((bond.a1idx, bond.a2idx)))
        for ring in rings
        if ring.is_aromatic
        for bond in ring.bonds
    }


class _NativeMappingSource:
    """Adapt query-aligned native tuples to the legacy matcher direction."""

    def __init__(self, problem):
        self._problem = problem

    def subgraph_monomorphisms_iter(self) -> Iterable[dict[int, int]]:
        query_keys = self._problem.query_keys
        for target_values in self._problem.cursor():
            yield {
                target: query
                for query, target in zip(query_keys, target_values)
            }

    def mapped_target_indices(self, query_key: int) -> Iterable[int]:
        """Project one query key directly from each raw native embedding."""
        position = self._problem.query_keys.index(query_key)
        for target_values in self._problem.cursor():
            yield target_values[position]
