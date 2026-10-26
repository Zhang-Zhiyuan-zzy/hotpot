"""Private adapters for Hotpot's native topology-search executor."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Iterable

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


def _compile_problem(substructure, mol):
    query = _prepare_query(substructure)
    target_spec = _pack_target_snapshot(
        mol,
        needs_atom_objects=query.needs_atom_objects,
        needs_bond_objects=query.needs_bond_objects,
        derived_requirements=query.derived_requirements,
    )
    return _NativeTarget(target_spec).problem(query.query_spec)


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
    target = _NativeTarget(target_spec)
    for query in queries:
        yield target.mapping_source(query.query_spec)


def _mapping_sources_with_automorphism(queries, mol):
    """Build ordered query sources and exact symmetry checks on one target."""
    queries = tuple(queries)
    target_spec = _pack_target_snapshot(
        mol,
        needs_atom_objects=any(query.needs_atom_objects for query in queries),
        needs_bond_objects=any(query.needs_bond_objects for query in queries),
        derived_requirements=_merge_derived_requirements(
            tuple(query.derived_requirements for query in queries)
        ),
    )
    node_labels, edge_labels = _exact_topology_labels(
        mol,
        target_spec["keys"],
        target_spec["edges"],
    )
    target = _NativeTarget(target_spec)
    sources = tuple(
        target.mapping_source(query.query_spec)
        for query in queries
    )
    automorphism = _NativeAutomorphismTask(
        target.automorphism(node_labels, edge_labels),
        target_spec["keys"],
        target_spec["edges"],
        node_labels,
        edge_labels,
    )
    return sources, automorphism


def _merge_derived_requirements(requirements):
    return {
        name: tuple(
            dict.fromkeys(
                value
                for requirement in requirements
                for value in requirement[name]
            )
        )
        for name in ("full", "ligand", "atom_fields", "bond_fields")
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
            "bond_fields": ("bond_order", "is_aromatic"),
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
                derived_requirements["bond_fields"],
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
    field_names = derived_requirements["atom_fields"]
    fields = {
        name: _pack_atom_field(atom, name)
        for name in field_names
    }
    if derived_requirements["total_h"]:
        fields["implicit_hydrogens"] = int(atom.implicit_hydrogens)
        fields["explicit_hydrogens"] = int(atom.explicit_hydrogens)
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
    record = {}
    if "degree" in required_fields:
        record["degree"] = int(semantics.degree(atom))
    if "connectivity" in required_fields:
        record["connectivity"] = int(semantics.connectivity(atom))
    if "valence" in required_fields:
        record["valence"] = _native_number(semantics.valence(atom))
    if "ring_count" in required_fields:
        record["ring_count"] = len(ring_sizes)
    if "ring_sizes" in required_fields:
        record["ring_sizes"] = ring_sizes
    return record


def _pack_bond(bond, needs_object, required_fields, is_aromatic):
    fields = {}
    if "bond_order" in required_fields:
        fields["bond_order"] = float(bond.bond_order)
    if "is_aromatic" in required_fields:
        fields["is_aromatic"] = bool(is_aromatic)
    return {
        "fields": fields,
        "bond_kind": bond.bond_kind.value,
        "is_aromatic": bool(is_aromatic),
        "object": bond if needs_object else None,
    }


def _pack_atom_field(atom, name):
    value = getattr(atom, name)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    return _native_number(value)


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


def _exact_topology_labels(mol, keys, edges):
    atoms = {atom.idx: atom for atom in mol.atoms}
    bonds = {
        tuple(sorted((bond.a1idx, bond.a2idx))): bond
        for bond in mol.bonds
    }
    node_values = tuple(
        (
            int(atoms[key].atomic_number),
            int(atoms[key].formal_charge),
            bool(atoms[key].is_aromatic),
            int(atoms[key].implicit_hydrogens),
            int(getattr(atoms[key], "isotope", 0)),
        )
        for key in keys
    )
    edge_values = tuple(
        (
            "aromatic",
            bonds[edge].bond_kind.value,
        )
        if bonds[edge].is_aromatic
        else (
            "bond_order",
            float(bonds[edge].bond_order),
            bonds[edge].bond_kind.value,
        )
        for edge in edges
    )
    return _integer_labels(node_values), _integer_labels(edge_values)


def _integer_labels(values):
    labels = {
        value: label
        for label, value in enumerate(sorted(set(values), key=repr))
    }
    return tuple(labels[value] for value in values)


@dataclass(frozen=True)
class _NativeAutomorphismTask:
    _native: object
    keys: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    node_labels: tuple[int, ...]
    edge_labels: tuple[int, ...]

    def equivalent(self, first: int, second: int) -> bool:
        return self._native.equivalent(first, second)


class _NativeMappingSource:
    """Expose native embeddings as query-to-molecule mappings."""

    def __init__(self, problem):
        self._problem = problem

    def iter_mappings(self) -> Iterable[dict[int, int]]:
        query_keys = self._problem.query_keys
        for target_values in self._problem.cursor():
            yield dict(zip(query_keys, target_values))

    def has_match(self) -> bool:
        """Probe one native embedding without materializing a mapping."""
        return self._problem.has_match()

    def mapped_target_indices(self, query_key: int) -> Iterable[int]:
        """Project one query key directly from each raw native embedding."""
        position = self._problem.query_keys.index(query_key)
        for target_values in self._problem.cursor():
            yield target_values[position]


class _NativeTarget:
    """Own one packed native target shared by all task-specific queries."""

    def __init__(self, target_spec: dict) -> None:
        self._native_module = _native_module()
        self._target = self._native_module.NativeTarget(target_spec)

    def problem(self, query_spec: dict):
        return self._native_module.NativeProblem(query_spec, self._target)

    def mapping_source(self, query_spec: dict) -> _NativeMappingSource:
        return _NativeMappingSource(self.problem(query_spec))

    def iter_mappings(self, query_spec: dict) -> Iterable[dict[int, int]]:
        return self.mapping_source(query_spec).iter_mappings()

    def automorphism(self, node_labels, edge_labels):
        return self._native_module._NativeAutomorphism(
            self._target,
            node_labels,
            edge_labels,
        )
