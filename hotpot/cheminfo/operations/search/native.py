"""Prepare carrier facts and orchestrate native substructure-search tasks."""

from __future__ import annotations

from dataclasses import dataclass

from ...adapters.native.substructure.matching import _NativeTarget, _NativeAutomorphismTask
from ...adapters.structure.substructure import _pack_target_values, _exact_topology_label_values
from ...adapters.structure.substructure_query import (
    _compile_query, _query_derived_requirements, _query_object_requirements,
)

__all__ = ()


@dataclass(frozen=True)
class _PreparedQuery:
    """Immutable ownership record for one compiled native query snapshot."""

    query_spec: dict
    needs_atom_objects: bool
    needs_bond_objects: bool
    derived_requirements: dict


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
    from ..chemistry import (
        _require_aromatic_representation, _require_implicit_hydrogens, _require_non_graph_hydrogens,
    )
    with mol._mutation_lock:
        if (derived_requirements['total_h']
                or any({'connectivity', 'valence'}.intersection(derived_requirements[name])
                       for name in ('full', 'ligand'))):
            _require_non_graph_hydrogens(mol)
        elif 'implicit_hydrogens' in derived_requirements['atom_fields']:
            _require_implicit_hydrogens(mol)
        if (derived_requirements['bond_aromatic'] or 'is_aromatic' in derived_requirements['atom_fields']
                or 'is_aromatic' in derived_requirements['bond_fields']):
            _require_aromatic_representation(mol)
        return _pack_target_values(mol, needs_atom_objects=needs_atom_objects,
            needs_bond_objects=needs_bond_objects, derived_requirements=derived_requirements)


def _exact_topology_labels(mol, keys, edges):
    from ..chemistry import _require_aromatic_representation, _require_implicit_hydrogens
    with mol._mutation_lock:
        _require_implicit_hydrogens(mol)
        _require_aromatic_representation(mol)
        return _exact_topology_label_values(mol, keys, edges)
