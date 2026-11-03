"""Prepared native search and exact automorphism for ordered MCA rules."""
from __future__ import annotations
from hotpot.cheminfo.operations.search import SmartsSemantics, Substructure
from hotpot.cheminfo.operations.search.native import _mapping_sources_with_automorphism, _prepare_query
from hotpot.cheminfo.rules.chemistry.mca_sites import (
    DetectedSite, NUCLEOPHILE_RULES, _is_eligible_site, _refined_node_colors, name_sites,
)


def _compile_rule(name: str, smarts: str):
    substructure = Substructure.from_smarts(
        smarts,
        semantics=SmartsSemantics.LIGAND_SKELETON,
    )
    anchors = [atom.idx for atom in substructure.query_atoms if atom.map_number == 1]
    if len(anchors) != 1:
        raise ValueError(f"MCA rule {name!r} must contain exactly one :1 anchor")
    return name, _prepare_query(substructure), anchors[0]

# Compile invariant queries once, never once per target.
_COMPILED_RULES = tuple(_compile_rule(*rule) for rule in NUCLEOPHILE_RULES)

def _remove_automorphic_sites(
    automorphism, sites: tuple[DetectedSite, ...]
) -> tuple[DetectedSite, ...]:
    colors = _refined_node_colors(automorphism.keys, automorphism.node_labels, automorphism.edges, automorphism.edge_labels)
    representatives: dict[int, list[int]] = {}
    result = []
    for site in sites:
        equivalent = any(
            automorphism.equivalent(site.atom_index, representative)
            for representative in representatives.get(colors[site.atom_index], ())
        )
        if not equivalent:
            representatives.setdefault(colors[site.atom_index], []).append(site.atom_index)
            result.append(site)
    return tuple(result)

def find_nucleophilic_sites(mol) -> tuple[DetectedSite, ...]:
    """Return symmetry-unique MCA sites found by Hotpot's native search."""

    rule_sources, automorphism = _mapping_sources_with_automorphism(
        (query for _, query, _ in _COMPILED_RULES),
        mol,
    )
    mapped_sources = tuple(
        (name, tuple(source.mapped_target_indices(anchor_query_index)))
        for (name, _, anchor_query_index), source in zip(_COMPILED_RULES, rule_sources)
    )
    eligibility = tuple(
        _is_eligible_site(atom.is_metal, tuple(bond.is_metal_ligand_bond for bond in atom.bonds))
        for atom in mol.atoms
    )
    sites = name_sites(mapped_sources, eligibility)
    return _remove_automorphic_sites(automorphism, sites)
