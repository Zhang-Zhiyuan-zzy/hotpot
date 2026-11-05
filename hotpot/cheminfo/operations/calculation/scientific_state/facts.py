"""Prepare an owned working copy for immutable scientific-state capture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from hotpot.cheminfo.adapters.structure.scientific_state import _extract_molecular_facts
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.operations.chemistry import _prepare_hydrogens, _require_non_graph_hydrogens
from hotpot.cheminfo.structure.snapshots import MolecularFacts
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.rules.chemistry.scientific_state.charge.contracts import (
    AtomFormalChargeEvidence, ComponentChargeEvidence, NetChargeCandidateEvidence,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateProvenance, CoverageStatus, StateValueAuthority,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.electronic_state.contracts import (
    ElectronicStateCandidateEvidence,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

__all__ = ["extract_molecular_facts"]


def extract_molecular_facts(mol: Molecule) -> MolecularFacts:
    """Extract immutable chemical facts without mutating ``mol``."""
    working_copy = mol.copy()
    _prepare_hydrogens(working_copy)
    table = working_copy._atom_table
    field, group = FIELD_INDEX['implicit_hydrogens'], GROUP_INDEX['IMPLICIT_H']
    known_rows = tuple(row for row in range(len(table))
                       if table._group_status[row, group] == _FactStatus.CURRENT
                       and int(table._presence[row]) & FIELD_PRESENCE_BITS[field])
    _require_non_graph_hydrogens(working_copy, rows=known_rows)
    return _extract_molecular_facts(working_copy)


_SOURCE_CHARGE_CODES = frozenset(
    ATOM_SOURCE_CODES[name]
    for name in ("explicit", "legacy_array", "rdkit", "openbabel", "ccdc",
                 "mca_dto", "cclib", "tmqm")
)


def _source_digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _source_radical_electrons(mol: Molecule) -> tuple[int | None, ...]:
    """Read canonical supplied radical occurrences without preparing chemistry."""
    with mol._mutation_lock:
        result = []
        for row in range(len(mol._atom_table)):
            values = {fact[1] for fact in mol._atom_table._stereo_facts[row]
                      if fact[1] is not None}
            if len(values) > 1:
                raise ValueError(f"Conflicting source radical declarations at atom {row}")
            result.append(next(iter(values)) if values else None)
        return tuple(result)


def _molecule_state_source_identity(
    mol: Molecule, facts: MolecularFacts,
) -> str:
    """Bind declarations and supplied site evidence omitted by structural identity."""
    with mol._mutation_lock:
        table = mol._atom_table
        field = FIELD_INDEX["formal_charge"]
        atom_charges = tuple(
            (present, int(table._sources[row, field]) if present else None,
             int(table._columns[field][row]) if present else None)
            for row in range(len(table))
            for present in (bool(int(table._presence[row]) & FIELD_PRESENCE_BITS[field]),)
        )
        declarations = tuple(
            (name, present, mol.__dict__[f"_{name}_value"] if present else None,
             mol.__dict__[f"_{name}_source"] if present else None)
            for name in ("charge", "spin")
            for present in (mol.__dict__[f"_{name}_present"],)
        )
        radical_occurrences = tuple(
            tuple((fact[1], fact[6]) for fact in table._stereo_facts[row])
            for row in range(len(table))
        )
    return _source_digest({
        "facts": facts.identity.chemical_identity_digest,
        "declarations": declarations, "atom_charges": atom_charges,
        "radical_occurrences": radical_occurrences,
    })


def _state_request_from_molecule(
    mol: Molecule, facts: MolecularFacts, request: StateRequest,
) -> StateRequest:
    """Bind actual supplied state to captured facts; invocation fields take priority.

    Formal-charge coverage status describes aggregate completeness, not whether
    a supplied atom value survives a coordination edit. Only presence and a
    supplied-source code admit an atom value here. No derived facade getter is
    consulted and no oxidation or spin default is chosen at this boundary.
    """
    with mol._mutation_lock:
        table = mol._atom_table
        field = FIELD_INDEX["formal_charge"]
        charges = tuple(
            int(table._columns[field][row])
            if int(table._presence[row]) & FIELD_PRESENCE_BITS[field]
            and int(table._sources[row, field]) in _SOURCE_CHARGE_CODES
            else None
            for row in range(len(table))
        )
        sources = tuple(int(table._sources[row, field])
                        if int(table._presence[row]) & FIELD_PRESENCE_BITS[field] else None
                        for row in range(len(table)))
        charge_declaration = (mol._charge_present, mol._charge_value, mol._charge_source)
        spin_declaration = (mol._spin_present, mol._spin_value, mol._spin_source)
    raw_source_identity = _molecule_state_source_identity(mol, facts)
    source_identity = _source_digest({
        "facts": facts.identity.chemical_identity_digest,
        "charges": charges, "sources": sources, "raw_source": raw_source_identity,
    })
    atom_evidence = request.atom_formal_charges
    if atom_evidence is None and any(charge is not None for charge in charges):
        atom_evidence = AtomFormalChargeEvidence(
            CandidateProvenance(
                source_identity, StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                "molecule:supplied-atom-formal-charges", (source_identity,),
            ), charges,
            CoverageStatus.COMPLETE if all(charge is not None for charge in charges)
            else CoverageStatus.PARTIAL,
        )
    component_evidence = request.component_charges
    if component_evidence is None and atom_evidence is not None:
        supplied = tuple(
            (component.identity, sum(atom_evidence.charges[index]
                                     for index in component.atom_indices))
            for component in facts.source_components
            if all(atom_evidence.charges[index] is not None
                   for index in component.atom_indices)
        )
        if supplied:
            identity = _source_digest((atom_evidence.provenance.identity, supplied))
            component_evidence = ComponentChargeEvidence(
                CandidateProvenance(
                    identity, atom_evidence.provenance.authority,
                    "molecule:supplied-component-formal-charge-sum",
                    (atom_evidence.provenance.identity,),
                ), supplied,
                CoverageStatus.COMPLETE if len(supplied) == len(facts.source_components)
                else CoverageStatus.PARTIAL,
            )
    charge_candidates = request.charge_candidates
    if request.net_charge is None and not charge_candidates:
        present, charge, source = charge_declaration
        if present:
            identity = _source_digest((facts.identity.chemical_identity_digest,
                                       "molecular-charge", charge, source, raw_source_identity))
            charge_candidates = (NetChargeCandidateEvidence(
                CandidateProvenance(
                    identity,
                    StateValueAuthority.VERIFIED_EXPLICIT if source == "explicit"
                    else StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                    f"molecule:charge:{source}",
                ), int(charge),
            ),)
        elif atom_evidence is not None and atom_evidence.coverage is CoverageStatus.COMPLETE:
            identity = _source_digest((atom_evidence.provenance.identity, "charge-sum"))
            charge_candidates = (NetChargeCandidateEvidence(
                CandidateProvenance(
                    identity, atom_evidence.provenance.authority,
                    "molecule:complete-supplied-formal-charge-sum",
                    (atom_evidence.provenance.identity,),
                ), sum(atom_evidence.charges),
            ),)
    electronic_candidates = request.electronic_candidates
    if request.two_s is None and request.multiplicity is None and not electronic_candidates:
        present, multiplicity, source = spin_declaration
        if present and multiplicity is not None:
            identity = _source_digest((facts.identity.chemical_identity_digest,
                                       "molecular-spin", multiplicity, source, raw_source_identity))
            electronic_candidates = (ElectronicStateCandidateEvidence(
                CandidateProvenance(
                    identity,
                    StateValueAuthority.VERIFIED_EXPLICIT if source == "explicit"
                    else StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                    f"molecule:spin:{source}",
                ), int(multiplicity) - 1, int(multiplicity), True,
            ),)
    return replace(
        request, atom_formal_charges=atom_evidence,
        component_charges=component_evidence, charge_candidates=charge_candidates,
        electronic_candidates=electronic_candidates,
    )
