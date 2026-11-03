"""Shared formatting of the scientific-state molecular subject identity."""

from hotpot.cheminfo.structure.snapshots import MolecularFacts


def _molecular_subject_identity(facts: MolecularFacts) -> str:
    return f"molecule:{facts.identity.chemical_identity_digest}"
