"""Metal--ligand coordination workflow prerequisites."""

from __future__ import annotations


__all__ = ()


def _require_explicit_complex(mol: "Molecule") -> None:
    """Require a metal center with at least one explicit metal--ligand bond."""
    if not mol.has_metal or not any(
        bond.is_metal_ligand_bond for bond in mol.bonds
    ):
        raise ValueError(
            "The complex workflow requires a molecule with at least one "
            "explicit metal-ligand bond"
        )
