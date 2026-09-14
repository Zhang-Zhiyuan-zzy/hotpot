"""Conversions from supported chemistry objects to Hotpot molecules."""

from __future__ import annotations

from os import PathLike

from openbabel import openbabel as ob, pybel
from rdkit import Chem

from .core import Molecule
from .core_utils import read_mol
from .obconvert import obmol2mol
from .rdconvert import from_rdmol


def is_molecule_input(value) -> bool:
    """Return whether ``value`` is one supported single-molecule input."""
    return isinstance(
        value,
        (str, PathLike, Molecule, Chem.Mol, ob.OBMol, pybel.Molecule),
    ) or callable(getattr(value, "to_rdmol", None))


def to_hotpot_mol(value, *, fmt=None, **kwargs) -> Molecule:
    """Return ``value`` as a Hotpot :class:`Molecule`.

    Hotpot molecules are returned unchanged. RDKit, Open Babel, and Pybel
    molecule objects are copied into a new Hotpot molecule. Strings and paths
    are parsed by Hotpot's standard molecule reader.
    """
    if isinstance(value, Molecule):
        return value
    if isinstance(value, Chem.Mol):
        return from_rdmol(value, Molecule())
    if isinstance(value, ob.OBMol):
        return obmol2mol(value, Molecule())
    if isinstance(value, pybel.Molecule):
        return obmol2mol(value.OBMol, Molecule())
    if isinstance(value, (str, PathLike)):
        return read_mol(value, fmt=fmt, **kwargs)
    if callable(getattr(value, "to_rdmol", None)):
        rdmol = value.to_rdmol()
        if not isinstance(rdmol, Chem.Mol):
            raise TypeError("to_rdmol() must return an RDKit Chem.Mol")
        return from_rdmol(rdmol, Molecule())
    raise TypeError(f"unsupported molecule type: {type(value)!r}")


__all__ = ["is_molecule_input", "to_hotpot_mol"]
