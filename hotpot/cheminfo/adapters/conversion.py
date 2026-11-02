"""Toolkit type recognition for the conversion operation."""

from os import PathLike


def _classify_input(value, molecule_type):
    if isinstance(value, molecule_type):
        return "hotpot"
    from rdkit import Chem
    if isinstance(value, Chem.Mol):
        return "rdkit"
    from openbabel import openbabel as ob, pybel
    if isinstance(value, ob.OBMol):
        return "openbabel"
    if isinstance(value, pybel.Molecule):
        return "pybel"
    if isinstance(value, (str, PathLike)):
        return "source"
    if callable(getattr(value, "to_rdmol", None)):
        return "protocol"
    return "unsupported"


def _is_rdmol(value):
    from rdkit import Chem
    return isinstance(value, Chem.Mol)
