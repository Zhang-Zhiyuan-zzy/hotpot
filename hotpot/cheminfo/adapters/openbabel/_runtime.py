"""Load the external Open Babel runtime only when an adapter first uses it."""

# PyPI Open Babel 3.2 can crash when pybel initializes before RDKit Chem.
from rdkit import Chem as _rdkit_chem  # noqa: F401
from openbabel import openbabel as ob, pybel


__all__ = ()
