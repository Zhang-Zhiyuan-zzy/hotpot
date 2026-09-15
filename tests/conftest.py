"""Process-wide test initialization for native chemistry extensions."""

# PyPI Open Babel 3.2.1 can crash if pybel loads before RDKit in one process.
# Hotpot enforces the same deterministic order in its package entry point.
from rdkit import Chem as _rdkit_chem  # noqa: F401
