"""Lightweight molecular graph input accepted by the shared converter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class MoleculeGraph:
    """Minimal molecular graph accepted by the public API.

    Bond tuples are ``(begin_index, end_index, bond_order)``. Coordinates are
    optional; when absent the package generates a deterministic conformer.
    """

    atomic_numbers: Sequence[int]
    bonds: Sequence[tuple[int, int, float]]
    formal_charges: Sequence[int] | None = None
    coordinates: Sequence[Sequence[float]] | None = None

    def to_rdmol(self) -> Chem.Mol:
        editable = Chem.RWMol()
        charges = (
            [0] * len(self.atomic_numbers)
            if self.formal_charges is None
            else self.formal_charges
        )
        for atomic_number, charge in zip(self.atomic_numbers, charges):
            atom = Chem.Atom(int(atomic_number))
            atom.SetFormalCharge(int(charge))
            editable.AddAtom(atom)
        bond_types = {
            1.0: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2.0: Chem.BondType.DOUBLE,
            3.0: Chem.BondType.TRIPLE,
        }
        for begin, end, order in self.bonds:
            editable.AddBond(int(begin), int(end), bond_types[float(order)])
        mol = editable.GetMol()
        Chem.SanitizeMol(mol)
        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=np.float64)
            conformer = Chem.Conformer(len(self.atomic_numbers))
            for index, xyz in enumerate(coordinates):
                conformer.SetAtomPosition(index, xyz)
            mol.AddConformer(conformer)
        return mol

    to_rdkit = to_rdmol
