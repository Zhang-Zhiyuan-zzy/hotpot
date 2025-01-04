"""
python v3.9.0
@Project: hotpot
@File   : core_
@Auther : Zhiyuan Zhang
@Data   : 2025/1/1
@Time   : 15:35
"""
import numpy as np
import weakref


class Molecule(object):
    def __init__(self, _atoms_attrs: np.ndarray = None, _bond_attrs: np.ndarray = None):
        self._atoms_attrs = None
        self._bonds_attrs = None

    @property
    def atom_counts(self) -> int:
        return len(self._atoms_attrs)

    @property
    def link_matrix(self) -> np.ndarray:
        return self._bonds_attrs[:, :2]

    @property
    def atoms(self) -> list["Atom"]:
        return [Atom(self, i) for i in range(self.atom_counts)]


class Atom(object):
    def __init__(self, _mol: Molecule, _idx: int):
        self._mol = _mol
        self._idx = _idx

    @property
    def mol(self) -> Molecule:
        return self._mol

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def neigh_idx(self) -> np.ndarray:
        mol_adj = self._mol.link_matrix
        mask_mat = mol_adj == self.idx
        bond_idx = np.where(np.any(mask_mat, axis=1))[0]
        return np.where(mask_mat[bond_idx][:, 0], mol_adj[bond_idx][:, 1], mol_adj[bond_idx][:, 0])
