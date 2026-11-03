"""Immutable task-local selections over complete ordered molecular Bonds."""

from dataclasses import dataclass

from ._identity import _SourceIdentity, _capture_source, _check_source


__all__ = ()


@dataclass(frozen=True)
class _TopologySelection:
    source: _SourceIdentity
    atom_uids: tuple[int, ...]
    bond_mask: bytes

    def __post_init__(self):
        object.__setattr__(self, "atom_uids", tuple(self.atom_uids))
        object.__setattr__(self, "bond_mask", bytes(self.bond_mask))


def _capture_topology_selection(mol, *, excluded_bonds=(), ligand_skeleton=False) -> _TopologySelection:
    with mol._mutation_lock:
        complete = tuple(mol._bonds)
        excluded = tuple(excluded_bonds)
        excluded_ids = {id(bond) for bond in excluded}
        member_ids = {id(bond) for bond in complete}
        if len(excluded_ids) != len(excluded) or any(
                id(bond) not in member_ids or bond._uid_owner is not mol._owner_token
                or bond.mol is not mol for bond in excluded):
            raise ValueError("Topology exclusions require distinct current Bond handles of this owner")
        mask = bytes(id(bond) not in excluded_ids
                     and (not ligand_skeleton or not bond.is_metal_ligand_bond)
                     for bond in complete)
        domains = ("topology", "representation") if ligand_skeleton else ("topology",)
        return _TopologySelection(_capture_source(mol, domains),
                                  tuple(atom._uid for atom in mol._atoms), mask)


def _check_topology_selection(mol, selection) -> None:
    with mol._mutation_lock:
        _check_source(mol, selection.source)
        if selection.atom_uids != tuple(atom._uid for atom in mol._atoms):
            raise RuntimeError("Topology selection Atom membership changed; capture a new selection")
        if len(selection.bond_mask) != len(mol._bonds) or any(
                value not in (0, 1) for value in selection.bond_mask):
            raise ValueError("Topology selection requires one binary byte per complete Bond row")


def _selected_bonds(mol, selection) -> tuple:
    with mol._mutation_lock:
        if selection is None:
            return tuple(mol._bonds)
        _check_topology_selection(mol, selection)
        return tuple(bond for bond, included in zip(mol._bonds, selection.bond_mask) if included)
