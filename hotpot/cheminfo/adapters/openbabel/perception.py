"""Open Babel perception from task-local external values."""

from typing import Any
from ...structure._builder import _freeze_value


def _retrieve_bonds_attrs_from_obmol(obmol: "ob.OBMol", i2r: dict[int, int]) -> dict[tuple[int, int], dict[str, Any]]:
    from ._runtime import ob

    bond_attrs = {}
    for obb in ob.OBMolBondIter(obmol):
        bond_order = obb.GetBondOrder()
        if bond_order == 0:
            bond_kind = "unknown"
        elif obb.IsAromatic():
            bond_kind = "aromatic"
        else:
            bond_kind = {
                1: "single",
                2: "double",
                3: "triple",
            }.get(bond_order, "unknown")

        bond_attrs[(i2r[obb.GetBeginAtomIdx()], i2r[obb.GetEndAtomIdx()])] = {
            "bond_order": bond_order,
            "bond_kind": bond_kind,
            "bond_direction": None,
            "bond_source": "openbabel",
            "bond_source_metadata": {
                "raw_bond_order": bond_order,
                "raw_flags": int(obb.GetFlags()),
                "is_aromatic": bool(obb.IsAromatic()),
                "is_amide": bool(obb.IsAmide()),
                "is_in_ring": bool(obb.IsInRing()),
                "is_wedge": bool(obb.IsWedge()),
                "is_hash": bool(obb.IsHash()),
            },
        }
    return bond_attrs


def _bond_proposals(obmol, idx_to_row):
    return tuple((pair, attrs['bond_order'], attrs['bond_kind'], attrs['bond_direction'],
                  attrs['bond_source'], _freeze_value(attrs['bond_source_metadata']),
                  attrs['bond_source_metadata']['is_aromatic'])
                 for pair, attrs in _retrieve_bonds_attrs_from_obmol(obmol, idx_to_row).items())


def _connect_the_dots(obmol):
    obmol.ConnectTheDots()


def _perceive_bond_orders(obmol, idx_to_row):
    from ._runtime import ob

    obmol.PerceiveBondOrders()
    return (tuple(bool(atom.IsAromatic()) for atom in ob.OBMolAtomIter(obmol)),
            _bond_proposals(obmol, idx_to_row))
