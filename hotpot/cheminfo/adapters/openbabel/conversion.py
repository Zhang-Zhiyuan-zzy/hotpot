"""Open Babel projection and factual extraction from prepared values."""

from dataclasses import replace
import numpy as np

from ...structure._builder import _freeze_value
from ...structure._fields import ATOM_SOURCE_CODES, FIELD_PRESENCE_BITS
from . import perception as _perception
from . import crystal as _crystal


def _new_obmol():
    from openbabel import openbabel as ob

    obmol = ob.OBMol()
    obmol.BeginModify()
    return obmol


def _factual_obmol(obmol, atom_rows, bond_rows, cell):
    row_to_idx = {}
    for row, (number, formal_charge, partial_charge, coordinates, aromatic) in enumerate(atom_rows):
        oba = obmol.NewAtom()
        oba.SetAtomicNum(number)
        oba.SetFormalCharge(formal_charge)
        oba.SetPartialCharge(partial_charge)
        oba.SetVector(*coordinates)
        row_to_idx[row] = oba.GetIdx()

    for bond in bond_rows:
        if bond[0] == "dative":
            raise ValueError("Open Babel cannot represent dative bonds losslessly")
        kind, first, second, order, aromatic = bond
        obmol.AddBond(row_to_idx[first], row_to_idx[second], int(order))

    obmol.EndModify()
    for row, atom in enumerate(atom_rows):
        obmol.GetAtom(row_to_idx[row]).SetAromatic(bool(atom[4]))
    for kind, first, second, order, aromatic in bond_rows:
        obmol.GetBond(row_to_idx[first], row_to_idx[second]).SetAromatic(aromatic)
    obmol.SetAromaticPerceived(True)
    if cell is not None:
        obmol.CloneData(_crystal.to_obcell(cell))
    return obmol, row_to_idx


def _obmol_record(obmol, initial, stereo_payload):
    from openbabel import openbabel as ob

    atoms = tuple(ob.OBMolAtomIter(obmol))
    idx_to_row = {atom.GetIdx(): row for row, atom in enumerate(atoms)}
    values = np.zeros((len(atoms), 19), dtype="<f8")
    for row, atom in enumerate(atoms):
        values[row, (0, 7, 8, 9, 10, 11, 12, 14)] = (
            atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetPartialCharge(),
            float(atom.IsAromatic()), atom.GetX(), atom.GetY(), atom.GetZ(), atom.GetImplicitHCount())
    supplied = (0, 7, 8, 9, 10, 11, 12, 14)
    presence = np.full(len(atoms), sum(FIELD_PRESENCE_BITS[index] for index in supplied), dtype="<u4")
    sources = np.zeros((len(atoms), 19), dtype="u1")
    sources[:, supplied] = ATOM_SOURCE_CODES["openbabel"]
    entries = tuple(_perception._retrieve_bonds_attrs_from_obmol(obmol, idx_to_row).items())
    raw_bonds = np.zeros((len(entries), 3), dtype="<f8")
    raw_bonds[:, 0] = [entry[1]["bond_order"] for entry in entries]
    cell_data = obmol.GetData(ob.UnitCell)
    cell = initial.cell
    if cell_data:
        unit_cell = ob.toUnitCell(cell_data)
        cell = tuple(getattr(unit_cell, name)() for name in (
            "GetA", "GetB", "GetC", "GetAlpha", "GetBeta", "GetGamma"))
    record = replace(
        initial, atom_count=len(atoms), atom_values=values.tobytes(),
        atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        bond_endpoints=tuple(endpoints for endpoints, _ in entries),
        bond_values=raw_bonds.tobytes(),
        bond_kinds=tuple(attrs["bond_kind"] for _, attrs in entries),
        bond_aromatic=tuple(attrs['bond_source_metadata']['is_aromatic'] for _, attrs in entries),
        bond_directions=tuple(attrs["bond_direction"] for _, attrs in entries),
        bond_sources=("openbabel",) * len(entries),
        bond_metadata=tuple(_freeze_value(attrs["bond_source_metadata"]) for _, attrs in entries),
        charge=(True, obmol.GetTotalCharge(), "openbabel"), cell=cell,
        stereo_payload=stereo_payload,
        legacy_frame_values=(("none",),) * 9,
    )
    return record


def extract_obatom_coordinate(obatom: "ob.OBMol"):
    return np.array((obatom.GetX(), obatom.GetY(), obatom.GetZ()))


def extract_obmol_coordinates(obmol: "ob.OBMol") -> np.ndarray:
    from openbabel import openbabel as ob
    return np.array([(oba.GetX(), oba.GetY(), oba.GetZ()) for oba in ob.OBMolAtomIter(obmol)])


def set_obmol_coordinates(obmol: "ob.OBMol", coords):
    from openbabel import openbabel as ob
    coords = np.array(coords)
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert obmol.NumAtoms() == coords.shape[0]

    if coords.dtype == np.float64 and obmol.GetMod() == 0 and obmol.GetCoordinates():
        obmol.SetCoordinates(ob.double_array(coords.flat))
    else:
        for oba, coord in zip(ob.OBMolAtomIter(obmol), coords):
            oba.SetVector(*coord)


def to_arrays(obmol):
    from openbabel import openbabel as ob
    # Populate the idx_to_row dictionary to map OBMol atom indices to the reordered indices
    idx_to_row = {oba.GetIdx():i for i, oba in enumerate(ob.OBMolAtomIter(obmol))}

    # Iterate over atoms in the molecule
    atoms_array = np.array([
        [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetPartialCharge(),
            float(atom.IsAromatic()),
            atom.GetX(),
            atom.GetY(),
            atom.GetZ(),
            atom.GetTotalValence(),
            atom.GetImplicitHCount()
        ]
        for atom in ob.OBMolAtomIter(obmol)
    ])

    bonds_array = np.array([
        [
            idx_to_row[bond.GetBeginAtomIdx()],
            idx_to_row[bond.GetEndAtomIdx()],
            bond.GetBondOrder()
        ]
        for bond in ob.OBMolBondIter(obmol)
    ])

    return atoms_array, bonds_array, idx_to_row


def _to_pybel(obmol):
    from openbabel import pybel as pb
    return pb.Molecule(obmol)


def _write_by_pybel(obmol, fmt, filename, overwrite, opt):
    _to_pybel(obmol).write(fmt, filename=filename, overwrite=overwrite, opt=opt)


def _inchi(obmol):
    return _to_pybel(obmol).write('inchi')


def _smiles(obmol, kekulize):
    from openbabel import pybel as pb

    molecule = pb.readstring('mol2', _to_pybel(obmol).write('mol2'))
    if kekulize:
        return molecule.write('can', opt={'k': None}).split()[0]
    return molecule.write('can').split()[0]


def _charge_model(model):
    from openbabel import openbabel as ob
    return ob.OBChargeModel.FindType(model)


def _compute_charges(charge_model, obmol, model_args):
    from openbabel import openbabel as ob

    copied = ob.OBMol(obmol)
    if model_args is None:
        succeeded = charge_model.ComputeCharges(copied)
    else:
        succeeded = charge_model.ComputeCharges(copied, model_args)
    return copied, succeeded


def _partial_charges(charge_model):
    return tuple(float(value) for value in charge_model.GetPartialCharges())
