"""RDKit graph packing and factual extraction from prepared transport values."""

from dataclasses import replace
import numpy as np

from ...structure._builder import _freeze_value
from ...structure._fields import ATOM_SOURCE_CODES, FIELD_PRESENCE_BITS
from . import stereo_transport as _stereo


def _dative_bond_types(types):
    return frozenset(value for value in (
        getattr(types, "DATIVE", None), getattr(types, "DATIVEONE", None),
        getattr(types, "DATIVEL", None), getattr(types, "DATIVER", None),
    ) if value is not None)


def _rdkit_bond_type(kind, raw_type_name, is_aromatic, order, types, dative_types, kind_to_type):
    if kind == "dative":
        raw_type = getattr(types, raw_type_name, None) if raw_type_name is not None else None
        if raw_type in dative_types:
            return raw_type
        return types.DATIVE
    if kind == "aromatic" or is_aromatic:
        return types.AROMATIC
    if kind in kind_to_type:
        return kind_to_type[kind]
    return types.values[order]


def _rdkit_bond_kind(bond_type, is_aromatic, types, dative_types):
    if is_aromatic or bond_type == types.AROMATIC:
        return "aromatic"
    if bond_type == types.ZERO:
        return "zero"
    if bond_type in dative_types:
        return "dative"
    if bond_type == types.SINGLE:
        return "single"
    if bond_type == types.DOUBLE:
        return "double"
    if bond_type == types.TRIPLE:
        return "triple"
    return "unknown"


def _new_rdmol():
    from rdkit import Chem
    return Chem.RWMol()


def _pack_rdmol(rdmol, atom_rows, bond_rows, coordinates, hydrogens, atom_facts, kekulize, sanitize):
    from rdkit import Chem

    hybridization = Chem.rdchem.HybridizationType
    hybridizations = {
        -1: hybridization.UNSPECIFIED, 0: hybridization.S, 1: hybridization.SP,
        2: hybridization.SP2, 3: hybridization.SP3, 4: hybridization.SP2D,
        5: hybridization.SP3D, 6: hybridization.SP3D2, 7: hybridization.OTHER,
    }
    types = Chem.BondType
    dative_types = _dative_bond_types(types)
    kind_to_type = {"zero": types.ZERO, "single": types.SINGLE,
                    "double": types.DOUBLE, "triple": types.TRIPLE}
    row_to_idx = {}
    for row, (number, formal_charge, partial_charge, aromatic, hyb) in enumerate(atom_rows):
        rda = Chem.Atom(number)
        rda.SetFormalCharge(formal_charge)
        atom_fact = atom_facts.get(row)
        if atom_fact is None:
            rda.SetNumExplicitHs(hydrogens[row])
        else:
            _stereo._apply_atom_fact(rda, atom_fact)
        if partial_charge is not None:
            rda.SetDoubleProp('_GasteigerCharge', partial_charge)
        rda.SetIsAromatic(aromatic)
        rda.SetHybridization(hybridizations[hyb])
        row_to_idx[row] = rdmol.AddAtom(rda)

    conf = Chem.Conformer(len(atom_rows))
    for row, (x, y, z) in enumerate(coordinates):
        conf.SetAtomPosition(row, Chem.rdGeometry.Point3D(x, y, z))
    rdmol.AddConformer(conf)

    for bond in bond_rows:
        first, second, kind, direction = bond[:4]
        if kind == "dative":
            if direction is None:
                raise ValueError("RDKit dative-bond export requires a bond direction")
            if direction == "atom2_to_atom1":
                first, second = second, first
            elif direction != "atom1_to_atom2":
                raise ValueError(f"Unsupported dative-bond direction: {direction}")
        raw_type_name, aromatic, order = bond[4:]
        rdmol.AddBond(row_to_idx[first], row_to_idx[second],
                      _rdkit_bond_type(kind, raw_type_name, aromatic, order,
                                       types, dative_types, kind_to_type))

    rdmol = rdmol.GetMol()
    if kekulize:
        Chem.Kekulize(rdmol)
    if sanitize:
        Chem.SanitizeMol(rdmol)
    return Chem.RWMol(rdmol), row_to_idx


def _finish_rdmol(rdmol):
    return rdmol.GetMol()


def _rdmol_record(rdmol, initial):
    from rdkit import Chem

    dative_types = _dative_bond_types(Chem.BondType)
    rdmol = Chem.Mol(rdmol)
    stereo_payload = _stereo._extract_stereo_payload(rdmol)
    source_bond_metadata = {
        bond.GetIdx(): {
            "raw_bond_type": str(bond.GetBondType()),
            "raw_bond_type_code": int(bond.GetBondType()),
            "is_aromatic": bool(bond.GetIsAromatic()),
            "is_conjugated": bool(bond.GetIsConjugated()),
            "stereo": str(bond.GetStereo()),
            "bond_dir": str(bond.GetBondDir()),
        }
        for bond in rdmol.GetBonds()
    }
    source_bond_types = {
        bond.GetIdx(): bond.GetBondType()
        for bond in rdmol.GetBonds()
    }
    Chem.Kekulize(rdmol)
    conformer = rdmol.GetConformer() if rdmol.GetNumConformers() else None

    atoms = tuple(rdmol.GetAtoms())
    idx_to_row = {atom.GetIdx(): row for row, atom in enumerate(atoms)}
    values = np.zeros((len(atoms), 19), dtype="<f8")
    presence = np.zeros(len(atoms), dtype="<u4")
    sources = np.zeros((len(atoms), 19), dtype="u1")
    for row, atom in enumerate(atoms):
        values[row, 0] = atom.GetAtomicNum()
        values[row, 7] = atom.GetFormalCharge()
        values[row, 9] = atom.GetIsAromatic()
        values[row, 14] = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
        supplied = [0, 7, 9, 14]
        if atom.HasProp('_GasteigerCharge'):
            values[row, 8] = atom.GetDoubleProp('_GasteigerCharge')
            supplied.append(8)
        if conformer is not None:
            values[row, 10:13] = tuple(conformer.GetAtomPosition(atom.GetIdx()))
            supplied.extend((10, 11, 12))
        presence[row] = sum(FIELD_PRESENCE_BITS[index] for index in supplied)
        sources[row, supplied] = ATOM_SOURCE_CODES["rdkit"]
    bonds = tuple(rdmol.GetBonds())
    raw_bonds = np.zeros((len(bonds), 3), dtype="<f8")
    raw_bonds[:, 0] = [bond.GetBondTypeAsDouble() for bond in bonds]
    record = replace(
        initial, atom_count=len(atoms), atom_values=values.tobytes(),
        atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        bond_endpoints=tuple((idx_to_row[bond.GetBeginAtomIdx()], idx_to_row[bond.GetEndAtomIdx()]) for bond in bonds),
        bond_values=raw_bonds.tobytes(),
        bond_kinds=tuple(_rdkit_bond_kind(source_bond_types[bond.GetIdx()],
                         source_bond_metadata[bond.GetIdx()]["is_aromatic"], Chem.BondType, dative_types) for bond in bonds),
        bond_aromatic=tuple(source_bond_metadata[bond.GetIdx()]['is_aromatic'] for bond in bonds),
        bond_directions=tuple("atom1_to_atom2" if source_bond_types[bond.GetIdx()] in dative_types
                              else None for bond in bonds),
        bond_sources=("rdkit",) * len(bonds),
        bond_metadata=tuple(_freeze_value(source_bond_metadata[bond.GetIdx()]) for bond in bonds),
        charge=(True, Chem.GetFormalCharge(rdmol), "rdkit"),
        stereo_payload=initial.stereo_payload,
        legacy_frame_values=(("none",),) * 9,
        partial_charge_result=None,
    )
    return record, stereo_payload, idx_to_row
