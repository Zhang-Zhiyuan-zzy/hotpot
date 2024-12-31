"""
python v3.9.0
@Project: hotpot0.5.0
@File   : ob2chem
@Auther : Zhiyuan Zhang
@Data   : 2024/6/1
@Time   : 17:31

Notes: Communication between hotpot object and openbabel object.
"""
import os
import io
from os import PathLike
from pathlib import Path
from typing import *

import networkx as nx
import numpy as np
from openbabel import openbabel as ob, pybel as pb


def _src_checks(src) -> Literal['path', 'str', 'bytes', 'StringIO', 'BytesIO', 'FileIO']:
    if isinstance(src, str):
        if os.path.exists(src):
            return 'path'
        else:
            return 'str'
    elif isinstance(src, PathLike):
        return 'path'
    elif isinstance(src, bytes):
        return 'bytes'
    elif isinstance(src, io.StringIO):
        return 'StringIO'
    elif isinstance(src, io.BytesIO):
        return 'BytesIO'
    elif isinstance(src, io.FileIO):
        return 'FileIO'
    else:
        raise TypeError(f'get unsupported input type {type(src)}')


def read(src, fmt=None, opt=None) -> Generator:
    """
    Read in a molecule from a string.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       string

    Optional parameters:
       opt    - a dictionary of format-specific options
                For format options with no parameters, specify the
                value as None.
    """
    src_type = None
    if not fmt:
        if isinstance(src, (str, PathLike)):
            p_src = Path(src)
            src_type = 'path'
            if p_src.is_file():
                fmt = p_src.suffix
            else:
                raise FileNotFoundError(f'file {p_src} not exist!')

    if not fmt:
        raise ValueError('the fmt has not been known!')

    if opt is None:
        opt = {}

    obconversion = ob.OBConversion()

    formatok = obconversion.SetInFormat(fmt)
    if not formatok:
        raise ValueError("%s is not a recognised Open Babel format" % fmt)
    for k, v in opt.items():
        if v is None:
            obconversion.AddOption(k, obconversion.INOPTIONS)
        else:
            obconversion.AddOption(k, obconversion.INOPTIONS, str(v))

    # Get the source type name
    if not src_type:
        src_type = _src_checks(src)

    if src_type == 'path':
        obreader = obconversion.ReadFile
    elif src_type in ('str', 'IOString'):
        obreader = obconversion.ReadString
    else:
        raise RuntimeError(f'the source type {type(src)} have not been supported')

    if src_type == 'IOString':
        src = src.read()

    def reader():
        """ OBMol generator """
        obmol = ob.OBMol()
        notatend = obreader(obmol, str(src))
        while notatend:
            yield obmol
            obmol = ob.OBMol()
            notatend = obconversion.Read(obmol)

    return reader()


def to_arrays(obmol):
    # Initialize lists to store atom and bond information
    atoms = []
    bonds = []
    idx_to_row = {}

    # Populate the idx_to_row dictionary to map OBMol atom indices to the reordered indices
    for i, atom in enumerate(ob.OBMolAtomIter(obmol)):
        idx_to_row[atom.GetIdx()] = i

    # Iterate over atoms in the molecule
    for i, atom in enumerate(ob.OBMolAtomIter(obmol)):
        atomic_number = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        partial_charge = atom.GetPartialCharge()
        is_aromatic = int(atom.IsAromatic())
        x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
        valence = atom.GetTotalValence()
        implicit_hydrogens = atom.GetImplicitHCount()
        # explicit_hydrogens = atom.GetExplicitHCount()

        # Append the atom information to the atoms list
        atoms.append([
            atomic_number, formal_charge, partial_charge,
            is_aromatic, x, y, z, valence, implicit_hydrogens,
            # explicit_hydrogens
        ])

    # Iterate over bonds in the molecule
    for bond in ob.OBMolBondIter(obmol):
        begin_atom_idx = idx_to_row[bond.GetBeginAtomIdx()]  # Map to reordered index
        end_atom_idx = idx_to_row[bond.GetEndAtomIdx()]  # Map to reordered index
        bond_order = bond.GetBondOrder()
        is_aromatic = bond.IsAromatic()

        # Append the bond information to the bonds list
        bonds.append([begin_atom_idx, end_atom_idx, bond_order, is_aromatic])

    # Convert lists to numpy arrays
    atoms_array = np.array(atoms)
    bonds_array = np.array(bonds)

    return atoms_array, bonds_array, idx_to_row

def to_obmol(mol):
    atoms_array, bonds_array = getattr(mol, '_atoms_data'), getattr(mol, '_bonds_data')

    obmol = ob.OBMol()

    # Create atoms in the molecule and set their properties
    row_to_idx = {}
    for i, atom_data in enumerate(atoms_array):
        atomic_number, formal_charge, partial_charge, is_aromatic, x, y, z, valence, impH = atom_data
        atom = obmol.NewAtom()
        atom.SetAtomicNum(int(atomic_number))
        atom.SetFormalCharge(int(formal_charge))
        atom.SetPartialCharge(float(partial_charge))
        atom.SetVector(x, y, z)
        atom.SetAromatic(bool(is_aromatic))  # Convert to bool

        # Store mapping from label to atom index (1-based indexing in OBMol)
        row_to_idx[i] = atom.GetIdx()

    # Create bonds in the molecule
    for bond_data in bonds_array:
        begin_atom_idx, end_atom_idx, bond_order, is_aromatic = bond_data
        obmol.AddBond(
            row_to_idx[begin_atom_idx],
            row_to_idx[end_atom_idx],
            int(bond_order)
        )
        bond = obmol.GetBond(row_to_idx[begin_atom_idx], row_to_idx[end_atom_idx])
        bond.SetAromatic(bool(is_aromatic))  # Convert to bool

    return obmol, row_to_idx


def ob_dump(mol: Union[ob.OBMol, pb.Molecule], fmt, **kwargs) -> Union[str, bytes]:
    """
    dump an openbabel molecule information to string or bytes
    Args:
        mol:
        fmt:
        filepath:
        **kwargs:

    Returns:

    """
    if isinstance(mol, ob.OBMol):
        mol = pb.Molecule(mol)
    elif not isinstance(mol, pb.Molecule):
        raise TypeError(f'the mol type {type(mol)} is not supported, just allow openbabal.OBMol or pybel.Molecule')

    return mol.write(fmt, **kwargs)


# class OBabel:
#     @staticmethod
#     def ob_read(src, fmt=None) -> ob.OBMol:
#         """ read chemical information by openbabel module """
#         if not fmt:
#             if isinstance(src, (str, PathLike)):
#                 p_src = Path(src)
#                 if p_src.is_file():
#                     fmt = p_src.suffix
#                 else:
#                     raise FileNotFoundError(f'file {p_src} not exist!')
#
#         if not fmt:
#             raise ValueError('the fmt has not been known!')
#
#         src_type = _src_checks(src)  # Get the source type name
#         if src_type == 'str':
#             pybel_mol = pb.readstring(fmt, src)
#         elif src_type == 'path':
#             pybel_mol = next(pb.readfile(fmt, str(src)))
#         elif src_type == 'IOString':
#             pybel_mol = pb.readstring(fmt, src.read())
#         else:
#             raise RuntimeError(f'the source type {type(src)} have not been supported')
#
#         return pybel_mol.OBMol
#
#     @staticmethod
#     def obmol_to_arrays(obmol):
#         # Initialize lists to store atom and bond information
#         atoms = []
#         bonds = []
#         idx_to_row = {}
#
#         # Populate the idx_to_row dictionary to map OBMol atom indices to the reordered indices
#         for i, atom in enumerate(ob.OBMolAtomIter(obmol)):
#             idx_to_row[atom.GetIdx()] = i
#
#         # Iterate over atoms in the molecule
#         for i, atom in enumerate(ob.OBMolAtomIter(obmol)):
#             atomic_number = atom.GetAtomicNum()
#             formal_charge = atom.GetFormalCharge()
#             partial_charge = atom.GetPartialCharge()
#             is_aromatic = int(atom.IsAromatic())
#             x, y, z = atom.GetX(), atom.GetY(), atom.GetZ()
#             valence = atom.GetTotalValence()
#             implicit_hydrogens = atom.GetImplicitHCount()
#             # explicit_hydrogens = atom.GetExplicitHCount()
#
#             # Append the atom information to the atoms list
#             atoms.append([
#                 atomic_number, formal_charge, partial_charge,
#                 is_aromatic, x, y, z, valence, implicit_hydrogens,
#                 # explicit_hydrogens
#             ])
#
#         # Iterate over bonds in the molecule
#         for bond in ob.OBMolBondIter(obmol):
#             begin_atom_idx = idx_to_row[bond.GetBeginAtomIdx()]  # Map to reordered index
#             end_atom_idx = idx_to_row[bond.GetEndAtomIdx()]  # Map to reordered index
#             bond_order = bond.GetBondOrder()
#             is_aromatic = bond.IsAromatic()
#
#             # Append the bond information to the bonds list
#             bonds.append([begin_atom_idx, end_atom_idx, bond_order, is_aromatic])
#
#         # Convert lists to numpy arrays
#         atoms_array = np.array(atoms)
#         bonds_array = np.array(bonds)
#
#         return atoms_array, bonds_array, idx_to_row
#
#     @staticmethod
#     def arrays_to_obmol(atoms_array, bonds_array):
#         obmol = ob.OBMol()
#
#         # Create atoms in the molecule and set their properties
#         row_to_idx = {}
#         for i, atom_data in enumerate(atoms_array):
#             atomic_number, formal_charge, partial_charge, is_aromatic, x, y, z, valence, impH = atom_data
#             atom = obmol.NewAtom()
#             atom.SetAtomicNum(int(atomic_number))
#             atom.SetFormalCharge(int(formal_charge))
#             atom.SetPartialCharge(float(partial_charge))
#             atom.SetVector(x, y, z)
#             atom.SetAromatic(bool(is_aromatic))  # Convert to bool
#
#             # Store mapping from label to atom index (1-based indexing in OBMol)
#             row_to_idx[i] = atom.GetIdx()
#
#         # Create bonds in the molecule
#         for bond_data in bonds_array:
#             begin_atom_idx, end_atom_idx, bond_order, is_aromatic = bond_data
#             obmol.AddBond(
#                 row_to_idx[begin_atom_idx],
#                 row_to_idx[end_atom_idx],
#                 int(bond_order)
#             )
#             bond = obmol.GetBond(row_to_idx[begin_atom_idx], row_to_idx[end_atom_idx])
#             bond.SetAromatic(bool(is_aromatic))  # Convert to bool
#
#         return obmol, row_to_idx
#
#     @staticmethod
#     def ob_dump(mol: Union[ob.OBMol, pb.Molecule], fmt, **kwargs) -> Union[str, bytes]:
#         """
#         dump an openbabel molecule information to string or bytes
#         Args:
#             mol:
#             fmt:
#             filepath:
#             **kwargs:
#
#         Returns:
#
#         """
#         if isinstance(mol, ob.OBMol):
#             mol = pb.Molecule(mol)
#         elif not isinstance(mol, pb.Molecule):
#             raise TypeError(f'the mol type {type(mol)} is not supported, just allow openbabal.OBMol or pybel.Molecule')
#
#         return mol.write(fmt, **kwargs)
