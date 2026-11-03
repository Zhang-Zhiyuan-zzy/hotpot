"""External RDKit copies and binary encoding for prepared inference input."""
from rdkit import Chem


def copy_feature_molecule(molecule, *, sanitize: bool):
    molecule = Chem.Mol(molecule)
    if sanitize:
        Chem.SanitizeMol(molecule)
    return molecule


def encode_molecule(molecule) -> bytes:
    return bytes(molecule.ToBinary(Chem.PropertyPickleOptions.AllProps))


def decode_molecule(binary: bytes):
    return Chem.Mol(binary)
