"""
python v3.9.0
@Project: hotpot
@File   : rdconvert
@Auther : Zhiyuan Zhang
@Data   : 2024/12/18
@Time   : 8:56
"""
import rdkit
from rdkit import Chem

_hphyb2rdkithyb = {
    -1: rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
    0: rdkit.Chem.rdchem.HybridizationType.S,
    1: rdkit.Chem.rdchem.HybridizationType.SP,
    2: rdkit.Chem.rdchem.HybridizationType.SP2,
    3: rdkit.Chem.rdchem.HybridizationType.SP3,
    4: rdkit.Chem.rdchem.HybridizationType.SP2D,
    5: rdkit.Chem.rdchem.HybridizationType.SP3D,
    6: rdkit.Chem.rdchem.HybridizationType.SP3D2,
    7: rdkit.Chem.rdchem.HybridizationType.OTHER
}


def to_rdmol(mol, kekulize: bool = True, sanitize: bool = False):
    rdmol = Chem.RWMol()

    # Create atoms in the molecule and set their properties
    row_to_idx = {}
    for i, atom in enumerate(mol.atoms):

        rda = Chem.Atom(atom.atomic_number)
        rda.SetFormalCharge(atom.formal_charge)
        rda.SetNumExplicitHs(atom.implicit_hydrogens)
        rda.SetDoubleProp('_GasteigerCharge', atom.partial_charge)
        rda.SetIsAromatic(atom.is_aromatic)
        rda.SetHybridization(_hphyb2rdkithyb[atom.hyb])

        idx = rdmol.AddAtom(rda)
        row_to_idx[i] = idx

    conf = Chem.Conformer(len(mol.atoms))
    for i, (x, y, z) in enumerate(mol.coordinates):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    rdmol.AddConformer(conf)

    for bond in mol.bonds:
        begin_atom_idx, end_atom_idx, bond_order = bond.atom1.idx, bond.atom2.idx, bond.bond_order
        rdmol.AddBond(
            row_to_idx[begin_atom_idx],
            row_to_idx[end_atom_idx],
            Chem.BondType.AROMATIC if bond.is_aromatic else Chem.BondType.values[bond_order]
        )

    rdmol = rdmol.GetMol()
    if kekulize:
        Chem.Kekulize(rdmol)

    if sanitize:
        Chem.SanitizeMol(rdmol)
    # Chem.AssignStereochemistry(rdmol)  TODO: after stereo module in hotpot

    return rdmol


def from_rdmol(rdmol, mol):
    # Get atom indices in a consistent order
    idx_to_row = {atom.GetIdx(): i for i, atom in enumerate(rdmol.GetAtoms())}

    # Iterate over atoms in the molecule
    for atom in rdmol.GetAtoms():
        mol.create_atom(
            atomic_number=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            partial_charge=atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0,
            is_aromatic=int(atom.GetIsAromatic()),
            coordinates=rdmol.GetConformer().GetAtomPosition(atom.GetIdx()),
            valence=atom.GetTotalValence(),
            implicit_hydrogens=atom.GetNumImplicitHs(),
        )

    # Iterate over bonds in the molecule
    for bond in rdmol.GetBonds():
        i = idx_to_row[bond.GetBeginAtomIdx()]  # Map to reordered index
        j = idx_to_row[bond.GetEndAtomIdx()]  # Map to reordered index
        bond_order = bond.GetBondTypeAsDouble()
        # is_aromatic = bond.GetIsAromatic()

        # Append the bond information to the bonds list
        mol.add_bond(i, j, bond_order=bond_order)

    return mol
