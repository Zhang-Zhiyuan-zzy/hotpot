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

_RDKIT_DATIVE_BOND_TYPES = frozenset(
    bond_type for bond_type in (
        getattr(Chem.BondType, "DATIVE", None),
        getattr(Chem.BondType, "DATIVEONE", None),
        getattr(Chem.BondType, "DATIVEL", None),
        getattr(Chem.BondType, "DATIVER", None),
    )
    if bond_type is not None
)


def _rdkit_bond_type(bond):
    kind_to_type = {
        "zero": Chem.BondType.ZERO,
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
    }
    if bond.bond_kind.value == "dative":
        raw_type_name = bond.bond_source_metadata.get("raw_bond_type")
        raw_type = (
            getattr(Chem.BondType, raw_type_name, None)
            if raw_type_name is not None
            else None
        )
        if raw_type in _RDKIT_DATIVE_BOND_TYPES:
            return raw_type
        return Chem.BondType.DATIVE
    if bond.bond_kind.value == "aromatic" or bond.is_aromatic:
        return Chem.BondType.AROMATIC
    if bond.bond_kind.value in kind_to_type:
        return kind_to_type[bond.bond_kind.value]
    return Chem.BondType.values[bond.bond_order]


def _rdkit_bond_kind(bond_type, is_aromatic):
    if is_aromatic or bond_type == Chem.BondType.AROMATIC:
        return "aromatic"
    if bond_type == Chem.BondType.ZERO:
        return "zero"
    if bond_type in _RDKIT_DATIVE_BOND_TYPES:
        return "dative"
    return {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
    }.get(bond_type, "unknown")


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
        begin_atom_idx, end_atom_idx = bond.atom1.idx, bond.atom2.idx
        if bond.bond_kind.value == "dative":
            if bond.bond_direction is None:
                raise ValueError(
                    "RDKit dative-bond export requires a bond direction"
                )
            if bond.bond_direction == "atom2_to_atom1":
                begin_atom_idx, end_atom_idx = end_atom_idx, begin_atom_idx
            elif bond.bond_direction != "atom1_to_atom2":
                raise ValueError(
                    f"Unsupported dative-bond direction: {bond.bond_direction}"
                )
        rdmol.AddBond(
            row_to_idx[begin_atom_idx],
            row_to_idx[end_atom_idx],
            _rdkit_bond_type(bond),
        )

    rdmol = rdmol.GetMol()
    if kekulize:
        Chem.Kekulize(rdmol)

    if sanitize:
        Chem.SanitizeMol(rdmol)
    # Chem.AssignStereochemistry(rdmol)  TODO: after stereo module in hotpot

    return rdmol


def from_rdmol(rdmol, mol):
    """Populate a Hotpot molecule from an RDKit molecule."""
    rdmol = Chem.Mol(rdmol)
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

    for atom in rdmol.GetAtoms():
        coordinates = (
            conformer.GetAtomPosition(atom.GetIdx())
            if conformer is not None
            else (0.0, 0.0, 0.0)
        )
        mol._create_atom(
            atomic_number=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            partial_charge=(
                atom.GetDoubleProp('_GasteigerCharge')
                if atom.HasProp('_GasteigerCharge')
                else 0.0
            ),
            is_aromatic=atom.GetIsAromatic(),
            coordinates=coordinates,
            valence=atom.GetTotalValence(),
            implicit_hydrogens=(
                atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            ),
        )

    for bond in rdmol.GetBonds():
        source_bond_type = source_bond_types[bond.GetIdx()]
        bond_kind = _rdkit_bond_kind(
            source_bond_type,
            source_bond_metadata[bond.GetIdx()]["is_aromatic"],
        )
        mol._add_bond(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_order=bond.GetBondTypeAsDouble(),
            bond_kind=bond_kind,
            bond_direction=(
                "atom1_to_atom2"
                if source_bond_type in _RDKIT_DATIVE_BOND_TYPES
                else None
            ),
            bond_source="rdkit",
            bond_source_metadata=source_bond_metadata[bond.GetIdx()],
        )

    mol._update_graph()
    mol.charge = Chem.GetFormalCharge(rdmol)
    return mol
