"""Ordered ESNUEL nucleophilic-site rules used by the MeCAP model."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True, slots=True)
class DetectedSite:
    atom_index: int
    site_type: str


# Rule order is scientific behavior: the first matching rule names each site.
NUCLEOPHILE_RULES = (
    ("Ether", "[OX2:1]([#6;!$(C([OX2])[#7,#8,#15,#16,F,Cl,Br,I]);!$([#6]=[#8]):2])[#6;!$(C([OX2])[#7,#8,#15,#16]);!$([#6]=[#8]):3]"),
    ("Ketone", "[OX1H0:1]=[#6X3:2]([#6;!$([CX3]=[CX3;!R]):3])[#6;!$([CX3]=[CX3;!R]):4]"),
    ("Amide", "[OX1:1]=[CX3;$([CX3][#6]),$([CX3H]):2][#7X3;!R:3]"),
    ("Enolate", "[#6;$([#6]=,:[#6]-[#8-]),$([#6-]-[#6]=,:[#8]):1]~[#6:2]~[#8;$([#8-]-[#6]=,:[#6]),$([#8]=,:[#6]-[#6-]):3]"),
    ("Aldehyde", "[OX1:1]=[$([CX3H][#6;!$([CX3]=[CX3;!R])]),$([CX3H2]):2]"),
    ("Imine", "[NX2;$([N][#6]),$([NH]);!$([N][CX3]=[#7,#8,#15,#16]):1]=[CX3;$([CH2]),$([CH][#6]),$([C]([#6])[#6]):2]"),
    ("Nitranion", "[#7X2-:1]"),
    ("Carbanion", "[#6-;!$([#6X1-]#[#7,#8,#15,#16]):1]"),
    ("Nitronate", "[#6:1]=[#7+:2](-[#8-:3])-[#8-:4]"),
    ("Ester", "[OX1:1]=[#6X3;!$([#6X3][CX3]=[CX3;!R]);$([#6X3][#6]),$([#6X3H]):2][#8X2H0:3][#6;!$(C=[O,N,S]):4]"),
    ("Carboxylic acid", "[OX1:1]=[CX3;$([R0][#6]),$([H1R0]):2][$([OX2H]),$([OX1-]):3]"),
    ("Amine", "[#7+0;$([N;R;!$([#7X2]);$(N-[#6]);!$(N-[!#6;!#1]);!$(N-C=[O,N,S])]),$([NX3+0;!$([#7X3][CX3;$([CX3][#6]),$([CX3H])]=[OX1])]),$([NX4+;!$([N]~[!#6]);!$([N]*~[#7,#8,#15,#16])]):1]"),
    ("Cyanoalkyl/nitrile anion", "[C:1]=[C:2]=[#7X1-:3]"),
    ("Nitrile", "[NX1:1]#[CX2;!$(CC=C=[#7X1-]);!$(CC=C):2]"),
    ("Isonitrile", "[CX1-:1]#[NX2+:2]"),
    ("Phenol", "[OX2H:1][$(c(c)c),$([#6X3;R](=[#6X3;R])[#6X3;R]):2]"),
    ("Silyl_ether", "[#8X2H0:1][#14X4:2]([!#1:3])([!#1:4])[!#1:5]"),
    ("Pyridine_like_nitrogen", "[#7X2;$([nX2](:*):*),$([#7X2;R](=[*;R])[*;R]):1]"),
    ("anion_with_charge_minus1", "[*-:1]"),
    ("double_bond", "[*;!$([!X4;!#1;!#6:1])+0:1]=[*+0:2]"),
    ("double_bond_neighbouratom_with_charge_plus1", "[*;!$([!X4;!#1;!#6:1])+0:1]=[*+1:2]"),
    ("triple_bond", "[*;!$([!X4;!#1;!#6:1])+0:1]#[*+0:2]"),
    ("triple_bond_neighbouratom_with_charge_plus1", "[*;!$([!X4;!#1;!#6:1])+0:1]#[*+1:2]"),
    ("atom_with_lone_pair", "[!X4;!#1;!#6:1]"),
)


def find_nucleophilic_sites(mol: Chem.Mol) -> tuple[DetectedSite, ...]:
    matching_mol = Chem.AddHs(Chem.Mol(mol, True))
    Chem.Kekulize(matching_mol)
    sites = []
    names = []
    for name, smarts in NUCLEOPHILE_RULES:
        pattern = Chem.MolFromSmarts(smarts)
        for match in matching_mol.GetSubstructMatches(pattern, uniquify=False):
            site = match[0]
            if site not in sites:
                sites.append(site)
                names.append(name)

    ranks = list(Chem.CanonicalRankAtoms(matching_mol, breakTies=False))
    kept_ranks = set()
    result = []
    for atom_index, name in zip(sites, names):
        rank = ranks[atom_index]
        if rank not in kept_ranks:
            kept_ranks.add(rank)
            result.append(DetectedSite(atom_index, name))
    return tuple(result)
