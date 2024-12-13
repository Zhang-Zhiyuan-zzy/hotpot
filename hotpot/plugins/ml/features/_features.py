"""
python v3.9.0
@Project: hp5
@File   : features
@Auther : Zhiyuan Zhang
@Data   : 2024/6/28
@Time   : 14:56
"""
from tqdm import tqdm

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors


def get_rdkit_mol_descriptor(list_smi, mol_names=None) -> pd.DataFrame:
    attr3d = [
        'InertialShapeFactor', "NPR1", 'NPR2', 'Asphericity', 'Eccentricity',
        'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'
    ]

    list_mol = []
    for smi in tqdm(list_smi, 'Initializing Molecule...'):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        list_mol.append(mol)

    ligand_2d = pd.DataFrame([CalcMolDescriptors(m) for m in tqdm(list_mol, 'calc 2d descriptors...')])
    ligand_3d = pd.DataFrame([{n: getattr(Chem.Descriptors3D, n)(mol) for n in attr3d}
                              for mol in tqdm(list_mol, 'calc 3d descriptors...')])

    descriptors = pd.concat([ligand_2d, ligand_3d], axis=1)
    if mol_names:
        descriptors.index = mol_names

    return descriptors
