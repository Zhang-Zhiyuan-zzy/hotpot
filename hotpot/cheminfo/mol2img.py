# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : mol2img
 Created   : 2025/7/30 11:22
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 Make 2d structural image give the Molecule structure
===========================================================
"""
from textwrap import dedent
from typing import Union, Iterable

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


def _find_single_maximum_subs(*smiles: str):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    res = rdFMCS.FindMCS(mols)
    return Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmarts(res.smartsString)))

def choose_best_colnum(n_mols, min_col=4, max_col=7, default_col=5):
    best_col = default_col
    best_fill = 0       # 填充率
    best_distance = abs(default_col-default_col)  # 初始化距离
    for colnum in range(min_col, max_col + 1):
        rows = (n_mols + colnum - 1) // colnum
        filled = (rows-1) * colnum
        last_row_filled = n_mols - filled if n_mols - filled > 0 else colnum
        fill_rate = last_row_filled / colnum
        distance = abs(colnum - default_col)
        # 优先填充率高者，若并列取更接近5者
        if (fill_rate > best_fill) or (fill_rate == best_fill and distance < best_distance):
            best_fill = fill_rate
            best_col = colnum
            best_distance = distance
    return best_col

def draw_grid(list_mols: Iterable[Union[str, "Molecule"]], save_path=None, mol_size=(600, 600)):
    from hotpot.cheminfo.core import Molecule

    list_mols = [m.smiles if isinstance(m, Molecule) else m for m in list_mols]

    # 0. Create Molecules
    mols = [Chem.MolFromSmiles(sm) for sm in list_mols]
    mols = [m for m in mols if m is not None]
    for m in mols:
        tmp = AllChem.Compute2DCoords(m)

    # Search shared substructures
    # shared_sub = _find_single_maximum_subs(*list_smiles)
    # for m in mols:
    #     if m.HasSubstructMatch(shared_sub):
    #         _ = AllChem.GenerateDepictionMatching2DStructure(m, shared_sub)

    # 2. Apply ACS Document 1996 style and set styles
    d2d_style = rdMolDraw2D.MolDrawOptions()
    d2d_style.useACS1996Style = True  # 启用ACS 1996样式
    # d2d_style.bondLineWidth = 6
    d2d_style.atomPalette = {
        7: (0, 0, 1),   # N: blue
        8: (1, 0, 0),   # O: red
    }

    # 3. Calculate the n_cols
    n_cols = choose_best_colnum(len(mols))

    # 4. Generate 2d Image
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=None,
        useSVG=False,
        returnPNG=False,
        drawOptions=d2d_style
    )

    # 4. Save image
    if save_path:
        img.save(save_path)
        print(f"Molecular 2D images have saved in: {save_path}")

    return img


if __name__ == "__main__":
    smiles_list = \
        dedent("""OP(=O)(C1=C2N=CC=C2[C@@H]2C(=N1)C1=C(C=C2)[C@]23[C@@](C(=N1)c1ccccn1)(CCCCC2)[C@]1(C([C@@]3(C)CC1)(C)C)C)O
        OP(=O)(C1=Nc2c3N=C(C4=Nc5c(C4)cc[nH]5)[C@H]4[C@@]5(c3ccc2[C@H]2[C@@H]1CCC2)CCCC5=CCCC4)O
        OS(=O)(c1nc2c3nc(ccc3ccc2c2c1cc[nH]2)C1=C[C@H]2[C@@H](S1)[C@H]1CC[C@]31[c]1(c2[nH]cc1)CCCC3)O
        CC1(C)C=CC([C@H]2[C@@H]1[C@H]1N=C(c3ccc4c(n3)c3nc(ccc3cc4)S(=O)(O)O)[C@H]3[C@H]([C@@H]1CC2)CCCC3)(C)C
        OP(=O)(c1ccc2c(n1)c1N=C(C3=NC4=C[C@H]5C(=C6C(=N5)C=CN6)C=C4C3)[C@H]3[C@H](c1cc2)[C@]1(C)CC[C@@]3(C1(C)C)C)O
        NC(=O)c1nc2c(c3c1[C@@H]1CCCC[C@H]1N3)ccc1c2N=C(C2=NC=C3[C@@H]2CCCC3)[C@H]2[C@@H]1C(C)(C)CCC2(C)C
        C1CC[c]23[c](-c4ccc5c(c4N=C3c3ncc4c(c3)CCC4)N=C(C3=CN=C[C@@H]53)C3=NC=CC3)(CC1)cccc2
        CC1(C)[C@H]2[C@H]3[C@@H](CC[C@H]2C([C@H]2[C@H]1CCC2)(C)C)[C@H]1[C@@H]3C(=Nc2c1ccc1c2nc(c2c1cc[nH]2)c1cccs1)c1ccccn1
        OP(=O)(C1=Nc2c3nc(ccc3ccc2[C@]23[C@@H]1CC=C3C=CC=C2)C1=Nc2c(C1)c[nH]c2)O
        OS(=O)(c1ccc2c(n1)c1nc(ccc1cc2)C1=C[C@H]2[C@@H](S1)c1ccnc1C[C@H]1[C@H]2CCC1)O
        CC1(C)[C@@]2(C)CC[C@@]1(C)[C@@H]1[C@@H]2Cc2c1csc2c1ccc2c(n1)c1nc(ccc1cc2)c1ncc2c(c1)CCCC2
        CC1(C)CCC([C@@H]2[C@@H]1[C@]13C(=Nc4c([C@@]3(C=C2)CCc2c1ncc2)ccc1c4nc(cc1)c1cccs1)C1=NCC=C1)(C)C
        C[C@@]12CC[C@](C2(C)C)(c2c1c1ccc(nc1c1c2ccc(n1)c1ncc2c(c1)CCC2)c1sc2c(c1)cccc2)C
        CC1(C)[C@H]2[C@@H](c3ccc4c(c3N=C2P(=O)(O)O)nc(c2c4cccc2)C2=NC=C[C@H]3[C@@H]2[C@@]2(C)CC[C@@]3(C2(C)C)C)C(c2c1cccc2)(C)C
        C[C@]12CC[C@](C2(C)C)([C@@]23[C@]1(CC2)C(=Nc1c3ccc2c1nc(c1cccs1)c1c2CC1)c1ccccn1)C
        C1CC[C@@H]2[C@H](CC1)[C@H]1CCC[C@H]1N=C2C1=Nc2c([C@H]3[C@@H]1CCC3)ccc1c2nc(cc1)C1=Nc2c(C1)cccc2
        OP(=O)(C1=Nc2c([C@H]3[C@@H]1[C@H]1C[C@@H]4CCC[C@H]4CC[C@@H]1CC3)ccc1c2nc(cc1)C1=NC=CC1)O
        [O-]C(=O)C1=Nc2c3N=C([C@H]4[C@H](c3ccc2[C@H]2[C]31=C(C=CC=C3)CC2)C(C)(C)CCC4(C)C)[P@@](=O)(O)N
        C1CC[C@]23[C@@H](C1)C[C@@H]2C(=Nc1c3ccc2c1nc(c1c2cccc1)c1cccs1)c1ccc2c([nH]1)ncc2
        OP(=O)(C1=Nc2c3N=C(C4=N[C@H]5[C@@H](C=C4)CCC5)C4=CN=C[C@H]4c3ccc2[C@H]2[C@@H]1[C@@]13[C@]2(CCCC1)[C@@]1(C([C@@]3(C)CC1)(C)C)C)O
        [O-]C(=O)c1ccc2c(n1)c1nc(ccc1c1c2C[C@H]2[C@H]1C(C)(C)CCC2(C)C)c1ccc2=C3C(=CN=C3)N=c2n1
        C1C[C@H]2[C@@H](C1)[C]13=C(CCC[C@@H]3N=C2c2ccc3c(n2)c2nc(ccc2cc3)C2=NC=CC2)c2c(N1)cc[nH]2
        C1C=CC(=N1)c1ccc2c(n1)c1nc(ccc1cc2)C1=C2C=NC=[C]32[C@H](S1)CCc1c3cc[nH]1
        [O-]C(=O)c1ccc2c(n1)c1N=C(C3=NC=C[C@@H]3c1cc2)c1nccc2c1nc1c2ncc1
        C1CC[C@@H]2[C@@H](CC1)[C@H]1[C@H](N=C2c2ccc3c(n2)c2N=C(c4cccs4)[C@@]45[C@](c2cc3)(CCCCC5)CCCC4)CCCc2c1ncc2
        c1ccc(nc1)c1nc2c3nc(ccc3ccc2c2c1cc[nH]2)C1=C[C@H]2[C@@H](S1)CCC1=C3C(=CC=N3)N=C21
        NC(=O)c1ccc2c(n1)c1N=C(c3ccccn3)[C@H]3[C@@H](c1cc2)[C@H]1[C@@H](C3)[C@@]2(C([C@@]1(C)[C@H]1[C@H]2C(C)(C)CCC1(C)C)(C)C)C
        C1CC[C@@H]2[C@@H](CC1)C1=C(C=C2)Cc2c1cc1c(c2)ccnc1c1ccc2c(n1)c1nc(ccc1cc2)C1=NC=CC1
        CC1(C)CC[C@@]([C@]23[C@@H]1[C@@H]1CC[C@@]21N=C(C3)c1ccc2c(n1)c1N=C(c3ccccn3)[C@H]3[C@@H](c1cc2)CC3)(C)N
        [O-]C(=O)c1ccc2c(n1)c1N=C([C@H]3[C@@H](c1cc2)C[C@H]1CCCCC[c]21c(C3)cc[nH]2)P(=O)(O)O
        C[C@]12CC[C@](C2(C)C)([C@@]23[C@]1(CC2)C(=Nc1c3ccc2c1N=C(C1=NC=CC1)C1=NC=C[C@@H]21)C1=NC=C[C@H]2[C@@H]1CCCCC2)C
        NC(=O)c1nc2c3=NC(=CCc3c3c(c2c2=CN=Cc12)CC1=CC=C2C(=C31)CC2)c1ccccn1
        c1ccc(nc1)C1=Nc2c([C@@H]3[C@@H]1CCC[C@H]1[C@H]3CCCC1)ccc1c2nc(C2=NC=CC2)c2c1cnc2
        C1CC[C@H]2[C@@H](CC1)[C@H]1[C@@H]2C=CN=C1c1nc2c3=NC(=CCc3ccc2c2=CC=Nc12)c1scc2c1C[C@@H]1CC[C@H]1C2
        NC(=O)c1ccc2c(n1)c1N=C(c3scc4c3C(C)(C)C3=CC=N[C@@H]3C4(C)C)[C@H]3[C]4(=C(CCC3)C=Cc3c4cccc3)c1cc2
        OS(=O)(c1nc2c3nc(ccc3ccc2c2c1C1=NC=CC1=C2)C1=Nc2ccccc2[C@H]2[C@@H]1CCC2)O
        [O-]C(=O)c1ccc2c(n1)c1nc(ccc1cc2)C1=N[C@@]23[C@@H](C1)CCC[C@@H]2CC[C@@H]1[c]23c[nH]cc2C1
        CC1(C)CCC([C@@H]2[C@@H]1C1=CC(=N[C@@H]3[C]1(=N2)CCC3)c1ccc2c(n1)c1nc(ccc1c1c2CC1)c1cc2c(s1)cccc2)(C)C
        CC1(C)CCC([C@H]2[C@@H]1c1c(-c3c(C2)cccc3)c2ccc(nc2c2c1ccc(n2)c1ccc2c([nH]1)ncc2)P(=O)(O)O)(C)C
        OS(=O)(C1=Nc2c3N=C(C4=CN=C[C@@H]4c3c3c(c2[C@H]2[C@@H]1CCCC2)CCCCC3)c1scc2c1[C@@]1(C)CC[C@@]2([C@@]1(C)O)C)O
        C1CC[C@]23[C@@](CC1)(CCC[C@H]1[C@@H]3CCCC1)SC(=C2)c1ccc2c(n1)c1nc(ccc1cc2)C1=NC[C@H]2[C@@H]1C=CC2
        CC1(C)CC[C@]([C@H]2[C@@H]1C(C)(C)[C@H]1[C@@H](C2(C)C)CN=C1c1ccc2c(n1)c1nc(ccc1cc2)c1scc2c1CCCCC2)(C)N
        CC1(C)CCC([C@H]2[C@@H]1c1c(N=C2P(=O)(O)O)c2N=C([C@H]3[C@H](c2c2c1CCCC=C2)[C@]1(C)CC[C@@]3(C1(C)C)C)S(=O)(O)O)(C)C
        CC1(C)CCC([C@H]2[C@@H]1[C@](C)(N)c1c(C2(C)C)cnc(c1)C1=Nc2c([C@H]3[C@@H]1CC3)ccc1c2nc(cc1)S(=O)(O)O)(C)C
        NC(=O)C1=NC2=C([C@@]34[C@@H]1CCC[C@@H]3C[C@H]1[C@@H]4C(C)(C)CCC1(C)C)[C@H]1[C@@H](c3c2nc(cc3)C2=NC=C3[C@H]2[C@]2(C)CC[C@@]3(C2(C)C)C)C(C)(C)CCC1(C)C
        C[C@]12CC[C@](C2(C)C)([C@@H]2[C@@H]1C=C(S2)C1=Nc2c3nc(ccc3ccc2[C@@]23[C@@]1(CCCCC2)[C@@]1(C)CC[C@@]3(C1(C)C)C)C1=N[C@H]2C(=C1)CC2)C
        OP(=O)(c1nc2-c3nc(ccc3[C@H]3[C@H](c2c2c1N[C@H]1[C@@H]2[C@]2(C([C@@]1(C)CC2)(C)C)C)CCC3)C1=N[C@H]2[C@@H](C=C1)Cc1c2cccc1)O
        [O-]C(=N)c1nc2c3=NC(=CCc3c3c(c2c2=CN=Cc12)cccc3)c1scc2c1CCC2
        OS(=O)(c1ccc2c(n1)c1N=C(C3=NC4=C(C3)CCC[C@H]3[C@@H]4[C@@]4(N)CC[C@@]3(C4(C)C)C)C3=NC=C[C@@H]3c1cc2)O
        C[C@@]12CC[C@@](C1(C)C)([C]13=[C]2(C=CN1)CC(=N3)C1=Nc2c([C@H]3[C@H]1CCCCC3)ccc1c2nc(cc1)c1ccccn1)N
        C1C[C@H]2[C@@H](C1)CC(=N2)c1ccc2c(n1)c1N=C(C3=N[C@@H]4[c]5(C=C3)cc[nH]c5CC4)C3=CN=C[C@@H]3c1cc2
        C1CCc2c(CC1)c1cncc1nc2c1ccc2c(n1)c1N=C(c3cccs3)[C@@]34[C@](c1cc2)(CCC3)CC4""")
    smiles_list = [s.strip() for s in smiles_list.splitlines()]

    mol_image = draw_grid(smiles_list, save_path="/mnt/d/zhang/OneDrive/Desktop/mol2img1.png")
