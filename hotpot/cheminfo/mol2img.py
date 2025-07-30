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

def draw_grid(list_mols: Iterable[Union[str, "Molecule"]], save_path=None, mol_size=(300, 300)):
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
        dedent("""OS(=O)(c1ccc2c(n1)c1nc(C3=N[C@H]4[C@@H](C3)[C@@]3(C)CN[C@@]4(C3(C)C)C)c3c(c1cc2)[C@@]1(C)CC[C@@]3(C1(C)C)C)O
            CC1(C)CCC(c2c1cc1c(c2)c2ccc(nc2c2c1ccc(n2)S(=O)(O)O)c1sc2c(c1)CCC2)(C)C
            OP(=O)(c1nc2c(c3c1CC3)c1CCc1c1c2nc(cc1)C1=NC=C2[C@@H]1[C@@]1(C)CC[C@@]2(C1(C)C)C)O
            C1CC[C@H]2[C@@H](CC1)CC(=N2)C1=Nc2c([C@H]3C1=N[C@H]1[C@H]3CC1)ccc1c2nc(cc1)c1ccccn1
            NC(=O)c1ccc2c(n1)c1nc(C3=NC=C4[C@@H]3CCC4)c3c(c1cc2)[C@@]1(C)NC[C@@]3(C1(C)C)C
            [O-]C(=O)c1ccc2c(n1)c1nc(C3=NC=C4C3=Nc3c4[nH]cc3)c3c(c1cc2)CCC3
            CC1(C)CCC(c2c1c(nc1c2ccc2c1nc(cc2)C1=N[C@H]2[C@@H](C1)CCC2)c1ncc2c(c1)CCC2)(C)C
            C1=CN=C(C1)c1ccc2c(n1)c1nc(ccc1c1c2CC1)c1nccc2c1ccc1c2cc[nH]1
            NC(=O)c1ccc2c(n1)c1nc(ccc1c1c2C(C)(C)CCC1(C)C)c1scc2c1CC=C2
            NC(=O)c1ccc2c(n1)c1nc(C(=O)[O-])c3c(c1c1c2CC1)CC=C3
            NC(=O)c1ccc2c(n1)c1nc(ccc1c1c2-c2cc[nH]c2CCC1)c1nccc2c1CCC2
            NC(=O)c1ccc2c(n1)c1nc(ccc1cc2)C1=NC=C2[C@@H]1C[C@H]1[C@]3([C@@H](C2)CCCCC3)[C@]2(C([C@@]1(C)CC2)(C)C)C
            NC(=O)c1ccc2c(n1)c1nc(C3=NC#CN3)c3c(c1cc2)C(C)(C)CCC3(C)C
            [NH-]C(=O)c1ccc2c(n1)c1nc(c3cccs3)c3c(c1cc2)CCCCC3
            [O-]C(=O)c1nc2c3nc(c4c(c3c3c(c2c2c1ncc2)[C@]1(C)CC[C@@]3(C1(C)C)C)CCCC4)P(=O)(O)O
            CC1(C)CCC([C@H]2[C@@H]1c1c3ccc4c(c3nc(c21)c1scc2c1cnc2)nc(cc4)c1ccccn1)(C)C
            OS(=O)(c1ccc2c(n1)c1nc(ccc1c1-c3c(CCc21)c1c([nH]3)CCCCC1)c1ccccn1)O
            [O-]C(=O)c1ccc2c(n1)c1nc(ccc1c1c2CCCCC1)C1=N[C@H]2[C@]3([C@@H]1CCCCC3)CCCCC2
            OP(=N)(c1ccc2c(n1)c1nc(ccc1c1c2ccc2c1ncc2)C1=NC=CC1)O
            OP(=O)(c1nc2c3nc(c4c(c3ccc2c2c1[C@@]1(C)CC[C@@]2([C@]1(C)N)C)ncc4)S(=O)(O)O)O
            [O-]C(=O)c1ccc2=C3C=NC=C3[C@@H]3C(=c2n1)N=C(c1ccccn1)C1=C3[C@@H]2CCCCC[C@H]2CC1
            [O-]C(=O)c1ccc2c(n1)c1nc(c3ccccn3)c3c(c1c1c2CCNC1)cnc3
            OP(=O)(c1ccc2c(n1)c1nc(ccc1cc2)c1sc2c(-c3cc[nH]c3CC2)c1)O
            c1ccc(nc1)c1ccc2c(n1)c1nc(ccc1cc2)C1=NC=C2C1=CCCC2
            [O-]C(=O)c1nc2c(c3c1ccc1c3CCCCC1)c1cc[nH]c1c1c2nc(cc1)P(=O)(O)O
            CC1(C)CCC(C2=CN=C([C@H]12)c1ccc2c(n1)c1N=C(C3=C4C=NC=C4C=CN3)C3=CN=C[C@@H]3c1cc2)(C)C
            OS(=O)(c1ccc2c(n1)c1nc(ccc1cc2)C1=C2C=C3C(=C2c2c(N1)[nH]cc2)CCCC3)O
            N1=CC2=c3c(=C4[C@@H](C2=C1)C=CC(=N4)c1ccc2c(n1)CC2)nc(c1c3cc[nH]1)C1=NC=CC1
            C1C=CC(=N1)c1nc2c3nc(ccc3ccc2c2c1cc[nH]2)c1cc2c(s1)C=CC2
            [O-]C(=O)c1ccc2c(n1)c1nc(c3nccc4c3CCC4)c3c(c1cc2)Cc1ccccc1CC3
            CC1(C)CCC(c2c1c1ccc3c(c1nc2C1=NCC=C1)nc(cc3)c1cc2c(s1)C[C@@H]1[C@H]2CCCC1)(C)C
            CC1(C)CCC(c2c1c1ccc3c(c1nc2c1cccs1)nc(c1c3CCC1)c1ncc2c(c1)ccn2)(C)C
            CC1(C)CCC([C@@H]2[C@@H]1[C@@]1(C)c3c([C@]2(C1(C)C)C)ccnc3c1ccc2c(n1)c1nc(ccc1cc2)C1=N[C@H]2[C@@H](C1)CC2)(C)C
            OS(=O)(c1ccc2c(n1)c1nc(c3ccccn3)c3=c4c(=Nc3c1c1=NC=Cc21)ncc4)O
            CC1(C)CCC([C@@H]2[C@@H]1Cc1c(CC2)c(nc2c1ccc1c2nc(c2c1ncc2)c1cccs1)c1ccccn1)(C)C
            C[C@]12CC[C@@](C2(C)C)(c2c1ccnc2c1ccc2c(n1)c1nc(ccc1cc2)c1cc2c(s1)c1CCCCc1cc2)C
            NC(=O)c1nc2c(c3c1C[C@H]1[C@H]3CC1)ccc1c2nc(C2=NC=CC2)c2c1[C@@]1(C)CC[C@@]2(C1(C)C)C
            OP(=O)(c1ccc2c(n1)c1nc(ccc1c1c2cc[nH]1)C1=N[C@H]2[C@@H](C1)[C@@]1(C)[C@H]3[C@@H]([C@@]2(C1(C)C)C)C(C)(C)CCC3(C)C)O
            [O-]C(=O)c1nc2c(c3c1CCN3)ccc1c2nc(cc1)c1nccc2c1CCC2
            [O-]C(=O)c1ccc2c(n1)c1nc(ccc1c1c2C[C@@]23CCCCC[C@H]3C[C@H]2CC1)c1cccs1
            [O-]C(=O)c1ccc2c(n1)c1nc(c3ccccn3)c3c(c1cc2)CC[C@H]1[C@@H](C3)C=CCCC1
            [O-]C(=O)c1nc2c(c3c1CCC3)ccc1c2nc(C(=O)[O-])c2c1cc[nH]2
            NC(=O)c1nc2c3nc(ccc3ccc2c2c1ncc2)c1ncc2c(c1)CCC=CC2
            NC(=O)c1ccc2c(n1)c1nc(ccc1c1c2CCC1)c1ncc2c(c1)[C@@]1(C)[C@H]3[C@@H]([C@@]2(C1(C)C)C)C(C)(C)CCC3(C)C
            CC1(C)CCC([C@@H]2[C@@H]1N=C(C2)c1ccc2c(n1)c1nc(ccc1c1c2C(C)(C)CCC1(C)C)c1ncc2c(c1)CCC2)(C)C
            NC(=O)c1ccc2c(n1)c1nc(ccc1c1c2cc[nH]1)c1nccc2c1CCC[C@H]1[C@H]2[C@]2(C)CC[C@@]1(C2(C)C)C
            [O-]C(=O)c1ccc2c(n1)c1nc(c3ccccn3)c3c(c1c1c2CC[C@H]2[C@@H]1CCC2)[C@]1(C)CC[C@@]3(C1(C)C)C
            CC1(C)CCC(c2c1c1ccc(nc1c1c2ccc(n1)P(=O)(O)O)c1ccc2c(n1)CCc1c(C2)cc[nH]1)(C)C
            C1CCc2c(CC1)cc(nc2)c1ccc2c(n1)c1nc(ccc1cc2)[C@H]1SC=C2C1=CN=C2
            CC1(C)CCC(c2c1c(nc1c2ccc2c1nc(cc2)c1cccs1)C1=NC[C@H]2[C@@H]1Cc1cc[nH]c1CC2)(C)C
            c1cc2c(n1)cc(nc2)c1ccc2c(n1)c1nc(ccc1cc2)C1=N[C@H]2[C@@]3([C@H]1CCCC3)CCC2
            CC1(C)CCC([C@@H]2C1=C(N=c1c2ccc2c1nc(c1ccccn1)c1c2=CC2=C1CCC2)c1cccs1)(C)C
            NC(=O)c1ccc2c(n1)c1nc(ccc1cc2)c1scc2c1C(C)(C)[C@H]1[C@@H](C2(C)C)[C@@]2(C([C@@]1(C)[C@H]1[C@H]2CCCC1)(C)C)C
            [NH-]C(=O)c1ccc2c(n1)c1nc(c3c(c1cc2)[C@]1(C)[C@H]2[C@H]([C@]3(C1(C)C)C)C(C)(C)CCC2(C)C)P(=O)(O)O
            NC(=O)c1ccc2c(n1)c1N=C(C(=O)[O-])C3=C=C=N[C@@H]3c1c1c2CCC1
            NC(=O)c1ccc2c(n1)c1nc(c3cccs3)c3c(c1cc2)[C@H]1CCCC=C[C@@H]1C3
            OS(=O)(c1ccc2c(n1)c1nc(c3cccs3)c3c(c1cc2)[C@H]1CCCC=C1CC3)O
            NC(=O)c1ccc2c(n1)c1nc(c3cc4c(s3)CCCCC4)c3c(c1cc2)CCCNC3
            CC1(C)CCC([C@@H]2[C@@H]1C[C@@]13[C@H](CC2)CCCCC1=CN=C3c1ccc2c(n1)c1nc(ccc1cc2)P(=O)(O)O)(C)C
            c1ccc(nc1)c1nc2c(c3c1CCCN3)ccc1c2nc(C2=NC=CC2)c2c1cccc2
            OS(=O)(c1ccc2c(n1)c1nc(ccc1cc2)c1cc2c(-c3ccccc3[C@H]3[C@@H]2CCC3)s1)O
            NC(=O)c1ccc2c(n1)c1nc(c3c(c1cc2)Cc1c(C3)c2c(c1)CCCCC2)S(=O)(O)O
            NC(=O)c1ccc2c(n1)c1nc(c3c(c1c1c2cc2c1ncc2)CCCCC3)S(=O)(O)O
            NC(=O)c1nc2c3nc(c4c(c3ccc2c2c1CC2)c1c(c4)C(C)(C)CCC1(C)C)P(=O)(O)O
            CC1(C)CCC(c2c1c1ccc(nc1c1c2ccc(n1)C1=N[C@H]2[C@@H](C1)CCC2)c1nccc2c1C(C)(C)CCC2(C)C)(C)C
            C[C@]12CC[C@@](C2(C)C)(c2c1ccnc2c1nc2c(c3c1CCCC3)ccc1c2nc(cc1)c1sc2c(c1)CCC2)C
            [O-]C(=O)c1ccc2c(n1)c1N=C(c3cccs3)C3=N[c]45[c]([C@@H]3c1cc2)(cc[nH]4)C(C)(C)CCC5(C)C
            OS(=O)(c1ccc2c(n1)c1N=C(c3ccccn3)C3=NC=C[C@@H]3c1c1c2c2ccccc2c1)O
            NC(=O)c1nc2c(c3c1CC3)ccc1c2nc(cc1)c1sc2c(c1)C(C)(C)[C@H]1[C@@H](C2(C)C)CCC1
            OS(=O)(c1ccc2c(n1)c1nc(C3=NC=CC3)c3c(c1c1c2C[C@H]2[C@H]1CC2)[C@@]1(C)CC[C@@]3(C1(C)C)C)O
            OP(=O)(c1nc2c(c3c1CCC3)ccc1c2nc(C2=NC=CC2)c2c1ccc1c2[C@]2(C)CC[C@@]1(C2(C)C)C)O
            CC1(C)CCC(c2c1ccnc2c1ccc2c(n1)c1nc(C3=NC=C4C3=CN=C4)c3c(c1cc2)[C@@]1(C)CC[C@@]3(C1(C)C)C)(C)C
            [O-]C(=O)c1ccc2c(n1)c1nc(ccc1c1-c3c(Cc21)cccc3)C1=NC=CN1
            [O-]C(=O)c1ccc2c(n1)c1nc(ccc1cc2)C1=C[C@@H]2C(=NC3=C2[C@]2(N)CC[C@@]3(C2(C)C)C)S1
            CC1(C)[C@@]2(C)CC[C@@]1(C)c1c2cc(s1)c1nc2c3nc(ccc3ccc2c2c1CCCC2)C1=N[C@H]2C(=C1)CC2
            OS(=O)(c1ccc2c(n1)c1nc(ccc1cc2)c1scc2c1CC[C@H]1[C@]3([C@@H]2CCC3)CC1)O
            C[C@]12CC[C@](C2(C)C)(c2c1ccnc2c1ccc2c(n1)c1nc(ccc1cc2)C1=N[C@H]2[C@H](C1)CC[C@@H]1[C@H]2CCCCC1)C
            CC1(C)[C@@]2(C)CC[C@@]1(C)c1c2cnc(c1)c1ccc2c(n1)c1nc(ccc1c1c2cc[nH]1)c1cc2c(s1)CCCC2
            OS(=O)(c1ccc2c(n1)c1nc(ccc1c1c2cc[nH]1)c1nccc2-c3c(Cc12)[nH]cc3)O
            CC1(C)CCC(c2c1ccnc2c1nc2c(c3c1CCC3)ccc1c2nc(cc1)C1=NC=C2[C@@H]1C=CC=C2)(C)C
            CC1(C)CCC([C@@H]2[C@@H]1c1c(scc1-c1c2cc[nH]1)c1ccc2c(n1)c1nc(ccc1cc2)c1ccccn1)(C)C
            NC(=O)c1nc2c3nc(ccc3c3c(c2c2c1C(C)(C)CCC2(C)C)C[C@H]1[C@H]3C(C)(C)CCC1(C)C)c1ccccn1
            OP(=O)(c1ccc2c(n1)c1N=C(c3cccs3)C3=CC=N[C@@H]3c1c1c2C=C1)O
            C[C@@]1(N)[C@H]2CCC[C@@H]2C(c2c1c(sc2)c1ccc2c(n1)c1nc(ccc1cc2)P(=O)(O)O)(C)C
            [O-]C(=O)c1ccc2c(n1)c1nc(C3=N[C@H]4[C@@H](C3)CC4)c3c(c1cc2)Cc1c(C3)c[nH]c1
            NC(=O)c1ccc2c(n1)c1nc(ccc1cc2)c1ncc2c(c1)[C@@]1(C)[C@H]3[C@@H]([C@@]2(C1(C)C)C)CC[C@H]1[C@H](C3)CCCCC1
            C1C=CC(=N1)c1nc2c(c3c1Cc1c(C3)ccc3c1CCCC3)ccc1c2nc(cc1)c1cccs1
            [O-]C(=O)c1nc2c(c3c1C[C@H]1CCC[C@@H]1CC3)ccc1c2nc(c2c1CCCC2)P(=O)(O)O
            OS(=O)(c1ccc2c(n1)c1nc(ccc1c1c2CC[C@H]2[C@@H]1CCC2)C1=NC=c2c1cnc2)O
            [O-]C(=N)c1ccc2c(n1)c1nc(ccc1c1c2N=C2C1=CN=C2)S(=O)(O)O
            OS(=O)(c1ccc2c(n1)c1nc(ccc1c1c2cccc1)c1nccc2c1-c1nccc1C2)O
            OS(=O)(c1ccc2c(n1)c1nc(ccc1c1c2cc[nH]1)c1sc2c(c1)Cc1c(C2)ncc1)O
            c1ccc(nc1)c1nc2c3nc(ccc3c3c(c2c2c1cc1c2ccn1)CC3)C1=NC=CC1
            CC1(C)c2c[nH]cc2C(c2c1c1ccc3c(c1nc2c1ccc2c(n1)CC2)nc(cc3)c1cccs1)(C)C
            [O-]C(=O)c1nc2c3nc(c4ccccn4)c4=CC=Nc4c3c3=CC=Nc3c2c2c1C(C)(C)CCC2(C)C
            NC(=O)c1nc2c3nc(c4ccccn4)c4c(c3ccc2c2c1c1CCc1[nH]2)CC4
            C[C@]12c3ccccc3[C@](C2(C)C)(c2c1c1ccc(nc1c1c2ccc(n1)c1cccs1)c1ccc2c(n1)CCCCC2)C
            """)
    smiles_list = [s.strip() for s in smiles_list.splitlines()]

    mol_image = draw_grid(smiles_list, save_path="/mnt/d/zhang/OneDrive/Desktop/mol2img1.png")
