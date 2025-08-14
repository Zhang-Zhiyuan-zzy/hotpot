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
import copy
import os
import os.path as osp
from textwrap import dedent
from typing import Union, Iterable, Optional

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

# Module configurations
_file_dir = osp.dirname(osp.abspath(__file__))
_font_dir = osp.join(_file_dir, 'fonts')
atom_palette = {7: (0, 0, 0.5), 8: (0.5, 0, 0)}

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

def _load_font_names():
    return os.listdir(_font_dir)


def _draw_configuration(
        font_size: int = 24,
        font: str = 'Arial',
        fontweight: str = 'bold',
        colorful_atom: bool = 'False',
        atom_color_palette: dict = None,
):
    """ Draw Options """
    options = rdMolDraw2D.MolDrawOptions()
    options.useACS1996Style = True
    options.bondLineWidth = 2
    options.maxFontSize = font_size
    options.minFontSize = font_size
    options.atomLabelBold = True

    font_names = _load_font_names()
    if isinstance(font, str) and font in font_names:
        font_suffix = f'_{fontweight.capitalize()}' if isinstance(fontweight, str) else ''
        options.fontFile = osp.join(_file_dir, 'fonts', font, f'{font}{font_suffix}.ttf')

    if not colorful_atom:
        options.useBWAtomPalette()
    else:
        _atom_palette = copy.copy(atom_palette)
        _atom_palette.update(atom_color_palette if isinstance(atom_color_palette, dict) else {})
        options.updateAtomPalette(_atom_palette)

    return options

def draw_grid(
        list_mols: Iterable[Union[str, "Molecule"]],
        save_path=None,
        mol_size=(300, 300),
        save_svg: Optional[bool] = None,
        font_size: int = 20,
        font: str = 'Arial',
        fontweight: str = 'bold',
        colorful_atom: bool = True,
        atom_color_palette: dict = None,
):
    """
    :param list_mols:
    :param save_path: Path to save image (optional)
    :param mol_size: (width, height) of image
    :param save_svg: Whether to save as SVG (otherwise PNG)
    :param font_size: Font size for atom labels
    :param font: Font name
    :param fontweight: Font weight for atom labels
    :param colorful_atom: Whether to color the atom labels
    :param atom_color_palette: Optional dict with atom color palette
    :return: SVG string or PIL image (depending on save_svg)
    """
    from hotpot.cheminfo.core import Molecule

    # Configure arguments
    if (
            save_svg is not None and
            save_path is not None and
            osp.splitext(save_path)[-1] == '.svg'
    ):
        save_svg = True
    else:
        save_svg = False

    list_mols = [m.smiles if isinstance(m, Molecule) else m for m in list_mols]

    # 0. Create Molecules
    mols = [Chem.MolFromSmiles(sm) for sm in list_mols]
    mols = [m for m in mols if m is not None]
    for m in mols:
        tmp = AllChem.Compute2DCoords(m)

    # Drawing options
    options = _draw_configuration(font_size, font, fontweight, colorful_atom, atom_color_palette)

    # 3. Calculate the n_cols
    n_cols = choose_best_colnum(len(mols))

    # 4. Generate 2d Image
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=None,
        useSVG=save_svg,
        returnPNG=False,
        drawOptions=options
    )

    # 4. Save image
    if save_path:
        if save_svg:
            with open(save_path, 'w') as writer:
                writer.write(img)
        else:
            img.save(save_path)
            print(f"Molecular 2D images have saved in: {save_path}")

    return img


def draw_single_mol(
        mol_in,
        save_path=None,
        mol_size=(600, 600),
        save_svg: bool = None,
        font_size: int = 24,
        font: str = 'Arial',
        fontweight: str = 'bold',
        colorful_atom: bool = False,
        atom_color_palette: Optional[dict] = None,
):
    """
    Draw a single molecule to an image (SVG or PNG) with adjustable font size.

    :param mol_in: SMILES string or Molecule object (with .smiles property)
    :param save_path: Path to save image (optional)
    :param mol_size: (width, height) of image
    :param save_svg: Whether to save as SVG (otherwise PNG)
    :param font_size: Font size for atom labels
    :param font: Font name
    :param fontweight: Font weight for atom labels
    :param colorful_atom: Whether to color the atom labels
    :param atom_color_palette: Optional dict with atom color palette
    :return: SVG string or PIL image (depending on save_svg)
    """
    # Guess SVG if needed
    if (
        save_svg is None
        and save_path is not None
        and osp.splitext(str(save_path))[-1] == ".svg"
    ):
        save_svg = True
    elif save_svg is None:
        save_svg = False

    # Flexible SMILES or Molecule object
    if hasattr(mol_in, "smiles"):
        sm = mol_in.smiles
    else:
        sm = mol_in

    mol = Chem.MolFromSmiles(sm, sanitize=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {sm}")

    AllChem.Compute2DCoords(mol)

    # Drawing options
    options = _draw_configuration(font_size, font, fontweight, colorful_atom, atom_color_palette)

    # Create and draw
    if save_svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(mol_size[0], mol_size[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])

    drawer.SetDrawOptions(options)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    if save_path:
        if save_svg:
            with open(save_path, 'w') as writer:
                writer.write(drawer.GetDrawingText())
        else:
            drawer.WriteDrawingText(save_path)

    return drawer.GetDrawingText()


if __name__ == "__main__":
    import os
    import glob
    from hotpot import read_mol

    mol_dir = '/mnt/d/zhang/OneDrive/Liu/nuclear medical patent/Molecule/mol_files'
    list_smiles = []
    for i, file_path in enumerate(glob.glob(os.path.join(mol_dir, '*.mol'))):
        file_name = osp.splitext(osp.split(file_path)[-1])[0]
        mol = read_mol(file_path)
        try:
            draw_single_mol(mol, osp.join(mol_dir, '..', 'svg', file_name + '.svg'))
        except ValueError:
            print(file_name)
            list_smiles.append(mol.smiles)
