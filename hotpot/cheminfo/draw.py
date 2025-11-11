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
import os
import io
import copy
import os.path as osp
from typing import Union, Iterable, Optional, Literal
from xml.etree import ElementTree as ET
from packaging import version

import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from .core import Molecule

__all__ = [
    "compute_atom_bond_colors",
    "create_svg_colorbar",
    "merge_colorbar_to_svg",
    "draw_single_mol",
    "draw_grid"
]

# Check dependents version
if version.parse(rdkit.__version__) < version.parse('2025.9'):
    raise version.InvalidVersion(f'Expected version of rdkit should later than 2025.9, got {rdkit.__version__}')

# Annotation definitions
RGBColor = tuple[float, float, float]
IndexColorType = Union[dict[int, RGBColor], list[int], tuple[int, ...]]
DrawOutputFormat = Literal['svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp']

# Module configurations
_file_dir = osp.dirname(osp.abspath(__file__))
_font_dir = osp.join(_file_dir, "fonts")
atom_palette = {7: (0, 0, 0.5), 8: (0.5, 0, 0), 1: (0, 0.5, 0.5)}

def _find_single_maximum_subs(*smiles: str):
    """Find the maximum common substructure (MCS) among molecules."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    res = rdFMCS.FindMCS(mols)
    return Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmarts(res.smartsString)))

def choose_best_colnum(n_mols, min_col=4, max_col=7, default_col=5):
    """Select the most balanced column number for molecule grid layout."""
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
        ref_rdmol,
        font_size: int = 24,
        font: str = 'Arial',
        fontweight: str = 'bold',
        colorful_atom: bool = 'False',
        atom_color_palette: dict = None,
):
    """Configure RDKit drawing options."""
    options = rdMolDraw2D.MolDrawOptions()
    # options.useACS1996Style = True
    Draw.SetACS1996Mode(options, Draw.MeanBondLength(ref_rdmol))
    options.bondLineWidth = 2
    options.maxFontSize = font_size
    options.minFontSize = font_size
    # options.atomLabelBold = True

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

def _parse_atom_bond_colors(
        atom_colors: IndexColorType = None,
        bond_colors: IndexColorType = None,
):
    """
    Normalize atom and bond highlighting data into a consistent 4-tuple format.

    Args:
        atom_colors: Either a list/tuple of atom indices or a dict mapping atom index -> RGB tuple.
        bond_colors: Same format as atom_colors, but for bonds.

    Returns:
        (hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors)
    """
    hl_atoms = hl_bonds = hl_atom_colors = hl_bond_colors = None

    if isinstance(atom_colors, (list, tuple)):
        hl_atoms = list(atom_colors)
    elif isinstance(atom_colors, dict):
        hl_atoms = list(atom_colors.keys())
        hl_atom_colors = atom_colors

    if isinstance(bond_colors, (list, tuple)):
        hl_bonds = list(bond_colors)
    elif isinstance(bond_colors, dict):
        hl_bonds = list(bond_colors.keys())
        hl_bond_colors = bond_colors

    return hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors

def _parse_atom_bond_color_lists(
        atom_color_lists: list[IndexColorType] = None,
        bond_color_lists: list[IndexColorType] = None,
):
    """
    Batch-parse multiple sets of atom/bond highlight color definitions.

    Args:
        atom_color_lists: List of atom color definitions (dicts or lists), or None.
        bond_color_lists: List of bond color definitions (dicts or lists), or None.

    Returns:
        (hl_atom_lists, hl_bond_lists, hl_atom_colors, hl_bond_colors)
        Each element is either a list of results or None.
    """
    # Shortcut: if both are None, return all None
    if atom_color_lists is None and bond_color_lists is None:
        return None, None, None, None

    def infinite_none():
        while True:
            yield None

    # Pair up the iterables, defaulting to infinite None generator when one is missing
    atom_iter = atom_color_lists if isinstance(atom_color_lists, list) else infinite_none()
    bond_iter = bond_color_lists if isinstance(bond_color_lists, list) else infinite_none()

    # Collect batched results per category
    list_results = [[], [], [], []]

    for atom_colors, bond_colors in zip(atom_iter, bond_iter):
        parsed = _parse_atom_bond_colors(atom_colors, bond_colors)
        for collected, value in zip(list_results, parsed):
            collected.append(value)

    # Replace completely empty result groups with None
    final_results = [values if any(v is not None for v in values) else None
                     for values in list_results]

    hl_atom_lists, hl_bond_lists, hl_atom_colors, hl_bond_colors = final_results
    return hl_atom_lists, hl_bond_lists, hl_atom_colors, hl_bond_colors


def compute_atom_bond_colors(
        mol: Chem.Mol,
        atom_values: np.ndarray,
        cmap_name: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: float = 0.75
) -> tuple[
        dict[int, RGBColor],
        dict[int, RGBColor],
        mcolors.Normalize,
        cm.ScalarMappable
]:
    """
    Compute color mappings for atoms and bonds based on numeric values.

    Args:
        mol: RDKit molecule object.
        atom_values: SHAP or importance values for each atom.
        cmap_name: Matplotlib colormap name.
        vmin, vmax: Manual color scale bounds. Auto-computed if None.
        threshold: Values below abs(threshold) will not be colored.

    Returns:
        (atom_colors, bond_colors, norm, sm)

        atom_colors : dict
            Mapping from atom index to RGB color tuple.
        bond_colors : dict
            Mapping from bond index to RGB color tuple.
        norm : mcolors.Normalize
            Normalization object for the colormap.
        sm : cm.ScalarMappable
            Scalar mappable object for generating colorbars.
    """
    n_atoms = mol.GetNumAtoms()
    atom_values = np.asarray(atom_values).flatten()
    assert len(atom_values) == n_atoms, f"atom_values length {len(atom_values)} != n_atoms {n_atoms}"

    # 设置颜色范围
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(atom_values))
        if abs_max == 0 or np.isnan(abs_max):
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max

    # 创建颜色映射
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # 计算原子颜色
    atom_colors = {}
    for i, val in enumerate(atom_values):
        if not np.isnan(val) and abs(val) > threshold:
            rgba = cmap(norm(val))
            atom_colors[i] = tuple(rgba[:3])  # RGB only

    # 计算键颜色（基于两端原子的平均值）
    bond_colors = {}
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()

        # 计算键的值为两端原子的平均
        bond_val = (atom_values[begin_atom] + atom_values[end_atom]) / 2

        if not np.isnan(bond_val) and abs(bond_val) > threshold:
            rgba = cmap(norm(bond_val))
            bond_colors[bond_idx] = tuple(rgba[:3])

    return atom_colors, bond_colors, norm, sm


def _fill_highlight_atoms(
        mol: Chem.Mol,
        atom_values: np.ndarray,
        cmap_name: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: float = 0.75
):
    atom_colors, bond_colors, norm, sm = compute_atom_bond_colors(
        mol, atom_values, cmap_name=cmap_name, vmin=vmin, vmax=vmax, threshold=threshold
    )
    hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors = _parse_atom_bond_colors(atom_colors, bond_colors)
    return (hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors), norm


def _fill_highlight_atoms_list(
        mols: list["Chem.Mol"],
        list_atom_values: list[np.ndarray],
        cmap_name: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: float = 0.75
):
    """
    Batch-compute atom and bond highlight colors for multiple molecules.

    This is the draw_grid() version of `_fill_highlight_atoms()`.
    Uses a shared color normalization (vmin/vmax) across all molecules
    to ensure consistent color scaling across the grid.

    Parameters
    ----------
    mols : list of Chem.Mol
        List of RDKit molecule objects.
    list_atom_values : list of np.ndarray
        List of atom-value arrays (e.g., SHAP or importance scores),
        one per molecule. Each must match its molecule’s atom count.
    cmap_name : str, optional, default="coolwarm"
        Matplotlib colormap name used for color mapping.
    vmin, vmax : float, optional
        Global color normalization range.
        If None, the range is inferred from all `list_atom_values`
        and automatically adjusted to be symmetric about zero
        for diverging colormaps such as "coolwarm".
    threshold : float, optional, default=0.75
        Minimum relative intensity (|normalized value|) required
        for an atom/bond to be visibly colored (below this → faded).

    Returns
    -------
    tuple
        ((list_hl_atoms, list_hl_bonds, list_hl_atom_colors, list_hl_bond_colors), norm)

        Each element in the 4 lists corresponds to one molecule’s
        RDKit-compliant highlight data. `norm` is the shared Matplotlib
        normalization object used across all molecules.

    Notes
    -----
    - The function computes a *shared* Matplotlib normalization (`norm`)
      rather than deriving it separately per molecule.
    - This ensures the colors across molecules are comparable and scaled consistently.
    - The default colormap ("coolwarm") is centered around 0, so the normalization
      range (`vmin`, `vmax`) is symmetric unless explicitly overridden.
    """

    # --- Input validation ---
    if not mols or not list_atom_values:
        raise ValueError("Both `mols` and `list_atom_values` must be provided and non-empty.")
    if len(mols) != len(list_atom_values):
        raise ValueError(f"Molecule count ({len(mols)}) and atom-value arrays ({len(list_atom_values)}) must match.")

    # === Compute the global normalization range ===
    all_values = np.concatenate([
        np.asarray(v).ravel()
        for v in list_atom_values
        if v is not None and len(v) > 0 and np.isfinite(v).any()
    ])
    if all_values.size == 0:
        raise ValueError("No valid numeric data in `list_atom_values` for normalization.")

    if vmin is None or vmax is None:
        # Automatically find the global min and max
        raw_min, raw_max = np.min(all_values), np.max(all_values)
        # For diverging colormaps such as 'coolwarm', use a symmetric range around zero
        max_abs = max(abs(raw_min), abs(raw_max))
        global_vmin, global_vmax = -max_abs, max_abs
    else:
        global_vmin, global_vmax = vmin, vmax

    norm_shared = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax, clip=True)

    # === Compute per-molecule highlight data ===
    list_hl_atoms, list_hl_bonds, list_hl_atom_colors, list_hl_bond_colors = [], [], [], []

    for mol, atom_values in zip(mols, list_atom_values):
        if atom_values is None or len(atom_values) == 0:
            # When a molecule has no valid numeric input:
            # - RDKit does NOT error if highlightAtoms or highlightAtomColors is None.
            # - However, passing an empty list ([]) can trigger dimensional or type mismatches.
            # Therefore, returning None ensures that the upper-level _parse_atom_bond_color_lists
            # can correctly identify and skip such entries.
            hl_atoms = hl_bonds = hl_atom_colors = hl_bond_colors = None
        else:
            atom_colors, bond_colors, _, _ = compute_atom_bond_colors(
                mol,
                atom_values,
                cmap_name=cmap_name,
                vmin=global_vmin,
                vmax=global_vmax,
                threshold=threshold
            )
            hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors = _parse_atom_bond_colors(
                atom_colors, bond_colors
            )

        list_hl_atoms.append(hl_atoms)
        list_hl_bonds.append(hl_bonds)
        list_hl_atom_colors.append(hl_atom_colors)
        list_hl_bond_colors.append(hl_bond_colors)

    return (list_hl_atoms, list_hl_bonds, list_hl_atom_colors, list_hl_bond_colors), norm_shared


def _load_rdmol(mol_in, sanitize=False):
    """ Load Rdkit.Chem.Mol and build 2D coordinates """
    if isinstance(mol_in, Chem.Mol):
        mol = mol_in
    elif hasattr(mol_in, "smiles"):
        mol = Chem.MolFromSmiles(mol_in.smiles, sanitize=sanitize)
    else:
        mol = Chem.MolFromSMiles(mol_in, sanitize=sanitize)

    # Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {mol_in}")

    AllChem.Compute2DCoords(mol)
    return mol

def save_or_convert_svg(svg_text: str, save_path: str = None, output_format: str = None) -> str:
    """
    Unified logic for saving or converting SVG output across drawing functions.

    Parameters
    ----------
    svg_text : str
        SVG XML content to save or convert.
    save_path : str, optional
        File path to save output. If None, file will not be saved.
    output_format : str, optional
        Desired output format (e.g., 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp').
        If None, inferred automatically from save_path extension or defaults to 'svg'.

    Returns
    -------
    str
        The original SVG text string.

    Notes
    -----
    This method supports both vector and bitmap formats:
        - Vector (via CairoSVG): `svg`, `png`, `pdf`, `ps`, `eps`
        - Bitmap (via Pillow, after intermediate PNG conversion): `jpeg`, `jpg`, `bmp`, `tiff`, `webp`
    """
    if not save_path and not output_format:
        return svg_text

    # === Determine format ===
    if output_format:
        ext = output_format.lower()
    elif save_path:
        ext = osp.splitext(save_path)[-1][1:].lower() if '.' in osp.basename(save_path) else 'svg'
    else:
        ext = 'svg'

    supported_formats = ('svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp')
    if ext not in supported_formats:
        raise ValueError(f"Unsupported output format: {ext}")

    # === Save / convert ===
    if save_path:
        svg_bytes = svg_text.encode("utf-8")

        # Direct SVG save
        if ext == "svg":
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(svg_text)
            return svg_text

        cairo_converters = {
            "png": cairosvg.svg2png,
            "pdf": cairosvg.svg2pdf,
            "ps": cairosvg.svg2ps,
            "eps": cairosvg.svg2ps,
        }

        # Vector or raster (direct)
        if ext in cairo_converters:
            cairo_converters[ext](bytestring=svg_bytes,write_to=save_path)
        else:
            # Bitmap conversion via Pillow
            import io
            from PIL import Image
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
            img = Image.open(io.BytesIO(png_bytes))
            fmt_map = {
                'jpg': 'JPEG',
                'jpeg': 'JPEG',
                'bmp': 'BMP',
                'tiff': 'TIFF',
                'webp': 'WEBP',
            }
            img.save(save_path, format=fmt_map.get(ext, 'PNG'))
    return svg_text


def create_svg_colorbar(
        name: str = 'coolwarm',
        norm: mcolors.Normalize = None,
        orientation: Literal['vertical', 'horizontal'] = 'vertical',
):
    length = 3
    width = 0.15

    if orientation == 'vertical':
        fig, ax = plt.subplots(figsize=(width, length))
    elif orientation == 'horizontal':
        fig, ax = plt.subplots(figsize=(length, width))
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    if norm is None:
        norm = mcolors.Normalize(vmin=-1, vmax=1)

    _ = mpl.colorbar.Colorbar(ax, cmap=name, orientation=orientation, norm=norm)

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    colorbar_svg = buf.getvalue()
    plt.close(fig)

    return colorbar_svg

def merge_colorbar_to_svg(main_svg, colorbar_svg, fig_size, orientation):
    if orientation == 'vertical':
        xy = '(530, 200)'
    else:
        xy = '(100, 10)'

    root_main = ET.fromstring(main_svg)
    root_cb = ET.fromstring(colorbar_svg)

    # 从colorbar SVG中取出真正的内容（去掉外层figure用的<svg>）
    # matplotlib的<svg>里一般有<g>组包含图形元素
    for child in list(root_cb):
        if child.tag.endswith('g'):
            # 调整位置，把colorbar放到右侧
            child.set('transform', f'translate{xy} scale(1.0)')
            root_main.append(child)

    return ET.tostring(root_main, encoding='unicode')


def draw_single_mol(
        mol_in: Union[str, 'Molecule'],
        save_path: Optional[str] = None,
        mol_size: tuple[int, int] = (600, 600),
        output_format: Optional[DrawOutputFormat] = None,
        font_size: int = 24,
        font: str = 'Arial',
        fontweight: str = 'bold',
        colorful_atom: bool = False,
        atom_color_palette: Optional[dict] = None,
        legend: str = '',
        sanitize: bool = False,

        # Highlight control
        atom_colors=None,
        bond_colors=None,
        atom_hl_values: np.ndarray = None,
        cmap_name: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: float = 0.75,
        colorbar: bool = False,
        cb_orientation: Literal['vertical', 'horizontal'] = 'vertical'
):
    """
    Draw a single molecule, always rendering to an SVG string first,
    then optionally converting it to other formats (PNG, TIFF, PDF, etc.) using CairoSVG and Pillow.

    Supported formats:
        - Direct CairoSVG: svg, png, pdf, ps, eps
        - Via Pillow: jpeg, jpg, tiff, bmp, webp
    """

    # === 0. Load molecule ===
    if isinstance(mol_in, Chem.Mol):
        mol = mol_in
    elif hasattr(mol_in, "smiles"):
        mol = Chem.MolFromSmiles(mol_in.smiles, sanitize=sanitize)
    else:
        mol = Chem.MolFromSmiles(mol_in, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"Invalid molecule input: {mol_in}")
    AllChem.Compute2DCoords(mol)

    # === 1. Drawing options ===
    options = _draw_configuration(mol, font_size, font, fontweight, colorful_atom, atom_color_palette)

    # === 2. Highlight colors ===
    if atom_colors or bond_colors:
        hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors = _parse_atom_bond_colors(atom_colors, bond_colors)
        norm = None
    elif atom_hl_values is not None:
        (hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors), norm = _fill_highlight_atoms(
            mol, atom_hl_values, cmap_name, vmin, vmax, threshold=threshold
        )
    else:
        hl_atoms = hl_bonds = hl_atom_colors = hl_bond_colors = None
        norm = None

    # === 3. Always draw to SVG ===
    drawer = rdMolDraw2D.MolDraw2DSVG(mol_size[0], mol_size[1])
    drawer.SetDrawOptions(options)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        legend=legend,
        highlightAtoms=hl_atoms,
        highlightBonds=hl_bonds,
        highlightAtomColors=hl_atom_colors,
        highlightBondColors=hl_bond_colors,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    if colorbar:
        colorbar_svg = create_svg_colorbar(cmap_name, norm, orientation=cb_orientation)
        svg_text = merge_colorbar_to_svg(svg_text, colorbar_svg, mol_size, cb_orientation)

    # Formatted output and save
    return save_or_convert_svg(svg_text, save_path, output_format)


def draw_grid(
        list_mols: Iterable[Union[str, 'Molecule']],
        save_path: Optional[str] = None,
        mol_size: tuple[int, int] = (300, 300),
        output_format: Optional[DrawOutputFormat] = None,
        font_size: int = 20,
        font: str = 'Arial',
        fontweight: str = 'bold',
        colorful_atom: bool = True,
        atom_color_palette: Optional[dict] = None,
        sanitize: bool = False,
        n_cols: Optional[int] = None,
        legends: Optional[list[str]] = None,

        # Highlight control
        atom_colors: Optional[list[dict]] = None,
        bond_colors: Optional[list[dict]] = None,
        list_atom_values: np.ndarray = None,
        cmap_name: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        threshold: float = 0.75,
        colorbar: bool = False,
        cb_orientation: Literal['vertical', 'horizontal'] = 'horizontal'
):
    """
    Draw a grid of molecules into a single combined image.

    The function always renders the molecular grid to an SVG string first,
    then optionally converts it to other formats (PNG, PDF, EPS, etc.)
    using CairoSVG or Pillow.

    Parameters
    ----------
    list_mols : Iterable[str or Molecule]
        Iterable of SMILES strings or Molecule objects to render.
    save_path : str, optional
        Path to save output image. If not provided, only the SVG string is returned.
    mol_size : tuple of (int, int), optional, default=(300, 300)
        Pixel dimensions of each molecule image in the grid.
    output_format : {'svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp'}, optional
        Desired output format. If None, inferred from ``save_path`` extension or defaults to 'svg'.
    font_size : int, optional, default=20
        Font size for atom labels.
    font : str, optional, default='Arial'
        Font family for text labels.
    fontweight : str, optional, default='bold'
        Font weight for atom labels ('normal' or 'bold').
    colorful_atom : bool, optional, default=True
        Whether to color atom labels by element or palette.
    atom_color_palette : dict, optional
        Mapping of element symbols to RGB color tuples.
    sanitize : bool, optional, default=False
        Whether to sanitize SMILES parsing.
    n_cols : int, optional
        Number of columns in the molecular grid. Automatically chosen if None.
    legends : list of str, optional
        Text legends displayed below each molecule. Must match the number of molecules.
    atom_colors : list of dict or list, optional
        List of color mappings or atom index lists for highlighting atoms.
    bond_colors : list of dict or list, optional
        List of color mappings or bond index lists for highlighting bonds.
        list_atom_values : list[np.ndarray], optional
        Per‑molecule numeric values (e.g., atomic contributions or SHAP scores)
        used for colormapped highlighting. If provided, overrides `atom_colors`
        and `bond_colors`.
    cmap_name : str, default='coolwarm'
        Matplotlib colormap name for continuous value highlighting.
    vmin, vmax : float, optional
        Global normalization limits for the colormap. If None, inferred from all
        values across molecules and symmetrized around zero when using a diverging
        palette such as "coolwarm".
    threshold : float, default=0.75
        Minimum normalized intensity (|value|) for visible highlighting.
    colorbar : bool, default=False
        Whether to append a colorbar to the combined SVG image.
    cb_orientation : {'vertical', 'horizontal'}, default='horizontal'
        Orientation of the colorbar when `colorbar=True`.

    Returns
    -------
    svg_image : str
        SVG XML string containing the full molecule grid image.

    Notes
    -----
    The function supports both vector and bitmap outputs:
        - Vector: ``svg``, ``pdf``, ``ps``, ``eps``
        - Bitmap: ``png``, ``jpeg``, ``jpg``, ``bmp``, ``tiff``, ``webp``

    Examples
    --------
    >>> mols = ["CCO", "C1CCCCC1"]
    >>> svg = draw_grid(mols)
    >>> svg[:200]
    '<svg xmlns="http://www.w3.org/2000/svg"...'

    >>> draw_grid(mols, save_path='grid.pdf')
    >>> draw_grid(mols, output_format='webp', save_path='grid.webp')
    """
    # === 0. Create RDKit molecules ===
    mols = [_load_rdmol(m, sanitize=sanitize) for m in list_mols]
    if not mols:
        raise ValueError("No valid molecules provided for grid drawing.")

    # === 1. Drawing options ===
    options = _draw_configuration(mols[0], font_size, font, fontweight, colorful_atom, atom_color_palette)

    if atom_colors or bond_colors:
        hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors = _parse_atom_bond_color_lists(atom_colors, bond_colors)
        norm = None
    elif isinstance(list_atom_values, list):
        (hl_atoms, hl_bonds, hl_atom_colors, hl_bond_colors), norm = _fill_highlight_atoms_list(
            mols, list_atom_values, cmap_name, vmin, vmax, threshold
        )
    else:
        hl_atoms = hl_bonds = hl_atom_colors = hl_bond_colors = None
        norm = None

    # === 2. Infer grid shape ===
    if not isinstance(n_cols, int):
        n_cols = choose_best_colnum(len(mols))

    if legends is not None:
        assert isinstance(legends, (list, tuple)), "legends must be list/tuple of strings"
        assert len(legends) == len(mols), (
            f"Expected {len(mols)} legends, got {len(legends)}"
        )
        legends = list(legends)

    # === 3. Always draw SVG grid first ===
    svg_text = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=mol_size,
        legends=legends,
        useSVG=True,
        returnPNG=False,
        highlightAtomLists=hl_atoms,
        highlightBondLists=hl_bonds,
        highlightAtomColors=hl_atom_colors,
        highlightBondColors=hl_bond_colors,
        drawOptions=options
    )

    if colorbar:
        colorbar_svg = create_svg_colorbar(cmap_name, norm, orientation=cb_orientation)
        svg_text = merge_colorbar_to_svg(svg_text, colorbar_svg, mol_size, cb_orientation)

    # Formatted output and save
    return save_or_convert_svg(svg_text, save_path, output_format)


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
