"""Carrier input sequencing for molecular drawing operations."""
from typing import Union, Iterable, Optional, Literal

import numpy as np

from ..adapters.visualization import draw as _draw

__all__ = ["draw_single_mol", "draw_grid"]

DrawOutputFormat = Literal['svg', 'png', 'pdf', 'ps', 'eps', 'jpeg', 'jpg', 'tiff', 'bmp', 'webp']


def _load_rdmol(mol_in, sanitize=False):
    """ Load Rdkit.Chem.Mol and build 2D coordinates """
    if isinstance(mol_in, _draw._rdmol_type()):
        mol = mol_in
    elif hasattr(mol_in, "smiles"):
        mol = _draw._parse_smiles(mol_in.smiles, sanitize=sanitize)
    else:
        mol = _draw._parse_smiles(mol_in, sanitize=sanitize)

    # Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {mol_in}")

    _draw._compute_2d_coords(mol)
    return mol


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
    if isinstance(mol_in, _draw._rdmol_type()):
        mol = mol_in
    elif hasattr(mol_in, "smiles"):
        mol = _draw._parse_smiles(mol_in.smiles, sanitize=sanitize)
    else:
        mol = _draw._parse_smiles(mol_in, sanitize=sanitize)
    if mol is None:
        raise ValueError(f"Invalid molecule input: {mol_in}")
    _draw._compute_2d_coords(mol)

    return _draw._draw_single_rdmol(
        mol,
        save_path=save_path,
        mol_size=mol_size,
        output_format=output_format,
        font_size=font_size,
        font=font,
        fontweight=fontweight,
        colorful_atom=colorful_atom,
        atom_color_palette=atom_color_palette,
        legend=legend,
        atom_colors=atom_colors,
        bond_colors=bond_colors,
        atom_hl_values=atom_hl_values,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        colorbar=colorbar,
        cb_orientation=cb_orientation,
    )


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

    return _draw._draw_grid_rdmols(
        mols,
        save_path=save_path,
        mol_size=mol_size,
        output_format=output_format,
        font_size=font_size,
        font=font,
        fontweight=fontweight,
        colorful_atom=colorful_atom,
        atom_color_palette=atom_color_palette,
        n_cols=n_cols,
        legends=legends,
        atom_colors=atom_colors,
        bond_colors=bond_colors,
        list_atom_values=list_atom_values,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        colorbar=colorbar,
        cb_orientation=cb_orientation,
    )
