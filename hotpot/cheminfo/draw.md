

# 🧪 `hotpot.cheminfo.draw`  Module
### Molecular Visualization Utilities

This package provides high‑level molecular drawing interfaces built upon **RDKit**, **Matplotlib**, and **CairoSVG**, enabling chemical visualization with customized highlighting, color normalization, and unified export formats.

| Function                       | Description                                                                                                                                                                                     |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`compute_atom_bond_colors`** | Compute atom and bond color mappings from per‑atom numeric values (e.g., SHAP scores or feature importances). Produces RDKit‑compatible color dictionaries and a shared colormap normalization. |
| **`create_svg_colorbar`**      | Create a Matplotlib‑based colorbar and return it as an SVG string. Useful for attaching a legend or scale beside rendered molecules.                                                            |
| **`merge_colorbar_to_svg`**    | Merge a colorbar SVG into a molecule or grid SVG layout, positioning the colorbar neatly relative to the figure canvas.                                                                         |
| **`draw_single_mol`**          | Render a single molecule to SVG with optional highlighting or value‑dependent coloring, supporting multiple output formats (SVG, PNG, PDF…).                                                    |
| **`draw_grid`**                | Render multiple molecules in a unified grid layout. Supports colormap highlights with shared normalization across all molecules and optional colorbar generation.                               |


---

## 🌍 Overview

The module provides both single‑molecule and grid‑based drawing functions, automatic colormap normalization, and smooth integration with standard chemical data.  
All exporters produce scalable vector graphics (SVG) by default and can convert to **PNG**, **PDF**, **EPS**, **JPEG**, **TIFF**, or **WebP** through CairoSVG and Pillow.

### 📘 Typical Use‑Case Flow
1. Compute color values → compute_atom_bond_colors()
2. Render molecule(s) → draw_single_mol() or draw_grid()
3. Add colorbar if needed → create_svg_colorbar()
4. Merge final visuals → merge_colorbar_to_svg()
5. Together, these enable a complete pipeline from per‑atom scoring to polished, publication‑ready visualizations.

---

## 🧩 API Reference

### 1️⃣ `compute_atom_bond_colors()`

Compute atom and bond colors based on continuous numeric inputs (e.g., atomic contributions, SHAP values).

```python
from hotpot.cheminfo.draw import compute_atom_bond_colors
```

#### Signature

```python
compute_atom_bond_colors(
    mol: Chem.Mol,
    atom_values: np.ndarray,
    cmap_name: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    threshold: float = 0.75
) -> tuple[dict[int, tuple[float, float, float]],
           dict[int, tuple[float, float, float]],
           matplotlib.colors.Normalize,
           matplotlib.cm.ScalarMappable]
```

#### Description
- Generates RDKit‑compatible color maps for atomic and bond highlighting.  
- The values are normalized over a symmetric range (`[-absmax, +absmax]` by default) to match diverging colormaps.

#### Parameters

| Name           | Type                        | Description                                                                  |
|----------------|-----------------------------|------------------------------------------------------------------------------|
| `mol`          | `Chem.Mol`                  | RDKit molecule object.                                                       |
| `atom_values`  | `np.ndarray`                | Numeric value for each atom.                                                 |
| `cmap_name`    | `str`, default `"coolwarm"` | Matplotlib colormap name.                                                    |
| `vmin`, `vmax` | `float`, optional           | Fixed normalization range; defaults inferred if omitted.                     |
| `threshold`    | `float`, default `0.75`     | Minimal normalized absolute value to include color (acts like alpha cutoff). |

#### Returns
A 4‑tuple:
1. `atom_colors`: dict `{atom_idx: (r, g, b)}`
2. `bond_colors`: dict `{bond_idx: (r, g, b)}`
3. `norm`: `matplotlib.colors.Normalize`  
4. `sm`: `matplotlib.cm.ScalarMappable` (for use in colorbars)

---

### 2️⃣ `create_svg_colorbar()`

Create a colorbar figure as an SVG string using Matplotlib.

```python
from hotpot.cheminfo.draw import create_svg_colorbar
```

#### Signature

```python
create_svg_colorbar(
    name: str = "coolwarm",
    norm: matplotlib.colors.Normalize = None,
    orientation: Literal["vertical", "horizontal"] = "vertical"
) -> str
```

#### Description
Generates a Matplotlib colorbar and serializes it directly to SVG format to merge with molecule drawings.

#### Parameters

| Name          | Type                           | Description                                                     |
|---------------|--------------------------------|-----------------------------------------------------------------|
| `name`        | str                            | Matplotlib colormap name.                                       |
| `norm`        | `mcolors.Normalize`, optional  | Colormap normalization; automatically set to `[-1, 1]` if None. |
| `orientation` | `'vertical'` or `'horizontal'` | Layout orientation of the colorbar figure.                      |

#### Returns
`str`: SVG XML for the colorbar plot.

#### Example
```python
from hotpot.cheminfo.draw import create_svg_colorbar
import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=-1, vmax=1)
cb_svg = create_svg_colorbar("coolwarm", norm, orientation="horizontal")
```

---

### 3️⃣ `merge_colorbar_to_svg()`

Merge a generated colorbar SVG into an existing molecule SVG layout.

```python
from hotpot.cheminfo.draw import merge_colorbar_to_svg
```

#### Signature

```python
merge_colorbar_to_svg(
    main_svg: str,
    colorbar_svg: str,
    fig_size: tuple[int, int],
    orientation: Literal["vertical", "horizontal"]
) -> str
```

#### Description
Combines two SVG strings, repositioning the colorbar `<g>` group onto the molecule panel.

#### Parameters

| Name | Type | Description |
|------|------|-------------|
| `main_svg` | `str` | Main molecule SVG content. |
| `colorbar_svg` | `str` | Colorbar SVG snippet from `create_svg_colorbar()`. |
| `fig_size` | `tuple[int, int]` | Width × height of the chem figure (for offseting). |
| `orientation` | `'vertical'` or `'horizontal'` | How to position the colorbar in the merged SVG. |

#### Returns
`str`: The merged SVG string containing both molecule grid and colorbar.  

#### Example

```python
svg_main = draw_single_mol("CCO")
cb_svg = create_svg_colorbar()
merged = merge_colorbar_to_svg(svg_main, cb_svg, (400, 400), "vertical")
```

---

### 4️⃣ `draw_single_mol()`

Render one molecule to SVG, optionally with per‑atom highlights and colorbars.

```python
from hotpot.cheminfo.draw import draw_single_mol
```

#### Signature
```python
draw_single_mol(
    mol_in,
    save_path=None,
    mol_size=(600, 600),
    output_format=None,
    font_size=24,
    font='Arial',
    fontweight='bold',
    colorful_atom=False,
    atom_color_palette=None,
    legend='',
    sanitize=False,
    atom_colors=None,
    bond_colors=None,
    atom_hl_values=None,
    cmap_name='coolwarm',
    vmin=None,
    vmax=None,
    threshold=0.75,
    colorbar=False,
    cb_orientation='vertical'
) -> str
```

#### Description
Creates an SVG representation for a single molecule, highlighting atoms and bonds using either discrete colors or continuous colormap values.

#### Parameters (key)
- `mol_in` — input molecule (SMILES string or `Chem.Mol`)
- `atom_hl_values` — array of atom‑wise numeric values to derive colormap
- `colorbar` — flag to append colorbar (merged automatically)
- `save_path` / `output_format` — optional output export control

#### Returns
SVG XML text for the rendered molecule image.  

#### Example
```python
svg = draw_single_mol("C=C(C)C", atom_hl_values=[0.2, 0.7, -0.5, 0.9], colorbar=True)
```

### 📘 Total Parameters List

| Name                     | Type                           | Default      | Description                                                                                                      |
|--------------------------|--------------------------------|--------------|------------------------------------------------------------------------------------------------------------------|
| **`mol_in`**             | `str` or `Chem.Mol`            | —            | Input molecule, can be a SMILES string or an RDKit `Chem.Mol` object.                                            |
| **`save_path`**          | `str`, optional                | `None`       | File path to save output. If omitted, only an SVG string is returned.                                            |
| **`mol_size`**           | `tuple[int, int]`              | `(600, 600)` | Width and height (in pixels) of the drawing canvas.                                                              |
| **`output_format`**      | `str`, optional                | `None`       | Output format — `'svg'`, `'png'`, `'pdf'`, `'eps'`, `'jpg'`, etc. If `None`, inferred from `save_path`.          |
| **`font_size`**          | `int`                          | `24`         | Font size for atom labels.                                                                                       |
| **`font`**               | `str`                          | `'Arial'`    | Font family used in drawing, must exist in the module’s `fonts` directory or system fonts.                       |
| **`fontweight`**         | `str`                          | `'bold'`     | Font weight for label rendering (`'normal'` or `'bold'`).                                                        |
| **`colorful_atom`**      | `bool`                         | `False`      | Whether atom labels are automatically colored by element type using RDKit’s default palette.                     |
| **`atom_color_palette`** | `dict`                         | `None`       | Custom atom palette: `{atomic_number: (r,g,b)}` in normalized RGB (0–1).                                         |
| **`legend`**             | `str`                          | `''`         | Optional caption or ID shown below the molecule.                                                                 |
| **`sanitize`**           | `bool`                         | `False`      | Enable SMILES sanitization during parsing (recommended True for complex SMILES).                                 |
| **`atom_colors`**        | `dict` or `list`, optional     | `None`       | Explicit per‑atom highlight color map, overrides `atom_hl_values`. Example: `{0: (1,0,0), 1: (0,0,1)}`.          |
| **`bond_colors`**        | `dict` or `list`, optional     | `None`       | Explicit per‑bond highlight color map, same format as `atom_colors`.                                             |
| **`atom_hl_values`**     | `np.ndarray`, optional         | `None`       | Per‑atom numeric values (e.g., SHAP or contribution values) mapped to colors using `cmap_name`.                  |
| **`cmap_name`**          | `str`                          | `'coolwarm'` | Matplotlib colormap used for continuous color mapping.                                                           |
| **`vmin, vmax`**         | `float`, optional              | `None`       | Colormap normalization limits. If not specified, inferred automatically from data and made symmetric (±max abs). |
| **`threshold`**          | `float`                        | `0.75`       | Relative normalized cutoff; smaller magnitudes are faded/uncolored.                                              |
| **`colorbar`**           | `bool`                         | `False`      | Whether to append a colorbar SVG automatically.                                                                  |
| **`cb_orientation`**     | `'vertical'` or `'horizontal'` | `'vertical'` | Orientation of the colorbar if appended.                                                                         |

---

---

### 5️⃣ `draw_grid()`

Render multiple molecules in a grid layout, ensuring consistent normalization and optional colorbar.

```python
from hotpot.cheminfo.draw import draw_grid
```

#### Signature

```python
draw_grid(
    list_mols,
    save_path=None,
    mol_size=(300, 300),
    output_format=None,
    font_size=20,
    font='Arial',
    fontweight='bold',
    colorful_atom=True,
    atom_color_palette=None,
    sanitize=False,
    n_cols=None,
    legends=None,
    atom_colors=None,
    bond_colors=None,
    list_atom_values=None,
    cmap_name='coolwarm',
    vmin=None,
    vmax=None,
    threshold=0.75,
    colorbar=False,
    cb_orientation='horizontal'
) -> str
```

#### Description
- Produces a multi‑molecule figure arranged into an adaptive grid.  
- Supports highlight coloring across molecules via atom‑wise numeric values.  
- When `list_atom_values` is provided, normalization is shared across the entire grid.

#### Parameters (key)

| Name               | Type             | Description                                            |
|--------------------|------------------|--------------------------------------------------------|
| `list_mols`        | iterable         | SMILES strings or RDKit molecules.                     |
| `list_atom_values` | list[np.ndarray] | Atom‑level numeric arrays per molecule.                |
| `n_cols`           | int, optional    | Number of columns in the grid (auto‑balanced if None). |
| `colorbar`         | bool             | Whether to append a shared colorbar.                   |

#### Returns
`str`: SVG text for the grid layout.  
Saved as file if `save_path` provided.

#### Example
```python
mols = ["CCO", "C1=CC=CC=C1", "CC(=O)O", "CCN(CC)CC"]
values = [np.random.randn(Chem.MolFromSmiles(m).GetNumAtoms()) for m in mols]

svg = draw_grid(
    mols,
    list_atom_values=values,
    colorbar=True,
    save_path="shap_grid.pdf"
)
```


#### 📘 Total Parameters List

| Name                     | Type                           | Default        | Description                                                                                                  |
|--------------------------|--------------------------------|----------------|--------------------------------------------------------------------------------------------------------------|
| **`list_mols`**          | Iterable[str or Molecule]      | —              | List of SMILES strings or `Chem.Mol` / `Molecule` objects.                                                   |
| **`save_path`**          | str, optional                  | `None`         | File path to save output image; returns only SVG string if omitted.                                          |
| **`mol_size`**           | tuple[int, int]                | `(300, 300)`   | Pixel dimensions for each molecule panel.                                                                    |
| **`output_format`**      | str, optional                  | `None`         | Output format to export (`'svg'`, `'png'`, `'pdf'`, etc.).                                                   |
| **`font_size`**          | int                            | `20`           | Font size used in atom and legend text.                                                                      |
| **`font`**               | str                            | `'Arial'`      | Font family name.                                                                                            |
| **`fontweight`**         | str                            | `'bold'`       | Font weight property.                                                                                        |
| **`colorful_atom`**      | bool                           | `True`         | Whether to color atoms (e.g., oxygen red, nitrogen blue) instead of monochrome.                              |
| **`atom_color_palette`** | dict                           | `None`         | Override of default atomic color map.                                                                        |
| **`sanitize`**           | bool                           | `False`        | Sanitize molecules before 2D layout computation.                                                             |
| **`n_cols`**             | int, optional                  | `None`         | Number of columns in grid. Determined automatically if unspecified (balanced layout).                        |
| **`legends`**            | list[str]                      | `None`         | Text labels for each molecule image. Must match molecule count.                                              |
| **`atom_colors`**        | list[dict or list], optional   | `None`         | Per‑molecule highlight definitions for specific atoms.                                                       |
| **`bond_colors`**        | list[dict or list], optional   | `None`         | Per‑molecule highlight definitions for specific bonds.                                                       |
| **`list_atom_values`**   | list[np.ndarray], optional     | `None`         | Each array contains per‑atom numeric values for corresponding molecule. Enables shared value‑based coloring. |
| **`cmap_name`**          | str                            | `'coolwarm'`   | Matplotlib colormap to apply to atom values.                                                                 |
| **`vmin, vmax`**         | float, optional                | `None`         | Normalization limits across all molecules. If `None`, inferred globally.                                     |
| **`threshold`**          | float                          | `0.75`         | Minimum normalized absolute intensity for color visibility.                                                  |
| **`colorbar`**           | bool                           | `False`        | Append a shared colorbar SVG to the combined image if True.                                                  |
| **`cb_orientation`**     | `'vertical'` or `'horizontal'` | `'horizontal'` | Orientation for colorbar placement.                                                                          |

---

## 🎨 Output Formats

| Type       | Extensions                                        | Converter         |
|------------|---------------------------------------------------|-------------------|
| **Vector** | `.svg`, `.pdf`, `.ps`, `.eps`                     | CairoSVG          |
| **Bitmap** | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | CairoSVG + Pillow |

Both `draw_single_mol()` and `draw_grid()` internally generate an **SVG** first and then use `CairoSVG` for vector/pdf export or Pillow for bitmap formats.

---

## 💡 Tips

- Ensure per‑atom value arrays match the molecule’s atom count.
- Use **diverging colormaps** (`coolwarm`, `RdBu`, `bwr`) for signed features.
- Legends can be added in grids via the `legends` parameter.
- Returned SVG strings can be displayed interactively in Jupyter via:
```pycon
from IPython.display import SVG
SVG(draw_grid(["CCO", "CCN"]))
```

---

## 🔖 Authors & Credits
Developed as part of **Hotpot Cheminfo**, integrating:  
- **RDKit** — molecular representation and drawing backend  
- **Matplotlib** — colormap scaling and colorbar rendering  
- **CairoSVG / Pillow** — export and rasterization utilities  

