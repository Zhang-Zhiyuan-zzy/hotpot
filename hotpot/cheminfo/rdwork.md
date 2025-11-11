# 🧬 `hotpot.cheminfo.rdwork` — aux module for rdkit module

This module provides tools to **generate**, **merge**, **analyze**, and **interpret** molecular fingerprints in RDKit — including mapping fingerprint-level feature attributions (such as SHAP values) back to individual atoms.

---

## ⚙️ Overview

The fingerprint utilities are designed to streamline feature extraction pipelines involving **Morgan**, **RDKit**, **AtomPair**, or **Torsion** fingerprints.  
They also support **SHAP-to-atom attribution mapping**, allowing visualization of fingerprint-based model explanations at the atomic level.

---

## 📘 Public API Summary

| Function | Description |
|-----------|-------------|
| `load_fp` | Generate fingerprints for one or more RDKit molecules, returning both vector and bit mapping info. |
| `merge_fps` | Combine fingerprints into a union fingerprint via bitwise OR. |
| `fps_to_array` | Convert multiple RDKit fingerprints into dense numpy arrays for ML models. |
| `assign_fp_values_atoms` | Distribute fingerprint-level scores (e.g., SHAP values) to atoms for visualization or interpretation. |

---

## 1️⃣ `load_fp()`

Generate fingerprints for a list of RDKit molecules.

```python
load_fp(
    list_mols,
    generator: Literal['Morgen', 'RdKit', 'AtomPair', 'Torison'] = 'Morgen',
    gen_kw: dict = None,
    sparse: bool = True,
    counts: bool = True,
) -> tuple[list[Chem.Mol], list, list]
```

### Description
Creates molecular fingerprints with flexible generator, counting, and sparsity options.  
For **Morgan** and **RDKit** fingerprints, the function also collects bit–atom mappings (`bitInfo`) for downstream atom‑level interpretation.

### Parameters

| Name | Type | Default | Description |
|------|------|----------|-------------|
| `list_mols` | list[Chem.Mol] | — | Input RDKit molecule objects. |
| `generator` | Literal | `"Morgen"` | Fingerprint type, one of `'Morgen'`, `'RdKit'`, `'AtomPair'`, `'Torison'`. |
| `gen_kw` | dict | `{}` | Extra keyword arguments for RDKit fingerprint generator (`radius`, `fpSize`, etc.). |
| `sparse` | bool | `True` | Return sparse (dict-like) fingerprint representation when `True`. |
| `counts` | bool | `True` | Include bit counts rather than binary bits when possible. |

### Returns
`tuple (mols, fps, bit_maps)` where:
- `fps`: list of fingerprint objects (`ExplicitBitVect` or `ULongSparseIntVect`)
- `bit_maps`: For `'Morgen'`/`'RdKit'`, bit–atom mapping dictionaries. Otherwise empty list.

### Example

```python
from rdkit import Chem
from hotpot.cheminfo.fingerprint import load_fp

mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN"]]
mols, fps, bit_info = load_fp(mols, generator='Morgen', gen_kw={'radius':2})
```

---

## 2️⃣ `merge_fps()`

Combine multiple fingerprints using logical OR.

```python
merge_fps(list_fp: list[cDS.ULongSparseIntVect]) -> cDS.ULongSparseIntVect
```

### Description
Merges all molecular fingerprints into a single `SparseIntVect` by element-wise OR — effectively capturing all activated bits across molecules.

### Example
```python
fp_union = merge_fps(fps)
```

---

## 3️⃣ `fps_to_array()`

Convert RDKit fingerprints to a NumPy array representation for ML pipelines.

```python
fps_to_array(fps: list, to_bit: bool = False) -> tuple[np.ndarray, list[int]]
```

### Description
Aligns all fingerprints by their active bit indices, forming a 2D array `[N_mols × N_bits]`.  
Optionally binarizes counts to `{0,1}` integers for model input.

### Parameters

| Name | Type | Default | Description |
|------|------|----------|-------------|
| `fps` | list | — | List of RDKit fingerprint objects. |
| `to_bit` | bool | `False` | Convert counts to binary bits (0/1). |

### Returns
- `arr`: NumPy array `[N_mols, N_bits]`
- `mfps`: list of aligned bit indices (features)

### Example

```python
arr, bit_ids = fps_to_array(fps)
print(arr.shape)  # e.g. (10, 2048)
```

---

## 4️⃣ `assign_fp_values_atoms()`

Distribute fingerprint-level scores (e.g., SHAP values) back to atoms of a molecule.

```python
assign_fp_values_atoms(
    mol: Chem.Mol,
    scores_values: np.ndarray,      # shape [E,]
    list_fp: list[int],             # fingerprint hashes
    fp_map: dict[int, list[tuple[int, int]]],
    mode: str = "sum",
    scale_to_unit: bool = True
) -> np.ndarray
```

### Description
Maps per-fingerprint importance values (e.g. SHAP outputs) to per-atom scores based on bit–atom environment relations extracted from RDKit fingerprint generators.

### Parameters

| Name | Type | Default | Description |
|------|------|----------|-------------|
| `mol` | `Chem.Mol` | — | Molecule used to calculate atom environments. |
| `scores_values` | `np.ndarray` | — | Fingerprint-level values (one per bit). Must be 1‑D. |
| `list_fp` | list[int] | — | Fingerprint hash values aligned with `scores_values`. |
| `fp_map` | dict | — | Mapping `{hash: [(center_atom, radius), ...]}` from RDKit bitInfo. |
| `mode` | str | `"sum"` | Aggregation per‑atom: `'sum'` to add all, `'mean'` to average overlapping contributions. |
| `scale_to_unit` | bool | `True` | Re‑scale output to range `[-1, 1]` based on absolute max value. |

### Returns
`np.ndarray [N_atoms,]` — atom-level scores (one per atom in molecule).

### Example

```python
from rdkit import Chem
import numpy as np
from hotpot.cheminfo.fingerprint import assign_fp_values_atoms

mol = Chem.MolFromSmiles("CCO")
scores = np.random.randn(2048)
_, fps, bit_info = load_fp([mol])
atom_scores = assign_fp_values_atoms(
    mol, scores, list(fp.GetNonzeroElements().keys()), bit_info[0]
)
```

---

## 🧠 Typical Workflow Summary

1. **Fingerprint computation**
   ```python
   mols, fps, bit_info = load_fp(mols, generator='Morgen', gen_kw={'radius':2})
   ```
2. **ML model training or SHAP inference** using array representation:
   ```python
   X, bits = fps_to_array(fps)
   shap_values = model_shap(X)
   ```
3. **Atom-level interpretation:**
   ```python
   atom_values = assign_fp_values_atoms(mols[0], shap_values[0], bits, bit_info[0])
   ```
4. **Visualization (with `hotpot.cheminfo.draw`):**
   ```python
   from hotpot.cheminfo.draw import draw_single_mol
   draw_single_mol(mols[0], atom_hl_values=atom_values, colorbar=True)
   ```

---

## 🔍 Notes & Caveats

- **Supported generators:** `Morgen` (Morgan fingerprint), `RdKit`, `Torison` (Torsion), `AtomPair`.  
- **BitInfo maps** are only available for **Morgan** and **RDKit** fingerprints.  
- Ensure SHAP or feature vectors correspond to the same hashing/bit order.  
- Sparse representations (`ULongSparseIntVect`) are more efficient for large datasets.

---

## 🧩 Dependencies
- **RDKit** ≥ 2025.09  
- **NumPy** ≥ 1.23  

---

## ✨ Credits
Fingerprint utilities are part of the **Hotpot Cheminfo toolkit**, focused on explainable chemical features and transparent attribution visualization workflows.

