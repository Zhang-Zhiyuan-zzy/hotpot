# 🧪 PubChemService — Quick Guide (Generated from GPT-5)

A lightweight wrapper for **PubChemPy**, enabling quick conversion between  
**CAS**, **SMILES**, **CID**, and **chemical names**, with caching for speed.

---

## 🚀 Example Tutorial

```pycon
from pubchem_service import PubChemService

pcs = PubChemService()

# Common conversions
pcs.smi_to_cas("CCO")  # → '64-17-5'
pcs.cid_to_name(702)  # → 'Ethanol'
pcs.name_to_smi("Ethanol")  # → 'CCO'

# General conversion interface
pcs.convert("64-17-5", "cas", "smiles")  # → 'CCO'
pcs.convert("CCO", "smiles", "name")  # → 'Ethanol'

# Cached automatically: repeated calls are instant
pcs.smi_to_cas("CCO")
```

---

## 📘 API Reference

### `class PubChemService(verbose: bool = False)`
Wrapper for PubChem chemical identifier conversions.

---

### 🔹 `convert(identifier, from_type, to_type)`
Universal converter.  
**from_type/to_type:** `'cas'`, `'smiles'`, `'cid'`, `'name'`  
→ returns `str` or `None`

**Example**
```pycon
pcs.convert("64-17-5", "cas", "cid")  # → 702
```

---

### 🔹 Individual Methods

| Function | Input → Output | Description |
|-----------|----------------|-------------|
| `smi_to_cas(smiles)` | SMILES → CAS | Get CAS from SMILES |
| `smi_to_name(smiles)` | SMILES → Name | Get first synonym (name) |
| `smi_to_cid(smiles)` | SMILES → CID | Get Compound ID |
| `cid_to_smi(cid)` | CID → SMILES | Get canonical SMILES |
| `cid_to_cas(cid)` | CID → CAS | Get CAS number |
| `cid_to_name(cid)` | CID → Name | Get first synonym |
| `name_to_smi(name)` | Name → SMILES | Convert chemical/common name |
| `name_to_cid(name)` | Name → CID | Get CID from name |

---

### Notes
- Cached for up to 256 results (speeds up repeat queries)
- Returns `None` if PubChem record not found
- Works with both chemical names and identifiers

---

**That’s it — load, query, and convert.**  
Perfect for Jupyter, batch scripts, or integration into data pipelines.
