# Hotpot MCA ONNX inference

This directory contains the inference-only ONNX implementation of the
site-resolved methyl cation affinity model (MCA, kJ/mol). It includes no
PyTorch model definition, optimizer state, training loop, training set, or
private checkpoint.

The package has two outputs:

- `atom_predictions`: an MCA estimate for every heavy atom supported by the
  model.
- `sites`: the important nucleophilic sites selected by the 24 ordered ESNUEL
  rules. These rules are evaluated by Hotpot's NetworkX-based SMARTS search,
  not by RDKit substructure matching.

## Installation

The package is an integrated Hotpot component rather than a standalone Python
package. Install Hotpot and its dependencies, including Open Babel, NetworkX,
RDKit and ONNX Runtime. The inference compatibility dependencies used by CI are
listed in `tests/requirements-inference.txt`.

For GPU inference, install the `onnxruntime-gpu` build matching the machine's
CUDA runtime in place of `onnxruntime`. `device="auto"` selects an initialized
CUDA provider when available and otherwise uses CPU. Model hashes are checked
when the ONNX session is created.

## Python API

Direct SMILES and batch inference:

```python
from hotpot.cheminfo.AImodels.mca import MCAPredictor

predictor = MCAPredictor(device="auto")
one = predictor.predict("C1CCCCN1")
many = predictor.predict(["C1CCCCN1", "c1ccncc1"])

for atom in one.atom_predictions:
    print(atom.atom_index, atom.element, atom.mca_kj_mol)
for site in one.sites:
    print(site.atom_index, site.site_type, site.mca_kj_mol)
```

Hotpot object integration:

```python
from hotpot import read_mol
from hotpot.calculator import mca

mol = read_mol("c1ccccc1CN")
mca(mol)

for atom in mol.atoms:
    print(atom.mca)
print(mol.mca_sites)  # {hotpot.Atom: MCA in kJ/mol}
```

`Atom.mca` and `Molecule.mca_sites` are read-only and raise an informative
`AttributeError` until the calculator has run.

Site classification uses the ligand-skeleton SMARTS profile. Metal centres and
atoms directly bonded to a metal are excluded from `Molecule.mca_sites`, because
the exported MCA model was not validated for coordinated reaction sites. The
per-atom inference result remains available through `Atom.mca`; this filter only
controls the smaller, high-confidence site mapping.

## Input normalization

Hotpot `Molecule` is the canonical object used for domain checks, atom indices,
site detection and result construction. `hotpot.to_hotpot_mol()` is the shared
conversion entry point for RDKit `Chem.Mol`, Open Babel `OBMol`, Pybel
`Molecule`, SMILES/path inputs and objects exposing `to_rdmol()`. Source atom
order is retained, so `atom_index` maps directly to `mol.atoms[atom_index]`.

RDKit is still used for Uni-Mol feature generation and conformer construction;
it is not used as the substructure-search backend. For an RDKit input, the
original RDKit stereochemical representation is retained at that model-backend
boundary because Hotpot does not yet represent complete atom/bond
stereochemistry.

Existing usable 3D coordinates are retained. Otherwise deterministic ETKDG
plus MMFF/UFF geometry generation is used. Explicit graph hydrogen atoms are
rejected because the exported model predicts heavy-atom rows. Charged molecules
are outside the validated training domain and are rejected by default; use
`MCAPredictor(allow_charged=True)` only for an explicitly out-of-domain estimate.
The molecule-level charge must agree with the sum of atom formal charges; an
inconsistent graph is rejected rather than assigning the missing charge
implicitly.

## Copy into another Hotpot checkout

Copy this complete directory to:

```text
hotpot/cheminfo/AImodels/mca/
```

The receiving checkout must also contain the compatible `cheminfo.search`,
`cheminfo.convert`, core-object properties, and calculator integration. If model
files are stored separately, set `HOTPOT_MCA_MODEL_DIR` to their absolute
directory or pass `model_dir`.

## CLI

```bash
python -m hotpot.cheminfo.AImodels.mca.cli 'C1CCCCN1' 'c1ccncc1'
```

The release graph is FP16 (about half the FP32 size). Its largest observed
deviation from the PyTorch checkpoint was 0.2331 kJ/mol on the 16-molecule
dynamic-shape validation set. An INT8 candidate was rejected because its error
was chemically unacceptable. The model estimates MCA, not the Mayr `N` or
`s_N N` experimental scales.
