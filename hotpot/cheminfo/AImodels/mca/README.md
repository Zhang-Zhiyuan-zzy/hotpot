# Hotpot MCA ONNX inference

This directory contains the inference-only ONNX implementation of the
site-resolved methyl cation affinity model (MCA, kJ/mol). It includes no
PyTorch model definition, optimizer state, training loop, training set, or
private checkpoint.

The package has two outputs:

- `atom_predictions`: an MCA estimate for every heavy atom supported by the
  model.
- `sites`: the important nucleophilic sites selected by the 24 ordered ESNUEL
  rules. These rules are evaluated by Hotpot's deterministic native RI-style
  SMARTS search, not by RDKit substructure matching. Their compiled queries
  share one target snapshot per molecule.

## Installation

The package is an integrated Hotpot component rather than a standalone Python
package. Install Hotpot and its dependencies, including Open Babel, NetworkX,
RDKit and ONNX Runtime. The inference compatibility dependencies used by CI are
listed in `tests/requirements-inference.txt`.

For GPU inference, install the `onnxruntime-gpu` build matching the machine's
CUDA runtime in place of `onnxruntime`. The CLI defaults to CPU. An explicit
GPU request is GPU-preferred: if the complete prediction call contains fewer
than 256 detected nucleophilic sites, Hotpot uses CPU inference without
starting CUDA on a cold predictor. This small-workload decision is silent by
default and is reported at debug logging level. Once the call reaches 256
sites and starts the GPU pipeline, CUDA initialization and OOM failures are
reported and never fall back to CPU. Model hashes are checked when the ONNX
session is created.

On first use, Hotpot derives an equivalent all-atom ONNX graph from the pinned
release graph. The derived graph reuses the verified model weights and computes
the shared molecular encoding once per distinct padding context instead of once
per target atom. It is stored below `HOTPOT_DERIVED_MODEL_HOME`, or below the
regular Hotpot model cache when that variable is unset. Read-only model bundles
remain supported; if the derived cache is also read-only, a temporary graph is
used. Original atom-row batch boundaries and padding are retained so existing
MCA values remain unchanged.

## Python API

Use the CPU predictor for a bounded workflow in which a persistent molecule
preprocessing pool and ONNX Runtime reuse the same CPU set in separate phases:

```python
from hotpot.cheminfo.AImodels.mca import MCACpuPredictor

with MCACpuPredictor(cpu_count=8) as predictor:
    one = predictor.predict("C1CCCCN1")
    many = predictor.predict(["C1CCCCN1", "c1ccncc1"])

for atom in one.atom_predictions:
    print(atom.atom_index, atom.element, atom.mca_kj_mol)
for site in one.sites:
    print(site.atom_index, site.site_type, site.mca_kj_mol)
```

For bulk GPU work, retain one predictor so its spawn workers and CUDA sessions
stay warm after the 256-site threshold is reached:

```python
from hotpot.cheminfo.AImodels.mca import MCAGpuPredictor

with MCAGpuPredictor(gpu_indices=(0, 1)) as predictor:
    for prediction in predictor.iter_predict(smiles_stream):
        consume(prediction)
```

The 256-site decision is made separately for each `predict()` or
`iter_predict()` call. A below-threshold call on an already warm predictor
submits no GPU inference, but leaves its existing persistent workers available
for a later bulk call.

`cpu_count=None` uses at most 8 CPUs from the process affinity. The same total
is used first by the fused site/conformer/feature pool and then by ONNX, so the
two stages never claim separate CPU budgets. Set `cpu_count` explicitly for a
different host allocation.

The GPU default is 1 site worker plus 3 conformer/feature workers per card.
There is no named configuration profile: change `site_cpu_per_gpu` and
`infer_cpu_per_gpu` directly when a job needs different allocation. The
predictor checks the process CPU affinity before starting and requires exactly
the configured number of distinct worker CPUs. Its rolling pipeline has two
bounds: preprocessing credit is released only after site detection and the
last feature batch handoff both finish, while a separate resident-atom cap is
released only by ordered output. This removes the whole-window barrier without
adding workers, CUDA sessions, request-queue capacity or in-flight GPU
requests. It does not change logical batches, MCA values, output order or
`is_Nuc_site`.

Hotpot object integration:

```python
from hotpot import read_mol
from hotpot.cheminfo.calculator import mca

mol = read_mol("c1ccccc1CN")
mca(mol, backend="cpu")

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
`MCACpuPredictor(allow_charged=True)` or the corresponding GPU option only for
an explicitly out-of-domain estimate.
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
hotpot mca 'C1CCCCN1'
hotpot mca molecules.sdf --backend gpu -o result.txt
hotpot mca 'C1CCCCN1' --plot mca.png
hotpot mca 'C1CCCCN1' --backend gpu --plot mca-all.png --all-site
```

The text output contains the one-based atom number, element, predicted MCA in
kJ/mol, and whether the atom was selected by nucleophilic-site detection. A
molecule file may contain multiple records. By default a plot labels and
colours only detected nucleophilic sites; `--all-site` includes every predicted
atom. The same interface remains available as
`python -m hotpot.cheminfo.AImodels.mca.cli`.
The CLI uses CPU when `--backend` is omitted; GPU-preferred mode must be
requested explicitly with `--backend gpu`. If the complete call has fewer than
256 predicted sites, it runs on CPU and does not start CUDA. This routing is
silent by default; use the top-level debug flag (`hotpot -d mca ...`) to report
it. Calls that reach 256 sites start the requested GPU pipeline.

On GPU, `--gpu-count N` selects logical indices `0..N-1`, while
`--gpu-indices 0,2` selects exact logical indices. If `--gpu-batch-size` is
omitted, Hotpot selects `64/128/256/512/1024/2048` from each card's total VRAM
and uses the smallest selected-card tier. A CUDA OOM does not retry or fall
back after GPU startup; the error reports the active batch and the next lower
explicit tier. CUDA/provider startup errors likewise do not trigger a CPU
fallback.

The release graph is FP16 (about half the FP32 size). Its largest observed
deviation from the PyTorch checkpoint was 0.2331 kJ/mol on the 16-molecule
dynamic-shape validation set. An INT8 candidate was rejected because its error
was chemically unacceptable. The model estimates MCA, not the Mayr `N` or
`s_N N` experimental scales.
