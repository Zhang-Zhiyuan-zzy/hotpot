# hotpot MCA inference

This is a source-independent ONNX inference package for site-resolved methyl
cation affinity (MCA, kJ/mol). It contains no PyTorch model definition, training
loop, optimizer state, training set or private checkpoint.

The directory is deliberately self-contained and uses relative imports. It can
be used in either layout:

```text
standalone_project/hotpot_mca/
hotpot/hotpot/cheminfo/AImodels/mca/
```

## Install

CPU:

```bash
python -m pip install -r hotpot_mca/requirements.txt
```

GPU: install the `onnxruntime-gpu` build matching the machine CUDA runtime in
place of `onnxruntime`. `device="auto"` uses an initialized ONNX CUDA provider
when possible and otherwise runs the same validated FP16 graph on CPU. Model
hashes are checked at session creation.

## Python API

```python
from hotpot_mca import MCAPredictor

predictor = MCAPredictor(device="auto")
one = predictor.predict("C1CCCCN1")
many = predictor.predict(["C1CCCCN1", "c1ccncc1"])

for site in one.sites:
    print(site.atom_index, site.site_type, site.mca_kj_mol)
```

An RDKit `Chem.Mol`, a `MoleculeGraph`, a pandas `Series`, or a hotpot
`Molecule` exposing `to_rdmol()` can be passed through the same method. Existing
3D conformers are retained; otherwise deterministic ETKDG plus MMFF/UFF geometry
generation is used. Geometry failure raises an error instead of silently using
2D or zero coordinates.

Charged molecules are outside the validated training domain and are rejected by
default. Use `MCAPredictor(allow_charged=True)` only when an explicitly
out-of-domain estimate is acceptable.

## Copy into hotpot

```bash
cp -a /home/zhangzhiyuan/MeCAP/infer_models/hotpot_mca \
  /path/to/hotpot/hotpot/cheminfo/AImodels/mca
```

Then import it with:

```python
from hotpot.cheminfo.AImodels.mca import MCAPredictor
```

No import edits are required. If model files are stored separately, set
`HOTPOT_MCA_MODEL_DIR=/absolute/path/to/model-directory` or pass `model_dir`.

## CLI

```bash
python -m hotpot_mca.cli 'C1CCCCN1' 'c1ccncc1'
```

The release graph is FP16 (about half the FP32 size). Its largest observed
deviation from the PyTorch checkpoint was 0.2331 kJ/mol on the 16-molecule
dynamic-shape validation set. An INT8 candidate was rejected because its error
was chemically unacceptable. The model estimates MCA, not the Mayr `N` or
`s_N N` experimental scales.
