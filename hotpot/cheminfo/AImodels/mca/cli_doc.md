# `hotpot mca`

`hotpot mca` predicts atom-resolved methyl cation affinity (MCA) in kJ/mol.
It reports a value for every supported heavy atom and separately identifies the
important nucleophilic sites selected by Hotpot's curated site-detection rules.

## Command synopsis

```bash
$ hotpot mca [--backend {cpu,gpu}] [options] <SMILES/FILE> [<SMILES/FILE> ...]
```

Each positional input may be a SMILES string or a molecule file. Multiple
inputs are processed together so that ONNX inference can use batching.

## Single-molecule prediction

Quote SMILES strings to prevent the shell from interpreting special
characters:

```bash
$ hotpot mca 'c1ccccc1CN'
```

The table contains a one-based atom number, element, predicted MCA and the
site-detection decision:

```text
No.  Atom  MCA(kJ/mol)  is_Nuc_site
1    C     123.45       False
2    N     314.45       True
```

`is_Nuc_site=True` means that the atom is covered by the curated nucleophile
rules. It does not control whether an MCA value is predicted: supported heavy
atoms receive values regardless of site classification.

## Batch SMILES input

Pass several SMILES strings as positional arguments:

```bash
$ hotpot mca 'CN' 'CCO' 'c1ccccc1N'
```

Alternatively, provide a SMILES file with one record per line. An optional
name may follow each SMILES:

```text
CN methylamine
CCO ethanol
c1ccccc1N aniline
```

```bash
$ hotpot mca molecules.smi --backend gpu -o results.txt
```

## Batch molecule files

Shell globs expand to multiple positional file arguments:

```bash
$ hotpot mca inputs/*.mol2 --backend gpu -o results.txt
$ hotpot mca structures/*.sdf structures/*.mol2 --backend gpu -o results.txt
```

A multi-record file such as SDF or SMI is expanded into its individual
molecules. File inputs and direct SMILES may also be mixed:

```bash
$ hotpot mca 'CN' inputs/example.mol2 collection.sdf --backend gpu
```

The file format is normally inferred from the extension. Override it when the
extension is absent or non-standard:

```bash
$ hotpot mca molecule.data --backend cpu --input-format mol2
```

## Saving text output

Write the complete report with Hotpot's output option:

```bash
$ hotpot mca inputs/*.mol2 --backend gpu -o results.txt
```

Standard shell redirection is equivalent:

```bash
$ hotpot mca inputs/*.mol2 --backend gpu > results.txt
```

For multiple molecules, the report contains a numbered heading and atom table
for every molecule.

## Plotting MCA values

By default, only detected nucleophilic sites are coloured and labelled:

```bash
$ hotpot mca 'c1ccccc1CN' --backend cpu --plot mca.png
```

Use `--all-site` to colour every predicted atom:

```bash
$ hotpot mca 'c1ccccc1CN' --backend cpu --plot mca-all.png --all-site
```

Multiple input molecules are drawn as a grid in one output image:

```bash
$ hotpot mca molecules.smi --backend gpu --plot mca-grid.png
```

## Backend, GPU resources and batching

`--backend` defaults to `cpu`. Request GPU-preferred execution explicitly with
`--backend gpu`. Hotpot first uses the normal CPU site-detection stage to decide
whether the complete call is large enough for GPU startup. A call with fewer
than 256 predicted nucleophilic sites uses CPU inference without starting
CUDA; this decision is silent by default. Use the top-level debug flag to show
the routing decision:

```bash
$ hotpot -d mca molecules.smi --backend gpu
```

As soon as the call reaches 256 sites, Hotpot starts the requested GPU
pipeline. CUDA/provider startup errors and CUDA OOM after that point are not
retried on CPU.

```bash
$ hotpot mca molecules.smi --batch-size 64
$ hotpot mca molecules.smi --backend gpu --gpu-count 2
$ hotpot mca molecules.smi --backend gpu --gpu-indices 0,2
```

The GPU backend defaults to 9 site workers and 7 conformer/feature workers per
card. These are direct defaults, not a named profile. Override them with
`--site-cpu-per-gpu` and `--infer-cpu-per-gpu`. Hotpot calculates the required
CPU count from the selected GPUs and these values, then fails before startup if
the process affinity contains too few CPUs.

GPU preprocessing uses a fixed-credit rolling admission window. As capacity is
released, the coordinator admits more work without waiting for every task in a
discrete window to finish. It does not increase configured workers, maximum
in-flight GPU requests or peak admission credit. Logical atom-row batches,
result order, MCA values and `is_Nuc_site` semantics are unchanged.

GPU atom-row batch defaults come from total card memory: less than 8 GiB uses
64; 8/16/24/40/80 GiB boundaries select 128/256/512/1024/2048. Heterogeneous
cards use the smallest selected-card tier. `--gpu-batch-size` may explicitly
select one of those values. If CUDA runs out of memory, Hotpot reports the
current batch and next lower tier without retrying or switching to CPU.

## Model selection

The pinned FP16 model is selected by default:

```bash
$ hotpot mca molecules.smi --backend cpu --variant fp16
```

A local compatible model bundle can be selected explicitly:

```bash
$ hotpot mca molecules.smi --backend gpu --model-dir /absolute/path/to/mca/models
```

The directory must contain a compatible `manifest.json` and all ONNX external
data files referenced by that manifest. The same path may be configured with
the `HOTPOT_MCA_MODEL_DIR` environment variable.

Without an explicit directory, Hotpot downloads the pinned model revision from
Hugging Face on first use and verifies its graph and external-data SHA-256
checksums. Use `--model-source local` to prohibit network access, or preinstall
the model:

```bash
$ hotpot models install mca
```

## Charged molecules and applicability

The released model was validated on neutral molecules. Charged inputs are
rejected by default. To request an explicitly out-of-domain estimate, use:

```bash
$ hotpot mca '[NH4+]' --backend cpu --allow-charged
```

Treat such values cautiously. Explicit hydrogen atoms are not valid prediction
targets for this model; provide the corresponding implicit-hydrogen molecular
graph. Molecules may contain at most 512 model atoms.

## Reproducible conformer generation

Existing usable 3D coordinates are retained. Otherwise Hotpot creates a 3D
conformer with a deterministic seed. Change that seed when required:

```bash
$ hotpot mca molecules.smi --backend cpu --conformer-seed 123
```

## Complete option reference

Use the concise argparse reference for the current installation:

```bash
$ hotpot mca --help
```

Use this extended guide at any time with:

```bash
$ hotpot mca --doc
```
