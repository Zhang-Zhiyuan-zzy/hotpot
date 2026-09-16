# `hotpot mca`

`hotpot mca` predicts atom-resolved methyl cation affinity (MCA) in kJ/mol.
It reports a value for every supported heavy atom and separately identifies the
important nucleophilic sites selected by Hotpot's curated site-detection rules.

## Command synopsis

```bash
$ hotpot mca [options] <SMILES/FILE> [<SMILES/FILE> ...]
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
$ hotpot mca molecules.smi -o results.txt
```

## Batch molecule files

Shell globs expand to multiple positional file arguments:

```bash
$ hotpot mca inputs/*.mol2 -o results.txt
$ hotpot mca structures/*.sdf structures/*.mol2 -o results.txt
```

A multi-record file such as SDF or SMI is expanded into its individual
molecules. File inputs and direct SMILES may also be mixed:

```bash
$ hotpot mca 'CN' inputs/example.mol2 collection.sdf
```

The file format is normally inferred from the extension. Override it when the
extension is absent or non-standard:

```bash
$ hotpot mca molecule.data --input-format mol2
```

## Saving text output

Write the complete report with Hotpot's output option:

```bash
$ hotpot mca inputs/*.mol2 -o results.txt
```

Standard shell redirection is equivalent:

```bash
$ hotpot mca inputs/*.mol2 > results.txt
```

For multiple molecules, the report contains a numbered heading and atom table
for every molecule.

## Plotting MCA values

By default, only detected nucleophilic sites are coloured and labelled:

```bash
$ hotpot mca 'c1ccccc1CN' --plot mca.png
```

Use `--all-site` to colour every predicted atom:

```bash
$ hotpot mca 'c1ccccc1CN' --plot mca-all.png --all-site
```

Multiple input molecules are drawn as a grid in one output image:

```bash
$ hotpot mca molecules.smi --plot mca-grid.png
```

## Device selection and batching

`--device auto` is the default. It uses a working ONNX Runtime CUDA provider
when one is available and otherwise uses CPU inference.

```bash
$ hotpot mca molecules.smi --device cpu
$ hotpot mca molecules.smi --device cuda
$ hotpot mca molecules.smi --device auto --batch-size 256
```

`--device cuda` is strict and fails if the CUDA execution provider cannot be
initialized. GPU inference requires `onnxruntime-gpu` and CUDA/cuDNN versions
compatible with that package. `--batch-size` controls the number of atom-site
rows submitted per ONNX call rather than the number of molecules read at once.

## Model selection

The packaged FP16 model is selected by default:

```bash
$ hotpot mca molecules.smi --variant fp16
```

A separately distributed model bundle can be selected explicitly:

```bash
$ hotpot mca molecules.smi --model-dir /absolute/path/to/mca/models
```

The directory must contain a compatible `manifest.json` and all ONNX external
data files referenced by that manifest. The same path may be configured with
the `HOTPOT_MCA_MODEL_DIR` environment variable.

## Charged molecules and applicability

The released model was validated on neutral molecules. Charged inputs are
rejected by default. To request an explicitly out-of-domain estimate, use:

```bash
$ hotpot mca '[NH4+]' --allow-charged
```

Treat such values cautiously. Explicit hydrogen atoms are not valid prediction
targets for this model; provide the corresponding implicit-hydrogen molecular
graph. Molecules may contain at most 512 model atoms.

## Reproducible conformer generation

Existing usable 3D coordinates are retained. Otherwise Hotpot creates a 3D
conformer with a deterministic seed. Change that seed when required:

```bash
$ hotpot mca molecules.smi --conformer-seed 123
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
