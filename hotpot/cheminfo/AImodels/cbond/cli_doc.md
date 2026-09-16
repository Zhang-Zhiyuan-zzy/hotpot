# `hotpot cbond`

`hotpot cbond` predicts metal-ligand coordination patterns with Hotpot's CBond
ONNX model. It accepts one metal and one ligand, builds the predicted
coordination bond or bonds, and reports the resulting structure as SMILES.

## Command synopsis

```bash
$ hotpot cbond <METAL> <LIGAND_SMILES/FILE> [options]
```

`METAL` may be an element symbol such as `Eu` or an atomic number such as
`63`. `LIGAND_SMILES/FILE` may be a quoted SMILES string or the path to a
supported molecule file.

## Predict the highest-ranked structure

Quote a SMILES string so that shell metacharacters are not interpreted:

```bash
$ hotpot cbond Eu 'CN'
```

By default, the command prints only the resulting SMILES. This compact output
is convenient for shell pipelines:

```text
C[NH2+][Eu]
```

A larger ligand can be submitted in exactly the same way:

```bash
$ hotpot cbond Eu 'O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C(C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3'
```

The atomic number form is equivalent to the element-symbol form:

```bash
$ hotpot cbond 63 'CN'
```

## Read a ligand from a molecule file

Pass a molecule-file path in place of the SMILES string. The format is
normally inferred from the filename extension:

```bash
$ hotpot cbond Eu inputs/ligand.mol2
$ hotpot cbond Eu inputs/ligand.sdf
```

Use `--input-format` when the extension is absent or non-standard:

```bash
$ hotpot cbond Eu inputs/ligand.data --input-format mol2
```

The command processes one ligand per invocation. A shell loop can process a
directory while retaining one output record per input file:

```bash
$ for ligand in inputs/*.mol2; do hotpot cbond Eu "$ligand"; done
```

## Enumerate candidate coordination structures

Use `--all-structures` to report every unique coordination structure retained
by the CBond search rather than only the highest-ranked result:

```bash
$ hotpot cbond Eu ligand.mol2 --all-structures
```

Results are sorted from highest to lowest relative probability:

```text
C[NH2+][Eu]  --> Rank 1: Prob: 74.9%
-----
CN[Eu]       --> Rank 2: Prob: 25.1%
-- End --
```

Each construction path has a path weight equal to the product of the sigmoid
values of its sequential raw CBond logits. Paths that produce the same final
coordination topology are merged by adding their weights. The merged weights
are then normalized over the unique structures returned by that invocation,
producing the reported `Prob` values.

`Prob` is therefore a relative ranking probability within the candidate set
defined by the input, model, and threshold. It is not a calibrated physical
probability, an equilibrium population, or a thermodynamic quantity. Changing
the threshold can change both the returned candidate set and its normalized
probabilities.

## Show coordination-bond details

Add `--bond-detail` to show the atom and score for every predicted metal-ligand
bond. For a single best structure, no rank or normalized probability is
printed:

```bash
$ hotpot cbond Eu ligand.mol2 --bond-detail
```

```text
C[NH2+][Eu]
Cbond Detail:
AtomIdx  Atom  Score
4        O     0.88682
11       N     0.23455
-- End --
```

`Score` is the raw per-step CBond model logit recorded along the selected
construction path. It is distinct from both its sigmoid value and the final
structure-level `Prob` reported by `--all-structures`.
`AtomIdx` uses Hotpot's zero-based atom index for the input ligand.

Combine both flags to inspect every ranked structure and its individual bond
decisions:

```bash
$ hotpot cbond Eu ligand.mol2 --all-structures --bond-detail
```

```text
C[NH2+][Eu]  --> Rank 1: Prob: 74.9%
Cbond Detail:
AtomIdx  Atom  Score
4        O     0.88682
11       N     0.23455
-----
CN[Eu]  --> Rank 2: Prob: 25.1%
Cbond Detail:
AtomIdx  Atom  Score
11       N     0.71124
-- End --
```

When multiple construction paths lead to the same final structure, `Prob`
includes all merged path weights. The detail table represents the retained
highest-weight path for that structure, and its row order is the predicted
bond-addition order.

## Save the output

Use `-o` or `--output` to write exactly the same report to a UTF-8 text file:

```bash
$ hotpot cbond Eu ligand.mol2 -o result.smi
$ hotpot cbond Eu ligand.mol2 --all-structures --bond-detail -o result.txt
```

Standard shell redirection is also supported:

```bash
$ hotpot cbond Eu ligand.mol2 --all-structures > result.txt
```

Use a text-like extension when requesting ranks or bond details because that
output contains more than a single SMILES record.

## Score threshold

The default threshold is `-0.125`:

```bash
$ hotpot cbond Eu ligand.mol2 --threshold -0.125
```

The threshold is applied to the model's **raw logit**, not to the final
normalized `Prob`. Candidates whose raw score does not exceed the threshold
are not expanded. A higher threshold retains fewer, more strongly scored bond
decisions; a lower threshold explores more alternatives and may produce more
candidate structures.

Because `Prob` is normalized only after candidate paths have been generated
and equivalent structures have been merged, probabilities from runs using
different thresholds are not directly comparable.

Exact enumeration caches each unique set of bonded donor atoms, but the
number of such sets can still grow exponentially. `--all-structures` stops
with an explicit error after 4096 unique states by default. Increase that
guard only when the larger search is intentional:

```bash
$ hotpot cbond Eu ligand.mol2 --all-structures --max-states 8192
```

Raising the score threshold is usually the more efficient way to reduce an
overly broad search. The single best-structure mode does not use this limit.

## Greedy search control

The default search continues past a candidate that has already been connected
and considers the next eligible site. Use `--no-greedy` to stop in that
situation:

```bash
$ hotpot cbond Eu ligand.mol2 --no-greedy
```

This option changes path construction and can therefore change both the final
coordination pattern and the relative probabilities produced by
`--all-structures`.

## Device and model selection

`--device auto` is the default. It uses a working ONNX Runtime CUDA provider
when one is available and otherwise uses CPU inference:

```bash
$ hotpot cbond Eu ligand.mol2 --device auto
$ hotpot cbond Eu ligand.mol2 --device cpu
$ hotpot cbond Eu ligand.mol2 --device cuda
```

`--device cuda` is strict and fails if the CUDA execution provider cannot be
initialized. GPU execution requires `onnxruntime-gpu` and compatible CUDA and
cuDNN libraries.

Select an external compatible CBond model bundle with `--model-dir`:

```bash
$ hotpot cbond Eu ligand.mol2 --model-dir /absolute/path/to/cbond/onnx
```

The same location may be configured with the `HOTPOT_CBOND_MODEL_DIR`
environment variable. The model directory must contain the expected manifest
and ONNX artifacts.

## Model scope and interpretation

The command predicts coordination connectivity; it does not optimize a 3D
complex, calculate binding energy, or estimate thermodynamic stability. The
current workflow supports one metal centre and one ligand graph per
invocation. The packaged runtime supports at most 32 ligand rings and at most
64 atoms in one ring.

Candidate ranks and scores should be interpreted as outputs of the installed
CBond model within its training and applicability domain. They are most useful
for ranking plausible coordination patterns before downstream structural or
quantum-chemical validation.

## Complete option reference

Show the concise option reference for the installed version:

```bash
$ hotpot cbond --help
```

Show this extended guide in the terminal:

```bash
$ hotpot cbond --doc
```
