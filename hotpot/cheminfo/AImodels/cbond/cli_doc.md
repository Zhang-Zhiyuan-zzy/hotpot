# `hotpot cbond`

`hotpot cbond` uses Hotpot's packaged CBond ONNX model to add coordination
bonds between one metal centre and one ligand graph. It prints the resulting
connectivity as SMILES; it does not optimize a three-dimensional structure.

## Verification basis

The concrete output blocks below were captured from an installed
`hotpot-zzy 0.5.3.0` wheel using the packaged CBond models, ONNX Runtime
1.30.0, Open Babel 3.2.1, and CPU inference. They are command output, not
invented formatting examples.

Only standard output is shown. ONNX Runtime may independently print provider
or telemetry warnings to standard error. Such warnings are not part of the
CBond result and are not written by `-o` or shell `>` redirection. Canonical
SMILES spelling and stereochemical annotations can vary with the Open Babel
version even when the predicted bond set is unchanged.

## Command synopsis

```bash
$ hotpot cbond <METAL> <LIGAND_SMILES/FILE> [options]
```

`METAL` may be an element symbol such as `Eu` or its atomic number, `63`.
Quote SMILES strings to protect shell metacharacters.

## Build one greedy structure

The default mode iteratively adds the currently highest-scoring eligible bond
and prints one SMILES line:

```bash
$ hotpot cbond Eu 'CN' --device cpu
```

```text
C[NH2+][Eu]
```

The atomic-number form was verified to produce the same result:

```bash
$ hotpot cbond 63 'CN' --device cpu
```

The default result is the product of a greedy sequence. It is not guaranteed
to be Rank 1 after `--all-structures` merges and normalizes every enumerated
path. The multi-structure example below demonstrates this distinction.

## Read a ligand from a file

The file format is normally inferred from its extension. For a morpholine
structure stored as `morpholine.mol2`, the tested command was:

```bash
$ hotpot cbond Eu morpholine.mol2 --device cpu
```

```text
C1C[N@H+]2[Eu][O]1CC2
```

The same molecule converted to `morpholine.sdf` produced the same output.
Use `--input-format` when the extension is absent or non-standard:

```bash
$ hotpot cbond Eu morpholine.data --input-format mol2 --device cpu
```

Each invocation accepts one ligand input. Use a shell loop for a collection of
files:

```bash
$ for ligand in inputs/*.mol2; do hotpot cbond Eu "$ligand"; done
```

## Enumerate terminal coordination structures

`--all-structures` explores every terminal donor-index state admitted by the
threshold policy and ranks the merged state weights. This command was run
against 2-amino-1,3-propanediol:

```bash
$ hotpot cbond Eu 'NCC(O)CO' --device cpu --all-structures
```

```text
C1O[Eu]2O[C@@H]1C[NH2+]2  --> Rank 1: Prob: 57.9%
-----
NC[C@@H]1CO[Eu]O1  --> Rank 2: Prob: 42.1%
-- End --
```

For comparison, default greedy inference on the same SMILES printed
`NC[C@@H]1CO[Eu]O1`, which is Rank 2 after full path aggregation. Therefore,
the default mode must not be interpreted as an alias for Rank 1.

For each ordered construction path, the backend multiplies
`sigmoid(raw_logit)` over its bond-addition steps. It then:

1. sums the weights of all paths reaching the same donor-atom index set;
2. normalizes the merged weights over all terminal states;
3. sorts the states by that normalized value.

`Prob` is this normalized relative path weight. It is not a calibrated
physical probability, equilibrium population, or thermodynamic quantity.

The merge key is the set of Hotpot atom indices, not canonical-SMILES graph
isomorphism. Symmetry-related donor sets can therefore remain separate and may
occasionally print the same canonical SMILES.

A state is terminal only when no unconnected candidate has a raw score above
the threshold. `--all-structures` does not return every arbitrary subset of
donor atoms.

## Show bond details

`--bond-detail` adds the raw model score for each bond selected by the default
greedy path:

```bash
$ hotpot cbond Eu 'CN' --device cpu --bond-detail
```

```text
C[NH2+][Eu]
Cbond Detail:
AtomIdx  Atom  Score
1        N     0.08525
-- End --
```

Without `--all-structures`, no rank or normalized probability is printed.

Combine both flags to inspect every ranked terminal state:

```bash
$ hotpot cbond Eu 'NCC(O)CO' --device cpu --all-structures --bond-detail
```

```text
C1O[Eu]2O[C@@H]1C[NH2+]2  --> Rank 1: Prob: 57.9%
Cbond Detail:
AtomIdx  Atom  Score
0        N     0.89586
3        O     3.60561
5        O     4.41528
-----
NC[C@@H]1CO[Eu]O1  --> Rank 2: Prob: 42.1%
Cbond Detail:
AtomIdx  Atom  Score
3        O     2.32296
5        O     3.09254
-- End --
```

`Score` is the raw per-step CBond logit, not a sigmoid probability. When
several construction orders reach one donor set, the table shows the
highest-weight representative path; row order is bond-addition order.

`AtomIdx` is Hotpot's zero-based atom index after the molecule has been parsed
and normalized. It is not necessarily an atom serial number stored in a MOL2
or SDF file.

## Save the result

`-o` writes the same standard-output report to a UTF-8 file:

```bash
$ hotpot cbond Eu morpholine.mol2 --device cpu -o result.smi
$ hotpot cbond Eu 'NCC(O)CO' --all-structures --bond-detail -o result.txt
```

Shell redirection was also verified:

```bash
$ hotpot cbond Eu 'NCC(O)CO' --all-structures > result.txt
```

Use a text-like extension when ranks or bond details are enabled because the
output is no longer a one-record SMILES file. ONNX Runtime warnings written to
standard error remain visible in the terminal and are not included in these
files.

## Threshold and empty results

The default threshold is `-0.125` and is applied to each raw model logit using
the strict rule `score > threshold`:

```bash
$ hotpot cbond Eu 'CN' --threshold -0.125
```

A higher threshold explores fewer bond additions; a lower threshold can
produce more states. Because the terminal candidate set can change,
normalized `Prob` values from different thresholds are not directly
comparable.

The two output modes intentionally report an empty search differently. This
single-structure command was verified to exit with status 1 and end with the
shown exception:

```bash
$ hotpot cbond Eu 'CN' --device cpu --threshold 999
```

```text
ValueError: No coordination bond exceeded the raw-score threshold 999.0
```

The current top-level CLI includes a Python traceback before that final line.
By contrast, all-structures mode exits with status 0 and prints:

```bash
$ hotpot cbond Eu 'CN' --device cpu --threshold 999 --all-structures
```

```text
No coordination structures exceeded the threshold.
-- End --
```

## Enumeration limit

Exact state enumeration can grow exponentially. The default limit is 4096
unique donor-index states. Exceeding it raises a `RuntimeError`; for example,
the tested `--max-states 1` run ended with:

```text
RuntimeError: CBond enumeration exceeded max_states=1; raise --threshold or --max-states
```

Raise the threshold to reduce the search, or deliberately increase the guard:

```bash
$ hotpot cbond Eu ligand.mol2 --all-structures --max-states 8192
```

The single greedy mode does not use this state limit.

## `--no-greedy`

After a bond is added, the model scores every candidate again. In default mode,
an already-connected highest-scoring atom is skipped so that the next eligible
atom can be considered. `--no-greedy` stops the path instead:

```bash
$ hotpot cbond Eu ligand.mol2 --no-greedy
```

This can change both the greedy result and the terminal states generated by
`--all-structures`. It is not a switch between greedy search and a global
optimizer; full terminal-state enumeration is controlled separately by
`--all-structures`.

## Device and model selection

`--device auto` is the default. It uses a working ONNX Runtime CUDA provider
when available and otherwise uses CPU:

```bash
$ hotpot cbond Eu ligand.mol2 --device auto
$ hotpot cbond Eu ligand.mol2 --device cpu
$ hotpot cbond Eu ligand.mol2 --device cuda
```

`--device cuda` is strict. In the CPU-only verification environment it exited
with status 1 and ended with:

```text
RuntimeError: CUDAExecutionProvider is not available
```

GPU execution requires `onnxruntime-gpu` plus mutually compatible CUDA and
cuDNN libraries.

Select an external compatible CBond bundle with `--model-dir`:

```bash
$ hotpot cbond Eu ligand.mol2 --model-dir /absolute/path/to/cbond/onnx
```

`HOTPOT_CBOND_MODEL_DIR` provides the same model-directory setting. The
directory must contain the expected manifest and ONNX artifacts.

## Scope and interpretation

- One metal centre and one ligand graph are supported per invocation.
- Candidate donor elements are currently O, N, S, P, Si, and B.
- The command predicts connectivity; it does not optimize geometry, calculate
  binding energy, or estimate thermodynamic stability.
- The packaged runtime accepts at most 32 ligand rings and 64 atoms in one
  ring.
- Scores and rankings remain model outputs and should be validated downstream
  when used outside the model's training domain.

## Complete option reference

```bash
$ hotpot cbond --help
$ hotpot cbond --doc
```
