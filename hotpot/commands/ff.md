# `hotpot ff`

`hotpot ff` prepares and optimizes molecular three-dimensional coordinates
with Hotpot's force-field workflows. It accepts direct SMILES, molecule files,
or standard input and writes only the optimized molecular structure to standard
output.

The command does not predict coordination bonds. Use `hotpot cbond` first when
a metal--ligand connectivity model is required; `hotpot ff` then builds and
optimizes the supplied topology.

## Command synopsis

```bash
$ hotpot ff [options] <SMILES/FILE> [<SMILES/FILE> ...]
$ hotpot ff [options] -
```

Each positional input is resolved in this order:

1. `-` means standard input;
2. an existing path is read as a molecule file;
3. any other value is parsed as a direct SMILES string.

Use `--input-format` to disambiguate an input whose format cannot be inferred.
Quote direct SMILES so that shell metacharacters are not interpreted.

## Default workflow

Without `--rebuild` or `--optimize-only`, the command selects its workflow from
the molecular topology. Hotpot considers coordinates three-dimensional when
the molecule has at least two atoms and their coordinate vectors are not all
coincident:

- an organic molecule with non-coincident coordinates is optimized from those
  coordinates;
- an organic molecule whose coordinates are all coincident is built in 3D and
  then optimized;
- a metal-containing molecule first receives a fresh native `FAST` build and
  optimization; if that candidate fails the geometry gate, Hotpot discards it
  and runs the complete complex build workflow from the original topology.

Use `--optimize-only` when an existing metal-complex geometry must be retained
as the optimization starting point.

Missing hydrogens are added before force-field work by default. Pass
`--no-add-hydrogens` only when the supplied hydrogen topology must be retained
exactly.

The complex workflow scans rings up to 16 atoms. This limit is fixed in the
current CLI: larger rings are reported as excluded evidence and are not
silently treated as proven non-intersections.

## Direct SMILES

Build and optimize a molecule from SMILES:

```bash
$ hotpot ff 'c1ccccc1CN'
```

Standard output is the optimized structure payload. It contains no progress
messages or decorative text, so it can be redirected directly:

```bash
$ hotpot ff 'c1ccccc1CN' > benzylamine.mol2
```

The default output format for a terminal or redirected standard output is
MOL2. Select another format explicitly when required:

```bash
$ hotpot ff 'c1ccccc1CN' --output-format sdf > benzylamine.sdf
```

## Molecule files

Process a molecule file with automatic routing:

```bash
$ hotpot ff complex.mol2 -o complex-optimized.mol2
```

`-o/--output` writes the same molecular payload that would otherwise be sent
to standard output. When `-o` is used, standard output is empty:

```bash
$ hotpot ff complex.mol2 > complex-redirected.mol2
$ hotpot ff complex.mol2 -o complex-option.mol2
```

The two files above contain equivalent payloads. An explicit
`--output-format` takes precedence over format inference from the output
suffix. Existing output paths are rejected unless `--overwrite` is supplied.

Use `--rebuild` to discard existing coordinates, construct a new starting
geometry, and optimize it:

```bash
$ hotpot ff complex.mol2 --rebuild -o complex-rebuilt.mol2
```

Use `--optimize-only` to require non-coincident input coordinates, prohibit an
implicit build, and optimize from the supplied geometry:

```bash
$ hotpot ff complex.mol2 --optimize-only -o complex-optimized.mol2
```

`--rebuild` and `--optimize-only` are mutually exclusive.

## Standard input and pipelines

`-` reads molecular data from standard input. Specify the format because stdin
has no filename suffix:

```bash
$ printf '%s\n' 'CCO' | hotpot ff - --input-format smi > ethanol.mol2
```

Multi-record stdin is consumed in record order using the selected format.

The default one-structure output of `hotpot cbond` is one connected SMILES
line, so it can feed `hotpot ff` directly:

```bash
$ hotpot cbond Eu 'O=C(N(C)CCC)C1=CC=CC=C1' \
    | hotpot ff - --input-format smi -o eu-complex.mol2
```

In this pipeline, `cbond` determines metal--ligand connectivity and `ff`
constructs and optimizes the three-dimensional complex. Force-field warnings
and progress remain on standard error and do not contaminate the molecular
payload.

The current human-readable output produced by `hotpot cbond --all-structures`
or `--bond-detail` contains ranks, separators, and detail tables. It is not a
SMILES stream and must not be piped into `hotpot ff`. Save or select an
individual undecorated CBond structure first.

## Batch inputs

Pass multiple SMILES or paths in one invocation. Shell globs are expanded by
the shell before Hotpot receives them:

```bash
$ hotpot ff inputs/*.mol2 --output-format sdf > optimized.sdf
$ hotpot ff 'CN' 'CCO' structures/example.mol2 -o optimized.sdf
```

`--jobs` controls molecule-level parallelism:

```bash
$ hotpot ff inputs/*.mol2 --jobs 16 -o optimized.sdf
```

Output order follows input order, independently of worker completion order.
`--jobs` does not make one Open Babel force-field optimization use that many
threads. Each worker needs its own molecule and force-field state, so memory
and CPU use rise with the selected job count.

If any batch member fails, successful structures remain in the ordered output,
the failed member contributes no optimized structure, and the process exits
with status 1. Use `--report` to retain per-input status and diagnostics.

## Workflow selection

`--route` selects the scientific workflow:

| Value | Meaning |
| --- | --- |
| `auto` | Use the organic route for an organic topology. For a metal-containing topology, try a fresh native `FAST` build first and fall back to the complete complex workflow only if it is rejected. This is the default. |
| `organic` | Use the ordinary molecular build/optimization workflow explicitly. |
| `complex` | Use coordination restoration, ring--bond untangling, and complex optimization explicitly. |

Explicit routes are strict requests; they are not silently replaced by another
route.

`--forcefield` accepts `auto`, `uff`, `mmff94`, `mmff94s`, `gaff`, or
`ghemical`. In `auto` mode, the current complex backend uses UFF and the
organic backend uses its normal default. A complex route explicitly requesting
anything except `uff` is rejected rather than silently rewritten to UFF.

```bash
$ hotpot ff ligand.mol2 --route organic --forcefield mmff94s
$ hotpot ff complex.mol2 --route complex --forcefield uff
```

Select the Open Babel optimization algorithm with `--algorithm`:

```bash
$ hotpot ff complex.mol2 --algorithm conjugate
$ hotpot ff ligand.mol2 --algorithm steepest
```

The supported values are `conjugate` and `steepest`. `--epochs` sets the
number of optimization epochs and `--steps-per-epoch` sets the Open Babel
steps performed in each epoch:

```bash
$ hotpot ff complex.mol2 --epochs 100 --steps-per-epoch 100
```

These options are computational limits, not guarantees that every run will
consume every permitted step.

## Quality assessment

`--quality` selects the force-field acceptance profile:

| Value | Meaning |
| --- | --- |
| `off` | Do not apply the final quality acceptance profile. Backend execution errors still fail. |
| `basic` | Apply the least restrictive predefined validation profile. |
| `standard` | Apply the normal predefined validation profile. This is the default. |
| `strict` | Apply the most restrictive predefined validation profile. |

The profile affects final acceptance, not the molecular output format. A
returned structure that fails the requested profile causes exit status 1 and
is identified in `--report`. A hard build or optimization failure emits no
optimized structure for that input.

## Reproducibility and limits

Use `--seed` to control Hotpot's stochastic coordinate construction and
perturbation steps:

```bash
$ hotpot ff complex.mol2 --rebuild --seed 2026 -o rebuilt.mol2
```

A seed does not remove differences caused by Open Babel versions, force-field
plugins, platform floating-point behavior, or worker scheduling.

`--timeout` specifies the wall-clock limit in seconds for the isolated
coordinate-building subprocess:

```bash
$ hotpot ff inputs/*.mol2 --timeout 900 --jobs 8 -o optimized.sdf
```

A coordinate-build timeout is a failed input, not a partial success. This
option does not impose a wall-clock limit on force-field optimization and has
no effect on an `--optimize-only` run.

## Trajectory evidence

Use `--trajectory DIRECTORY` to serialize the recorded force-field frames and
topology revisions for later inspection:

```bash
$ hotpot ff complex.mol2 \
    --trajectory reproduce/trajectory \
    -o complex-optimized.mol2
```

Choose the earliest retained workflow stage with `--trajectory-start`:

| Value | First retained stage |
| --- | --- |
| `ligand-build` | Initial component building. |
| `coordination-restoration` | The built ligand and unbonded metal, immediately before coordination restoration. |
| `complex-untangling` | Full-complex ring--bond untangling. |
| `final-optimization` | Final optimization only. |

The default is workflow-specific: a complex build starts at
`coordination-restoration`, optimization of an existing complex starts at
`complex-untangling`, and an organic workflow starts at `final-optimization`.

```bash
$ hotpot ff complex.mol2 \
    --trajectory reproduce/trajectory \
    --trajectory-start ligand-build \
    -o complex-optimized.mol2
```

Recording earlier stages increases memory, serialization work, and disk use.
Trajectory evidence already written may remain available when a later stage
hard-fails; its presence does not imply successful optimization.

## JSON report

`--report FILE` writes a machine-readable JSON document with ordered
per-input execution status, requested and effective backend information,
energies in kJ/mol where available, quality results, warnings, and failure
diagnostics:

```bash
$ hotpot ff inputs/*.mol2 \
    --report reproduce/ff-report.json \
    -o optimized.sdf
```

The report is separate from the structure payload. JSON numbers are always
finite JSON values; unavailable measurements are represented as `null`, never
as `NaN` or `Infinity`. Any atom indices present in reports or trajectory
evidence use Hotpot's zero-based indexing. Known force-field failures retain
available typed setup, quality, build, worker, and trajectory-presence evidence
under the corresponding result's `error.evidence` object.

## Output streams and exit status

- Standard output contains only serialized optimized structures.
- Standard error contains warnings, progress, diagnostics, and failure text.
- `-o/--output` receives exactly the payload otherwise written to standard
  output and leaves standard output empty.
- A hard-failed input produces no optimized structure. Requested trajectory
  evidence may still remain on disk for diagnosis.
- A structure that fails the selected quality profile is still emitted for
  inspection, while a concise warning is written to standard error.
- Exit status `0` means every returned structure passed the requested quality
  profile.
- Exit status `1` means a known force-field failure occurred or at least one
  returned structure failed the requested quality profile.
- Exit status `2` means command-line parsing or invocation validation failed.
- Unexpected program errors remain nonzero failures and are not converted to
  empty successful results.

## Option summary

| Option | Purpose |
| --- | --- |
| `--input-format FORMAT` | Override input inference, especially for stdin or non-standard suffixes. |
| `-o, --output PATH` | Write the structure payload to a path instead of stdout. |
| `--output-format FORMAT` | Select structure serialization explicitly. |
| `--overwrite` | Permit replacement of existing output/report/trajectory targets. |
| `--rebuild` | Rebuild initial 3D coordinates before optimization. |
| `--optimize-only` | Require and optimize existing 3D coordinates without building. |
| `--route {auto,organic,complex}` | Select automatic or explicit workflow routing. |
| `--forcefield {auto,uff,mmff94,mmff94s,gaff,ghemical}` | Select the Open Babel force field. |
| `--algorithm {conjugate,steepest}` | Select the minimization algorithm. |
| `--epochs N` | Set the optimization epoch limit. |
| `--steps-per-epoch N` | Set Open Babel steps per epoch. |
| `--convergence-level {openbabel,fast,balanced,strict}` | Select convergence evidence; `fast` is the default and `strict` preserves the former behavior. |
| `--no-add-hydrogens` | Preserve the supplied hydrogen topology. |
| `--quality {off,basic,standard,strict}` | Select final quality acceptance. |
| `--seed N` | Seed stochastic build and perturbation operations. |
| `--timeout SECONDS` | Set the coordinate-building subprocess limit; it does not limit optimization. |
| `--trajectory DIRECTORY` | Serialize trajectory and topology evidence. |
| `--trajectory-start STAGE` | Select the earliest recorded workflow stage. |
| `--report FILE` | Write the ordered JSON execution report. |
| `--jobs N` | Set molecule-level parallel workers. |

Use concise parser help without running an optimization:

```bash
$ hotpot ff --help
```

Display this extended Markdown guide with:

```bash
$ hotpot ff --doc
```
