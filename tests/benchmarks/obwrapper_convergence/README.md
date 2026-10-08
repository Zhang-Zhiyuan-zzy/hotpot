# obWrappers convergence-level benchmark

This optional benchmark compares all four convergence policies from an
identical three-dimensional starting point for every molecule. It reuses the
unique `build_complete` frame saved by the independent obWrappers workflow for
both ligand and complex targets.

The optimization clocks wrap the complete public `obWrappers.optimize(...)`
call, including molecule packing, the single Python/C++ boundary crossing, and
result materialization. Native-module loading, source-trajectory I/O, standard
geometry validation, and artifact writing are outside both clocks. Frame and
epoch-history retention are disabled.

## Full run

From the repository root:

```bash
$ python -m tests.benchmarks.obwrapper_convergence \
    --source movie/benchmarks/extractants_eu_187_four_workflows_m05_20261008/obwrappers \
    --cohort movie/benchmarks/extractants_eu_187_native_backends_traces_m05_20261008/canonical_cases.json \
    --input molecules/extractant/extractants.smi \
    --output movie/benchmarks/obwrapper_convergence_levels_20261008 \
    --workers 16
```

Use `--resume` to retain completed case/target pairs after an interrupted run.
Use `--cases 1,62,187` or `--targets ligand` for a focused run. The default
scientific budget is 100 epochs with 100 steps per epoch and UFF; override it
with `--epochs`, `--steps-per-epoch`, or `--forcefield`.

The manifest records the policy definitions used by this implementation:

| Level | Evidence required after the Open Babel stop signal |
|---|---|
| `OPENBABEL` | A numerically usable state only |
| `FAST` | RMS gradient <= 3 and maximum gradient <= 10 kJ/(mol Å) |
| `BALANCED` | RMS gradient <= 1 and maximum gradient <= 5 kJ/(mol Å) |
| `STRICT` | Maximum gradient <= 0.1 in backend energy units per Å |

## Pairing and denominators

For each case and target, `OPENBABEL`, `FAST`, `BALANCED`, and `STRICT` receive
independent molecule instances reconstructed from exactly the same atoms,
bonds, and coordinate bytes. The shared start has a SHA-256 fingerprint in
every result row. Savings are paired differences relative to that target's
`STRICT` run. The execution order is deterministically rotated across cases
and targets so that warm-cache effects do not systematically favor one level.

A non-finite `build_complete` frame is recorded four times with
`status=nonfinite_start`; it is not optimized, but remains in every level's
denominator. This preserves the known complex case 62 rather than silently
making the evaluated cohort smaller.

## Outputs

- `manifest.json`: input/source/cohort hashes, Git identity, source and loaded
  native-binary hashes, dependency versions, and all scientific settings.
- `cases/<case>/<target>.json`: the four paired raw records.
- `results.csv`: one flat row per case, target, and level.
- `summary.json`: target-level and global timing, convergence, geometry, and
  paired-savings aggregates.
- `report.md`: concise human-readable tables.

`geometry_passed` is the result of Hotpot's standard structure-acceptance gate.
It is deliberately independent of optimizer convergence. The summary also
reports paired geometry regressions and improvements against `STRICT`; energy
deltas compare the selected minimum-energy frames that are returned to users.
