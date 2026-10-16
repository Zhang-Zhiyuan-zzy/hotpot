# Scientific-state pre-activation resolver baseline

This opt-in benchmark measures the exact planned
`COORDINATION_SCREENING_V1` provider/rule manifest without activating it. It
does not change the profile artifact, bypass static eligibility, or claim an
active-profile automatic-selection rate.

The benchmark writes:

- `manifest.json`: corpus and inactive profile identities;
- `records.jsonl`: source-bound per-record facts, timings, search metrics,
  decisions and replay evidence; and
- `summary.json`: throughput, Python allocation peak, expanded-path,
  truncation, decision, profile-boundary and replay aggregates.

Run a plain SMILES corpus:

```bash
$ python -m tests.benchmarks.scientific_state \
    --smiles-corpus /path/to/input.smi \
    --output /tmp/hotpot-state-baseline
```

Run the canonical structure registry without copying or rewriting its assets:

```bash
$ python -m tests.benchmarks.scientific_state \
    --structure-registry hotpot/dataset/data/structures/structures.csv \
    --structure-root hotpot/dataset/data/store \
    --output /tmp/hotpot-state-structures
```

Deterministic replay is enabled by default. `--no-replay` measures only the
primary pass. `python_peak_bytes` is measured by `tracemalloc` and excludes
Open Babel and other native allocations.

## Scientific boundary

The profile artifact is currently `planned`. Static eligibility therefore
contains `PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE`, and the production decision
compiler is not executed. `automatic_selection_fraction` must not be cited as
an activation result; `activation_coverage_is_final` is always `false` in this
benchmark.

XYZ records have no bond topology. Existing MOL2 and metal-pair records may
represent metal-ligand bonds as `unknown` rather than `coordination`. The
benchmark reports `PROFILE.COORDINATION_BOUNDARY_REQUIRED` for those records
and never promotes such bonds silently. Final Phase 10 activation validation
requires a reviewed corpus adapter that preserves explicit `DATIVE` semantics,
followed by approval of the scientific profile itself.
