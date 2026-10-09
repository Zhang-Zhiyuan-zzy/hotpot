# Coordination-complex benchmark

This opt-in benchmark measures an end-to-end chemical workflow:

1. read ligand SMILES;
2. predict Eu coordination bonds with Hotpot CBond;
3. construct and optimize the coordination complex with
   `forcefields.complexes_build()`;
4. apply the force-field geometry and numerical quality gate;
5. retain the complete immutable trajectory, including topology revisions;
6. export the workflow-selected structure after a completed run, or the last
   finite frame after a force-field exception;
7. write machine-readable results and a Markdown report;
8. optionally render every case and an experiment contact sheet with PyMOL.

The built-in `extractants-eu-187` suite reads
`molecules/extractant/extractants.smi`. The runner verifies that this corpus
contains exactly 187 records and stores its SHA-256 digest in `manifest.json`.
The manifest also records the Git commit and whether the worktree was dirty.
It is deliberately not a default pytest test.

The recorded CBond policy is intentionally asymmetric: the first bond uses a
raw-score threshold of `-0.5`, while every subsequent bond uses `-0.125`.

## Commands

Run the complete standard benchmark with 16 workers and require all PNG
artifacts:

```bash
python -m tests.benchmarks.coordination_complexes \
  --suite extractants-eu-187 \
  --workers 16 \
  --render required \
  --output movie/benchmarks/extractants_eu_187
```

Run a short plumbing check. The smoke profile uses deliberately reduced force
field effort and its scientific outcomes must not be compared with the
standard profile:

```bash
python -m tests.benchmarks.coordination_complexes \
  --smoke \
  --output /tmp/hotpot-complex-smoke
```

Run the standard settings on the first five molecules:

```bash
python -m tests.benchmarks.coordination_complexes \
  --limit 5 \
  --output /tmp/hotpot-complex-five
```

Reproduce selected corpus cases:

```bash
python -m tests.benchmarks.coordination_complexes \
  --cases 54,61,109 \
  --workers 3 \
  --render auto \
  --output /tmp/hotpot-complex-failures
```

Resume with exactly the same scientific configuration:

```bash
python -m tests.benchmarks.coordination_complexes \
  --workers 16 \
  --resume \
  --output movie/benchmarks/extractants_eu_187
```

An output directory containing a manifest is never silently overwritten; use
`--resume` with the same input digest, selection, profile, and settings, or
choose a new output directory.

Rebuild aggregate tables and reports without rerunning chemistry:

```bash
python -m tests.benchmarks.coordination_complexes \
  --aggregate-only \
  --output movie/benchmarks/extractants_eu_187
```

A custom SMILES corpus uses the same end-to-end Hotpot backend:

```bash
python -m tests.benchmarks.coordination_complexes \
  --input molecules/my_suite.smi \
  --metal Eu \
  --output movie/benchmarks/my_suite
```

## Five independent workflow benchmarks

The comparison uses the same 187 ligands and frozen 181-member Eu–ligand
cohort in five independently launchable workflows. Each workflow evaluates
both the isolated ligand and, when CBond produced a topology, the metal–ligand
complex. All final structures are assessed by the same Hotpot `standard`
geometry gate. Every Hotpot workflow explicitly uses the `FAST` convergence
policy.

| Launcher | Ligand workflow | Eu–ligand workflow |
|---|---|---|
| `rdkit_benchmark` | RDKit ETKDG + MMFF/UFF | RDKit ETKDG + MMFF/UFF |
| `openbabel_benchmark` | Native OBBuilder + UFF | Native OBBuilder + UFF |
| `obwrappers_benchmark` | `obWrappers.build()` + `optimize()` | `obWrappers.build()` + `optimize()` |
| `optimize_complex_benchmark` | `ff.build3d()` + `ff.optimize()` | `ff.build_complex3d()` + `ff.optimize_complex()` |
| `auto_optimize_benchmark` | `ff.build3d()` + `ff.auto_optimize()` | `ff.build_complex3d()` + `ff.auto_optimize()` |

Run each workflow separately with 16 workers:

```bash
$ REFERENCE=movie/benchmarks/extractants_eu_187
$ OUTPUT=movie/benchmarks/five_workflows
$ python -m tests.benchmarks.coordination_complexes.rdkit_benchmark \
  --input molecules/extractant/extractants.smi \
  --reference "$REFERENCE" \
  --output "$OUTPUT/rdkit" --workers 16

$ python -m tests.benchmarks.coordination_complexes.openbabel_benchmark \
  --input molecules/extractant/extractants.smi \
  --cohort "$OUTPUT/rdkit/canonical_cases.json" \
  --output "$OUTPUT/openbabel" --workers 16

$ python -m tests.benchmarks.coordination_complexes.obwrappers_benchmark \
  --input molecules/extractant/extractants.smi \
  --cohort "$OUTPUT/rdkit/canonical_cases.json" \
  --output "$OUTPUT/obwrappers" --workers 16

$ python -m tests.benchmarks.coordination_complexes.optimize_complex_benchmark \
  --input molecules/extractant/extractants.smi \
  --cohort "$OUTPUT/rdkit/canonical_cases.json" \
  --output "$OUTPUT/hotpot_optimize_complex" --workers 16

$ python -m tests.benchmarks.coordination_complexes.auto_optimize_benchmark \
  --input molecules/extractant/extractants.smi \
  --cohort "$OUTPUT/rdkit/canonical_cases.json" \
  --output "$OUTPUT/hotpot_auto" --workers 16
```

Alternatively, replace `--cohort` with `--reference <completed-hotpot-run>` to
export the frozen cohort from an existing standard benchmark. Add `--resume`
to continue an interrupted run whose manifest is unchanged.

Aggregate the five completed workflows without rerunning chemistry:

```bash
$ OUTPUT=movie/benchmarks/five_workflows
$ python -m tests.benchmarks.coordination_complexes.workflow_comparison \
  --rdkit "$OUTPUT/rdkit" \
  --openbabel "$OUTPUT/openbabel" \
  --obwrappers "$OUTPUT/obwrappers" \
  --hotpot-optimize-complex "$OUTPUT/hotpot_optimize_complex" \
  --hotpot-auto "$OUTPUT/hotpot_auto" \
  --output-dir assets/readme
```

The reported efficiency is the median build-plus-optimize time among workflows
that completed both operations. CBond inference, final geometry validation,
serialization, and rendering are outside this timing boundary.

Each workflow root contains `manifest.json`, `summary.json`, and one
`cases/NNNN/report.json` with separate `ligand` and `complex` targets.

## Output contract

```text
<output>/
├── manifest.json
├── results.csv
├── summary.json
├── integrity.json
├── report.md
├── optimized_all.sdf
├── optimized_passed.sdf
├── final.png                       # when rendering is enabled
├── render_report.json
└── cases/
    └── NNNN/
        ├── input.smi
        ├── cbond.smi
        ├── report.json
        ├── optimized.mol2
        ├── optimized.sdf
        ├── final.png               # when rendering succeeds
        └── trajectory/
            ├── archive.json
            ├── main/
            │   ├── trajectory.json
            │   ├── trajectory.sdf
            │   └── coordinates.npz
            └── ligand_build_attempts/
                └── NNNN/                 # when ligand-build branches exist
```

`trajectory/` is the lossless scientific record. A completed quality failure
exports the trajectory's selected frame; an interrupted force-field run exports
its last finite frame. MOL2 and SDF do not preserve Hotpot's dative-bond
semantics in every writer, so an exported visualization frame records
`visualization_topology_lossy` in its case report.
