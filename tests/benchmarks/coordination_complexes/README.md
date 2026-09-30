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

`--render off` is the default and keeps rendering independent of scientific
execution. `--render auto` uses PyMOL when installed. `--render required`
turns missing PyMOL into an explicit error. Install the project with its
`pymol` extra before using the required mode.

Only the `hotpot` backend implements the complete CBond-to-force-field
pipeline above. The separate backend comparison described below deliberately
holds CBond output fixed; it does not claim that RDKit or Open Babel provides
CBond inference.

## Fixed-topology backend comparison

`backend_comparison.py` compares only 3D construction and native force-field
optimization. It exports the 178 CBond-successful cases from one completed
Hotpot run as a coordinate-free contract containing the original ligand
SMILES, donor indices, and expected atom/bond counts and topology hash. The
runner reconstructs and verifies the selected explicit-hydrogen topology;
coordinates are not retained. RDKit uses its native dative representation for
metal bonds; Hotpot and Open Babel use single bonds. RDKit and Open Babel
independently generate all coordinates, and the same Hotpot `standard`
geometry gate is used for all three reported workflows.

Run RDKit and native Open Babel with 16 workers, retaining the existing
Hotpot result as the frozen reference:

```bash
$ python -m tests.benchmarks.coordination_complexes.backend_comparison \
  --reference movie/benchmarks/extractants_eu_187_59e5741_20260930 \
  --output movie/benchmarks/extractants_eu_178_backend_comparison \
  --workers 16
```

Use `--resume` to reuse already completed per-case backend reports. Use
`--backends rdkit` or `--backends openbabel` to run one adapter. Aggregation
still requires reports for every selected backend and verifies that every
backend covers exactly the same canonical cohort.

The native protocols are:

- RDKit: donor-to-Eu dative bonds, relaxed property-cache valence handling
  (`strict=False`), seeded ETKDGv3 with random coordinates and smoothing
  failures allowed, then fully parameterized MMFF when available or full/
  explicitly labelled partial UFF otherwise;
- Open Babel: `OBBuilder`, followed by UFF conjugate gradients for up to
  10,000 steps; its upstream builder does not guarantee deterministic output
  from `OB_RANDOM_SEED`; and
- Hotpot: the frozen result from the reference run, without rerunning its
  chemistry.

Build success, force-field support, complete parameterization, finite
optimized output, backend-reported convergence, topology preservation, and
common-gate acceptance are separate fields. “Finite optimized output” does not
mean that the backend convergence threshold was reached. In particular, a
partial-UFF RDKit result is never reported as fully parameterized. A `standard`
gate pass establishes only the tested numerical, topology, and geometry
invariants; it does not establish experimental-structure accuracy or
metal-force-field validity.

The comparison output is:

```text
<comparison-output>/
├── canonical_cases.json
├── comparison.json
├── comparison.csv
├── case_results.csv
├── comparison.png
└── cases/
    ├── rdkit/NNNN/report.json
    └── openbabel/NNNN/report.json
```

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
