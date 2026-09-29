# 187-molecule coordination benchmark: 2026-09-29

## Reproduction identity

- Code: `1e14f28671b4b1cb22ca3e7d1d76e8fe5b2f769d`
- Worktree at launch: clean
- Input: `molecules/extractant/extractants.smi`
- Input count: 187
- SHA-256: `77ef4913363bb10f0d134d6150c4f9e7214ddfaf03f8e6fc23f57308d5d3443f`
- Backend: Hotpot CBond on CPU, followed by Hotpot's UFF complex workflow
- Parallelism: 16 spawned worker processes
- Output: `movie/benchmarks/extractants_eu_187_1e14f28_20260929`

```bash
python -m tests.benchmarks.coordination_complexes \
  --suite extractants-eu-187 \
  --workers 16 \
  --render required \
  --output movie/benchmarks/extractants_eu_187_1e14f28_20260929
```

## Results

| Metric | Result |
|---|---:|
| Input cases | 187 |
| CBond completed | 178 |
| Force-field/validation completed | 178 |
| Quality passed | 175 |
| Quality failed | 3 |
| CBond failed | 9 |
| Success among all inputs | 93.5829% |
| Success after CBond | 98.3146% |
| Force-field reported convergence | 158 |
| Complete trajectory archives | 178 |
| Optimized or inspectable structures | 178 |
| Per-case `final.png` files | 187 |
| Molecular PyMOL renders | 178 |
| Explicit failure placeholders | 9 |
| Total main-trajectory frames | 8,545 |
| Wall time | 135.986 s |
| Aggregate case time | 2,060.755 s |
| Median case time | 10.058 s |
| P95 case time | 20.606 s |
| Maximum case time | 69.295 s |
| Throughput | 1.375 cases/s |

The nine CBond failures are cases 22, 134, 136, 139, 141, 182, 185, 186,
and 187. Cases 54, 61, and 109 reach the force-field endpoint but fail the
standard geometry gate. There are no missing case reports, no missing
trajectory archives after successful CBond construction, and no missing
optimized structure for any case with an exportable output frame. Every one
of the 178 molecular renderings contains explicit hydrogen atoms and exactly
one metal atom.

## Comparison with the `2d42ef6` run

| Metric | `2d42ef6` evidence | `1e14f28` run | Change |
|---|---:|---:|---:|
| Passed / quality failed / CBond failed | 175 / 3 / 9 | 175 / 3 / 9 | identical |
| Wall time | 135.842 s | 135.986 s | +0.144 s (+0.106%) |
| Aggregate case time | 2,062.645 s | 2,060.755 s | -1.890 s (-0.092%) |
| Median case time | 10.155 s | 10.058 s | -0.097 s (-0.956%) |
| Maximum case time | 69.633 s | 69.295 s | -0.338 s (-0.485%) |
| Backend convergence reports | 159 | 158 | -1 warning-level result |
| Main trajectory frames | 8,484 | 8,545 | +61 |

All 187 case statuses are identical, including the exact three quality
failures. Cases 54, 61, and 109 reproduce the same energies, distances, frame
counts, and failure checks exactly. The façade cleanup therefore shows no
observable chemical-performance regression.

Nine passing cases followed different numerical paths. Their initial main
trajectory coordinates are exactly equal between runs, but subsequent frame
counts or energies differ. Case 72 was rerun alone from the same initial
coordinates and again followed a third trajectory. Thus the configured seed
does not currently make the Open Babel optimization path bitwise deterministic;
status and quality-gate equivalence are the appropriate regression criteria
until that separate reproducibility issue is isolated.

## Artifact contract

The output root contains `manifest.json`, `results.csv`, `summary.json`,
`integrity.json`, `report.md`, combined SDF files, status-specific SMILES files,
`render_report.json`, and a 4,480 x 11,020 `final.png`. Each successful CBond
case contains its input, connected complex, complete topology-aware trajectory,
exported selected structure, report, and 1,000 x 750 PyMOL image. CBond failures
retain their input and report and receive an explicit placeholder image; they
are never presented as optimized structures.
