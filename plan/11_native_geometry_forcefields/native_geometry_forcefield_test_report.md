# Native Geometry and Force-Field Refactor: Test Report

## 1. Executive result

The final validation target was clean commit
`59e5741b9cac26b8dc787edd481a40ba6ad41d38`. Its last production-code
change is `2518a0c9e3624e2705cfcd9459461d835464b355`; `59e5741` adds only the
native API guide.

The tested result is:

- the native extensions build and the complete inference-compatibility test
  selection passes on Python 3.9 through 3.14;
- the final 187-case Eu-complex benchmark passes 177 cases, has one declared
  geometry-quality failure, and retains the same nine upstream CBond failures;
- force-field success after successful CBond construction is 177/178
  (99.4382%), compared with 175/178 (98.3146%) at the pre-refactor baseline;
- complete-invocation wall time is 125.233 s, 10.753 s (7.91%) below the
  135.986 s baseline;
- native segment--cycle geometry is between 24.5x and 1085.9x faster at the
  median than the historical Python measurements, depending on workload;
- the final convergence fix deliberately performs more Stage 3 work than the
  faster pre-fix intermediate build. It removes false convergence rather than
  relaxing any quality threshold;
- cases 54 and 109 change from quality failure to pass, case 125 is restored
  from the pre-fix regression to pass, and case 61 remains an explicit quality
  failure.

No failed structure is reclassified as successful merely because a terminal
frame exists. Failed CBond cases have placeholders, while every CBond-success
case has an optimized output and a trajectory archive.

A separate deep artifact audit passed. It strictly parsed all 733 JSON files,
reopened all 178 trajectory archives through the production reader, validated
all 20,291 main frames and 186 ligand-build branches, and compared each
exported structure with its authoritative selected trajectory frame. The full
result and its limits are recorded in Section 8.3.

## 2. Test identity and environment

| Item | Value |
|---|---|
| Date / timezone | 2026-09-30 / Asia/Shanghai |
| Host | `chemlex-ai-2` |
| OS / kernel | Linux x86-64, Ubuntu kernel `5.15.0-139-generic` |
| CPU | AMD Ryzen Threadripper PRO 3995WX |
| CPU topology | 64 physical cores / 128 logical CPUs |
| CPU affinity | logical CPUs 0--127 |
| CPU frequency governor | `ondemand` |
| RAM / swap | 503 GiB / no swap |
| Compiler | `g++ 9.4.0` |
| Primary benchmark Python | CPython 3.11.16 |
| NumPy | 1.26.4 |
| Open Babel | 3.2.1 |
| ONNX Runtime | 1.30.0; CPU and Azure execution providers available |
| Final manifest commit | `59e5741b9cac26b8dc787edd481a40ba6ad41d38` (clean) |
| Final production-code commit | `2518a0c9e3624e2705cfcd9459461d835464b355` |
| Input | `molecules/extractant/extractants.smi` |
| Input SHA-256 | `77ef4913363bb10f0d134d6150c4f9e7214ddfaf03f8e6fc23f57308d5d3443f` |

The three end-to-end benchmark manifests use the same 187 inputs, `standard`
profile, seed `20260921`, 16 spawned workers, CBond threshold `-0.125`, 100
epochs, 100 steps per epoch, and the same build/untangling/restoration limits.
This controls scientific settings across the three benchmark directories.

## 3. Reproducible commands

### 3.1 Python 3.9--3.14 build and test matrix

```bash
UV_CACHE_DIR=/tmp/hotpot-uv-cache \
  bash tests/run_inference_compatibility.sh 3.9 3.10 3.11 3.12 3.13 3.14 \
  > /tmp/hotpot_final_compat_matrix.log 2>&1
```

The compatibility runner rebuilds all native extensions for each interpreter,
runs its complete cross-version test selection, and then runs the dedicated
SMARTS suite. It is broad but is not an invocation of every repository test.

### 3.2 Public geometry boundary profile

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python \
  -m tests.performance.profile_geometry_boundary \
  --repeats 20 \
  --output /tmp/59e5741.geometry-profile.json
```

### 3.3 Segment--cycle benchmark

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python \
  -m tests.performance.test_segment_cycle_relation_benchmark \
  --repeats 20 \
  --output /tmp/59e5741.segment-cycle.json
```

### 3.4 Trajectory-retention/RSS profile

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python \
  -m tests.performance.profile_forcefield_retention \
  --cases 1,54,61,109,125 \
  --repeats 3 \
  --profile standard \
  --output /tmp/59e5741.retention-rss.json
```

Each retention setting is run in a fresh worker process. RSS therefore covers
the complete process, while its wall time covers `ff.complexes_build` only.

### 3.5 Final 187-case benchmark

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python \
  -m tests.benchmarks.coordination_complexes \
  --suite extractants-eu-187 \
  --profile standard \
  --workers 16 \
  --render required \
  --render-workers 16 \
  --output movie/benchmarks/extractants_eu_187_59e5741_20260930
```

This was a fresh run without `--resume`; its wall time therefore covers the
complete validation invocation.

## 4. Python 3.9--3.14 compatibility matrix

Source: `/tmp/hotpot_final_compat_matrix.log`.

| Python | Open Babel | Native build | Compatibility suite | Dedicated SMARTS suite | Result |
|---:|---:|---|---|---|---|
| 3.9 | 3.1.0 | passed | 1189 passed, 4 skipped, 3 xfailed; 34 warnings; 235.99 s | 255 passed; 2.15 s | pass |
| 3.10 | 3.2.1 | passed | 1189 passed, 4 skipped, 3 xfailed; 5 warnings; 49 subtests; 223.94 s | 255 passed; 2.22 s | pass |
| 3.11 | 3.2.1 | passed | 1189 passed, 4 skipped, 3 xfailed; 5 warnings; 49 subtests; 224.66 s | 255 passed; 2.16 s | pass |
| 3.12 | 3.2.1 | passed | 1189 passed, 4 skipped, 3 xfailed; 44 warnings; 49 subtests; 220.10 s | 255 passed; 1.89 s | pass |
| 3.13 | 3.2.1 | passed | 1189 passed, 4 skipped, 3 xfailed; 44 warnings; 49 subtests; 219.57 s | 255 passed; 1.84 s | pass |
| 3.14 | 3.2.1 | passed | 1189 passed, 4 skipped, 3 xfailed; 44 warnings; 49 subtests; 221.13 s | 255 passed; 1.82 s | pass |

The observed interpreter builds were CPython 3.9.25, 3.10.20, 3.11.16,
3.12.13, 3.13.15, and 3.14.7. The higher warning count on 3.12--3.14 includes
runtime deprecation diagnostics and is not a test failure. The three xfails
and four skips are declared test outcomes, not unexpected failures. The log
contains no `FAILED` or test `ERROR` result.

## 5. Geometry correctness and performance

### 5.1 Public-boundary characterization

The final native module was loaded from
`hotpot.cheminfo.geometry._geometry_native`. Twenty repetitions of the frozen
public characterization workload produced:

| Metric | Phase 0 Python baseline | Final | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Minimum | 52.981 ms | 2.603 ms | -50.378 ms | -95.09% |
| Median | 54.364 ms | 2.639 ms | -51.724 ms | -95.14% (20.60x) |
| P95 | 54.581 ms | 3.025 ms | -51.555 ms | -94.46% (18.04x) |
| Maximum | 54.605 ms | 3.148 ms | -51.456 ms | -94.23% |
| Total profiled calls | 36,568 | 3,672 | -32,896 | -89.96% |
| Python geometry calls | 9,265 | 1,169 | -8,096 | -87.38% |
| Native geometry calls | 0 | 36 | +36 | expected cut-over |
| Peak Python traced memory | 110,141 bytes | 199,823 bytes | +89,682 bytes | +81.42% |

The call counts are dispatch/profile observations, not counts of primitive
operations performed inside C++. Moving a large geometric kernel behind one
native entry naturally decreases visible Python calls while the native call
count remains small. The roughly 88 KiB increase in Python-traced peak memory
does not measure native allocations or process RSS; it is reported rather than
hidden and does not negate the 20.60x median workload speedup.

### 5.2 Segment--cycle microbenchmarks

The historical values are the initial Python implementation recorded in
`tests/performance/README.md`. Final values come from
`/tmp/59e5741.segment-cycle.json`.

| Workload | Historical median | Final median | Absolute decrease | Relative decrease | Median speedup | Final P95 |
|---|---:|---:|---:|---:|---:|---:|
| Planar 6-member pair | 1.334 ms | 0.0403 ms | 1.2937 ms | 96.98% | 33.10x | 0.0501 ms |
| Planar 8-member pair | 1.698 ms | 0.0405 ms | 1.6575 ms | 97.61% | 41.88x | 0.0443 ms |
| Nonplanar 6-member pair | 21.183 ms | 0.0636 ms | 21.1194 ms | 99.70% | 333.06x | 0.0706 ms |
| Nonplanar 8-member pair | 312.748 ms | 0.2880 ms | 312.4600 ms | 99.91% | 1085.94x | 0.3103 ms |
| Lazy frame gate | 8.027 ms | 0.2812 ms | 7.7458 ms | 96.50% | 28.54x | 0.3214 ms |
| Dense 16-pair scan | 16.186 ms | 0.6603 ms | 15.5257 ms | 95.92% | 24.51x | 0.7002 ms |

Correctness evidence in the same result remains tri-state: the curated four
relations contain one `PIERCES`, two `DOES_NOT_PIERCE`, and one
`UNDETERMINED` result. The dense 16-pair scan returns 15 non-piercing and one
piercing relation and completes the scan. The nonplanar six- and eight-member
fixtures remain `UNDETERMINED` where surface construction cannot prove a
unique relation; the performance change did not coerce uncertainty into a
binary answer.

The historical benchmark used CPython 3.11.15 and NumPy 2.3.5, whereas the
final run used CPython 3.11.16 and NumPy 1.26.4. The large differences are
consistent with the implemented native cut-over, but this environment mismatch
means the quoted speedups are descriptive rather than a perfectly controlled
same-environment A/B experiment.

## 6. 187-case scientific outcome

### 6.1 Compared revisions

| Label | Commit | Purpose | Artifact directory |
|---|---|---|---|
| Baseline | `1e14f28671b4b1cb22ca3e7d1d76e8fe5b2f769d` | clean pre-native-workflow comparison | `movie/benchmarks/extractants_eu_187_1e14f28_20260929` |
| Pre-fix intermediate | `0219e640bc9be2334eb49bcff9d2022400372ca0` | native workflow before global-gradient convergence verification | `movie/benchmarks/extractants_eu_187_0219e64_20260930` |
| Final | `59e5741b9cac26b8dc787edd481a40ba6ad41d38` | final manifest; production code at ancestor `2518a0c` | `movie/benchmarks/extractants_eu_187_59e5741_20260930` |

### 6.2 Quality outcomes

| Metric | Baseline | Pre-fix | Final | Final minus baseline |
|---|---:|---:|---:|---:|
| Passed | 175 | 176 | 177 | +2 cases |
| Quality failures | 3 | 2 | 1 | -2 cases |
| CBond failures | 9 | 9 | 9 | no change |
| Overall success | 93.5829% | 94.1176% | 94.6524% | +1.0695 percentage points (+1.14% relative) |
| Success after CBond | 98.3146% | 98.8764% | 99.4382% | +1.1236 percentage points (+1.14% relative) |
| Complete trajectory archives | 178 | 178 | 178 | no change |

The unchanged nine CBond failures are outside force-field optimization: eight
have no inferred coordination bond above `-0.125`, and case 134 violates the
single-metal input contract. Among the 178 structures entering force-field
processing, final quality acceptance is 177/178.

The exact baseline-to-final status diff contains only two rows: cases 54 and
109 change from `failed_quality` to `passed`. There are no newly failing cases.
The pre-fix-to-final diff contains only case 125, which changes from
`failed_quality` back to `passed`.

### 6.3 Performance: final versus baseline

Negative time deltas are improvements. Aggregate times sum work across all
cases and workers; wall time measures the complete 16-worker invocation.

| Metric | Baseline | Final | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Wall time | 135.986 s | 125.233 s | -10.753 s | -7.91% |
| Aggregate case time | 2060.755 s | 1849.083 s | -211.672 s | -10.27% |
| Aggregate CBond time | 23.868 s | 23.583 s | -0.285 s | -1.20% |
| Aggregate force-field time | 2028.691 s | 1817.256 s | -211.435 s | -10.42% |
| Median case time | 10.058 s | 9.083 s | -0.975 s | -9.69% |
| P95 case time | 20.606 s | 20.071 s | -0.535 s | -2.60% |
| Maximum case time | 69.295 s | 42.414 s | -26.881 s | -38.79% |
| Throughput | 1.375 cases/s | 1.493 cases/s | +0.118 cases/s | +8.59% |

The baseline reporter predates per-stage timing, so it cannot support a valid
Stage 1/2/3 delta. The final totals are Stage 1 ligand build 543.905 s, Stage 2
coordination restoration 59.167 s, and Stage 3 complex optimization 958.042 s.
These stage values overlap other force-field bookkeeping and must not be added
to infer a new wall time.

### 6.4 Why the final build is slower than the pre-fix intermediate

The pre-fix intermediate had a 95.751 s wall time and a 496.684 s aggregate
Stage 3 time. The final build has a 125.233 s wall time and a 958.042 s Stage 3
time: respectively +29.482 s (+30.79%) and +461.358 s (+92.89%). This is an
intentional correctness cost.

Open Babel's stop indication was previously accepted without checking the
global maximum atom gradient. The final engine verifies that the maximum atom
gradient meets Open Babel's intended 0.1 threshold (converted into the active
energy unit) and continues within the original epoch/step budget when it does
not. Consequently:

- only 16 final cases are truthfully marked converged, versus 159 pre-fix;
- 162 finite cases are labeled budget-exhausted rather than falsely converged;
  161 pass the quality gate and case 61 remains the one quality failure;
- total main trajectory frames rise from 9,142 to 20,291;
- case 125 receives the optimization work needed to clear its bond-length
  quality failure.

The final result is still faster than baseline while enforcing stricter and
more truthful convergence semantics. Comparing only against the unusually fast
pre-fix build would conceal that semantic difference.

## 7. Focus cases 54, 61, 109, and 125

| Case | Baseline | Pre-fix | Final | Final interpretation |
|---:|---|---|---|---|
| 54 | failed quality | passed | passed | Native metal placement/optimization clears the baseline Eu/local-contact failure; all final hard checks pass. It uses the full 100-epoch budget and is correctly reported non-converged rather than rejected. |
| 61 | failed quality | failed quality | failed quality | No threshold was relaxed. Two Eu coordination bonds remain too short: 1.5073 and 1.5012 A versus 1.7485 A limits; ratios 0.5603 and 0.5581 are below 0.65. The terminal finite frame and full trajectory are retained for diagnosis. |
| 109 | failed quality | passed | passed | The baseline 0.3515 A Eu--N collapse and other close-contact/bond defects are absent. The final structure passes all hard checks and reaches verified convergence in 33 epochs (maximum gradient 0.09284). |
| 125 | passed | failed quality | passed | The pre-fix build stopped at a 1.7277 A bond against a 1.7485 A limit. Global-gradient verification prevents that premature stop; the final run uses the full budget and passes at 1508.428 kJ/mol. |

Final per-case force-field times are 4.228 s (54), 20.369 s (61), 3.121 s
(109), and 9.037 s (125). Case 61 is the only post-CBond quality failure and
is therefore visible rather than silently replaced by an unvalidated frame.

## 8. Trajectory retention and memory

### 8.1 Final benchmark artifacts

The final run records 178 trajectory archives, one for every CBond-successful
case. It contains 20,291 main-trajectory frames (median 120, maximum 132), 178
optimized MOL2/SDF structures, and 187 per-case `final.png` files. PyMOL
rendered 178 molecular structures; the nine CBond failures have explicit
placeholder images. Every rendered structure reports exactly one metal atom,
and its hydrogen count is nonzero. The all-case contact sheet is present. The
directory uses 325,688,121 bytes on disk at the time of inspection.

The generated `integrity.json` reports:

- 187 expected and 187 observed reports;
- no missing or unexpected case indices;
- 178 CBond successes, validation reports, trajectory archives, and optimized
  structures;
- no missing archive or optimized-output index;
- status counts consistent with the per-case reports.

### 8.2 In-memory retention profile

Five representative cases (1, 54, 61, 109, 125), three fresh-process repeats,
and both `save_movie` modes were measured. Median values across the 15 samples
per mode are:

| Mode | Frames | Retained coordinate bytes | Peak RSS | FF wall time |
|---|---:|---:|---:|---:|
| `save_movie=False` | 26 | 14,400 | 270,296 KiB | 3.7554 s |
| `save_movie=True` | 122 | 294,240 | 270,380 KiB | 3.7565 s |
| Difference | +96 | +279,840 bytes | +84 KiB | +0.0011 s |

Across paired runs, the median RSS delta is +396 KiB (+0.147%) and the median
wall-time delta is +0.0197 s (+0.424%). Paired RSS deltas range from -3,836 to
+2,552 KiB, which is larger than and changes sign around the measured median.
Therefore the process-level RSS effect is below this experiment's noise floor;
the exact retained-coordinate accounting is the reliable memory measure.
Retaining complete frames raises coordinate payload in proportion to molecule
size and trajectory length (median paired increase 279,840 bytes; range 55,680
to 482,664 bytes) but adds no material force-field runtime in this sample.

Disk serialization is not included in this RSS profile because
`trajectory_path` is `null`. The 187-case benchmark is the evidence for actual
on-disk archive generation.

### 8.3 Independent deep artifact audit

The audit did more than trust the generated `integrity.json`:

- all 733 JSON files were parsed with non-standard `NaN` and `Infinity`
  rejected, followed by a recursive finite-number check;
- all 187 reports and 187 CSV rows were semantically consistent; the audit
  independently recomputed status counts, success rates, failure identities,
  aggregate timings and distribution statistics and matched the stored
  summary, integrity and human-report claims;
- all 178 archives reopened through `ForceFieldTrajectoryArchive.read()`;
  frame indices, coordinate and topology revisions, selected/terminal indices,
  coordinate shapes, finite coordinates and energies, bond orders and bond
  index ranges were valid;
- each main and ligand-build `trajectory.sdf` had exactly one record per
  corresponding frame;
- all 178 MOL2 outputs matched the authoritative selected frame in atom-symbol
  order, undirected bond set, and coordinates within the 5.1e-5 Angstrom text
  serialization tolerance;
- every output contained exactly one Eu atom, whose neighbors exactly matched
  the donor indices stored in its CBond report;
- `optimized_all.sdf` contains 178 records and `optimized_passed.sdf` 177;
- all 187 case PNGs and the 4480 x 11020 root contact sheet are valid images;
  the split is 178 molecular renders and nine explicit CBond-failure
  placeholders;
- no staging, backup, temporary file or symbolic-link residue remains.

This proves internal artifact, topology and selected-frame consistency. It
does not independently reconstruct each hydrogenated/Kekule target topology
from the original SMILES. Cases 54, 61, 109 and 125 received a visual spot
check for gross rendering/topology defects; the other PNGs were validated as
images and by metadata rather than reviewed individually for chemical
plausibility.

## 9. Evidence limitations

1. The baseline, pre-fix, and final 187 measurements are one run each. They use
   identical scientific settings and the same machine, but are not randomized,
   interleaved repetitions; scheduler and frequency variation are not
   quantified. Treat small differences as indicative, not confidence-bounded.
2. Aggregate case and stage times are sums across 16 parallel workers and are
   not directly comparable to wall time. Stage timings were not recorded by
   the baseline schema, so no baseline per-stage speedup is claimed.
3. Historical geometry timings used a slightly different Python/NumPy
   environment and their raw JSON was not retained. The final raw JSON and
   reproducible driver exist, but the historical-to-final speedup is
   descriptive.
4. Per-phase native-refactor ablations were not preserved at every commit.
   End-to-end changes cannot be decomposed honestly into one speedup number per
   implementation phase without rebuilding and testing each boundary under a
   shared contract.
5. `forcefield_converged` counts before and after `2518a0c` do not have the same
   semantics. Earlier counts trusted Open Babel's premature stop; final counts
   require the global maximum atom gradient to pass. They must not be treated
   as a convergence-rate regression.
6. The quality gate establishes structural/numerical acceptance, not agreement
   with experimental crystal structures or quantum-chemical reference
   geometries. No RMSD-to-reference accuracy claim is made.
7. `movie/` and `/tmp` evidence is local and not Git-tracked. The commands and
   manifest hashes make it reproducible, but preservation depends on retaining
   those directories or archiving them elsewhere.
8. The compatibility matrix validates native builds and the repository's
   defined inference-compatibility selection on the six interpreters available
   on this host. It neither invokes every repository test nor constitutes a
   cross-OS, cross-compiler, or cross-architecture matrix.
9. The deep audit proves internal serialization and selected-frame consistency,
   but neither it nor the four-image visual spot check substitutes for
   comparison with experimental structures or full human review of all 187
   structures.
10. A project-global `pytest --collect-only` audit found 1,535 tests but stopped
    with nine collection errors. Unrelated example/plugin tests require the
    optional `torch`, `torch_geometric`, `numba`, `requests`, and
    `scikit-learn` stacks; default pytest import mode also collides on duplicate
    test basenames. This is why the report claims the committed compatibility
    selection, not an unexecuted project-global pass.

## 10. Artifact index

| Evidence | Location |
|---|---|
| Compatibility log | `/tmp/hotpot_final_compat_matrix.log` |
| Public geometry profile | `/tmp/59e5741.geometry-profile.json` |
| Segment--cycle profile | `/tmp/59e5741.segment-cycle.json` |
| Retention/RSS profile | `/tmp/59e5741.retention-rss.json` |
| Artifact-audit output | `/tmp/59e5741.artifact-audit.json` |
| Local artifact-audit driver | `/tmp/final_artifact_audit.py` |
| Project-global collection audit | `/tmp/59e5741.pytest-collect.log` |
| Baseline benchmark | `movie/benchmarks/extractants_eu_187_1e14f28_20260929/` |
| Pre-fix benchmark | `movie/benchmarks/extractants_eu_187_0219e64_20260930/` |
| Final benchmark | `movie/benchmarks/extractants_eu_187_59e5741_20260930/` |
