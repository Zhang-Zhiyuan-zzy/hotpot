# Force-field three-stage refactor and validation report

Date: 2026-09-24

Branch: `refactor/forcefield-three-stage`

Planning baseline: `b55ae69`

Validated production commit: `f0a6e7c`

## 1. Result

The three-stage complex construction and optimization workflow is implemented
and validated on the standard 187-molecule Eu--extractant set.

| Metric | Previous baseline | Current | Change |
|---|---:|---:|---:|
| Inputs | 187 | 187 | unchanged |
| CBond success | 178 | 178 | unchanged |
| Quality pass | 171 | 171 | unchanged |
| Quality failure | 7 | 7 | unchanged |
| CBond failure | 9 | 9 | unchanged |
| Validation wall time | 380.590 s | 157.776 s | -58.54%; 2.41x faster |
| Aggregate case time | 5669.518 s | 2365.233 s | -58.28%; 2.40x faster |
| Median case time | 20.699 s | 11.672 s | -43.61% |
| P95 case time | 81.496 s | 23.587 s | -71.06% |
| Maximum case time | 207.612 s | 71.749 s | -65.44% |

All 187 status assignments are identical to the previous baseline. No case
changed from pass to fail or from fail to pass.

The complete validation artifacts are stored in:

```text
movie/extractants_eu_three_stage_refactor_16c_20260924/
├── cases/                         # 187 per-case directories
│   └── NNNN/
│       ├── report.json
│       ├── trajectory/            # lossless trajectory archive, when CBond succeeds
│       ├── optimized.mol2
│       ├── optimized.sdf
│       └── final.png
├── optimized_all.sdf
├── optimized_passed.sdf
├── final_png/                     # 12 contact sheets
├── summary.json
├── integrity.json
├── trajectory_integrity.json
├── baseline_comparison.json
└── analysis_report.md
```

## 2. Implemented workflow

### 2.1 Stage 1: ligand construction

Each non-metal connected component calls `OBBuilder` once. Candidate retries
start from a copy of that embedded structure and use perturbation plus force-
field optimization. They do not call `OBBuilder` again.

Each candidate uses `ligand_skeleton` topology checkpoints. A confirmed
ring--bond piercing initializes a fixed watch set. Repair attempts repeat:

```text
open one eligible ring edge
→ perturb
→ short optimization
→ restore the exact edge and metadata
→ rescan only the watched ring--bond pairs
```

When the watch set clears, the workflow returns to one full AABB-screened
checkpoint. If the repair budget is exhausted, it restores the best closed-
topology frame, performs exactly one final checkpoint scan, warns, and returns.
It no longer performs a second settling optimization that can overwrite that
fallback frame.

A ligand candidate with a remaining confirmed piercing cannot pass the basic
gate. `UNDETERMINED` is recorded and warned about, but does not trigger an
automatic ring-opening operation.

### 2.2 Stage 2: coordination-bond construction

All intended metal--ligand bonds are initially hidden. For each candidate bond,
the code prepares the current `full_graph` Relevant Cycles before adding that
candidate. It then checks only the hypothetical finite metal--donor segment
against those rings.

This boundary has two useful properties:

- existing organic and metal--organic chelate rings are visible;
- the candidate's own newly closed chelate ring does not yet exist, so it
  cannot create a self-closure false positive.

The first candidate without a confirmed piercing is restored and relaxed.
Coordinates or topology changes invalidate the workspace; the next round
prepares a new one. Stage 2 does not perform a terminal whole-molecule scan;
Stage 3 owns that checkpoint.

Metal relocation is conditional. It is permitted only while the complete
system has zero accepted coordination bonds and every pending path for the
selected unbound center is confirmed blocked. Ligands remain fixed. Once any
coordination bond is accepted, remaining blocked candidates use the bounded
relaxation path. An infeasible relocation that changes no coordinates proceeds
directly to the bounded fallback instead of rebuilding and rescanning the same
workspace.

### 2.3 Stage 3: full-complex optimization

Stage 3 performs a `full_graph` checkpoint at entry. Confirmed piercings enter
the same fixed-watch repair workflow. Once topology is clear, the main Open
Babel optimizer runs as a purely numerical loop; its epochs do not execute
bond--ring scans. A final `full_graph` checkpoint validates the result. If this
checkpoint finds a confirmed piercing, the workflow re-enters repair and then
stabilizes a successfully changed structure with a short numerical segment.

Numerical evidence has explicit coordinate ownership. Energy, gradient,
`best_epoch`, and convergence values are reused only while the current
coordinates equal the last optimizer-selected frame. If repair exhausts its
budget after changing coordinates, the returned report retains historical
epoch and step counts but marks current-frame numerical values unknown
(`NaN`, `best_epoch=-1`, `termination_reason=topology_blocked`). This prevents
an optimized report from describing a different coordinate frame.

### 2.4 Geometry execution

The geometry package supplies facts only. Forcefields decides whether those
facts permit bond restoration, require repair, produce a warning, or fail
acceptance.

The implementation uses:

- immutable cycle topology templates;
- one frame workspace for all candidates sharing coordinates and topology;
- a strict AABB broad phase;
- prepared per-cycle numeric kernels;
- scalar exact segment--cycle classification only for AABB-overlapping pairs;
- three states: `PIERCES`, `DOES_NOT_PIERCE`, and `UNDETERMINED`.

Only Relevant Cycles of at most 16 atoms are actionable in forcefield repair.
Larger cycles are counted and warned about.

## 3. Production commits

| Commit | Purpose |
|---|---|
| `1f98f68` | Define the three-stage workflow and validation contract. |
| `59d055a` | Add stage scan-boundary characterization tests. |
| `0d38595` | Add the scalar prepared-cycle seam. |
| `489d7d2` | Add checkpoint-based acceptance. |
| `a5954d2` | Remove topology work from numerical epochs. |
| `0b4e4f0` | Align numerical frame report semantics. |
| `b995b07` | Reuse checkpoint evidence. |
| `42dd4a4` | Screen hidden coordination candidates. |
| `07e3db5` | Enforce zero-piercing ligand acceptance. |
| `17be3ac` | Cache immutable cycle topology. |
| `113c3ce` | Add bounded metal-position search. |
| `b2942ee` | Gate complex optimization by full-graph topology. |
| `fbd0922` | Connect relocation to the blocked unbound-metal path. |
| `16beffb` | Precompute single-frame cycle kernels. |
| `aea851f` | Add reusable bond--ring workspaces. |
| `33906d9` | Add opt-in stability stopping; default remains disabled. |
| `e80a226` | Reuse the Stage 2 screening workspace. |
| `0ed5d10` | Align the Relevant Cycle integration fence. |
| `eb66b25` | Isolate fixed-watch ring repair. |
| `6ddc6a6` | Serialize complete topology-checkpoint evidence. |
| `5e75b2a` | Require globally zero accepted bonds before metal relocation. |
| `3f5fe7b` | Preserve relaxation after one metal becomes anchored. |
| `d220a09` | Stop repair cleanly when its budget is exhausted. |
| `d574d5c` | Align returned numerical reports with repaired coordinates. |
| `0336029` | Align the scan-count regression fence. |
| `f0a6e7c` | Skip unchanged rescans after infeasible relocation. |

## 4. Automated verification

The final focused regression suite passed **634 tests** under Python 3.11.16.
It covers geometry, prepared-cycle equivalence, public forcefield APIs,
acceptance, optimizer behavior, trajectory serialization, metal relocation,
three-stage scan boundaries, complex construction, and Relevant Cycle
integration.

An attempt to collect all of `tests/test_cheminfo` stopped at
`test_elements.py` because the active validation environment does not contain
the optional `numba` package. This is an environment collection limitation;
the affected three-stage modules and their integration tests completed.

The final one-molecule smoke run passed before the full run. Its v4 trajectory
archive, MOL2, SDF, and PNG were read back successfully; the MOL2-versus-archive
maximum coordinate difference was \(4.98\times10^{-5}\) Å.

## 5. Full 187-molecule validation

### 5.1 Artifact integrity

The audit reports `passed=true`:

| Artifact check | Result |
|---|---:|
| Input reports | 187 / 187 |
| CBond-complete cases | 178 |
| Readable trajectory archives | 178 / 178 |
| Optimized MOL2 | 178 / 178 |
| Optimized SDF | 178 / 178 |
| Final PNG | 178 / 178 |
| Main trajectory frames | 8898 |
| Ligand-build branches | 284 |
| Ligand-build frames | 2330 |
| Global/case integrity issues | 0 / 0 |
| Maximum MOL2/archive coordinate difference | 4.9996e-5 Å |

The nine CBond failures do not enter forcefields and therefore correctly have
no optimization trajectory or final structure.

### 5.2 Topology and screening evidence

| Measurement | Count |
|---|---:|
| Total topology checkpoints | 1150 |
| Stage 1 `ligand_skeleton` checkpoints | 787 |
| Stage 3 `full_graph` checkpoints | 363 |
| Selected rings across checkpoints | 5987 |
| Excluded rings larger than 16 | 53 |
| Candidate bond--ring pairs | 406103 |
| AABB-separated pairs | 363777 (89.58%) |
| Exact-kernel pairs | 42326 (10.42%) |
| Confirmed piercing observations | 80 |
| Mathematically undetermined observations | 6709 |

Confirmed piercing evidence occurred at intermediate checkpoints in 31 cases.
Every one of those cases finished with zero confirmed piercing. The 6709
undetermined observations are geometric ambiguity records and are not counted
as confirmed interpenetration.

Stage 2 recorded 703 bond trials: 689 accepted and 14 rejected. All intended
bonds were eventually restored without a forced bond. No metal relocation was
needed by this mononuclear Eu dataset; relocation behavior is covered by unit
tests, including the multi-metal boundary.

`ComplexBuildDiagnostics.attempt_count=284` is the number of ligand candidate
attempts, not builder calls. The trajectory structure implies 180 OBBuilder
calls: 176 cases have one non-metal component, while cases 0059 and 0060 have
two. This agrees with the one-builder-call-per-component implementation.

### 5.3 Quality failures

None of the seven failed-quality structures contains a confirmed final
ring--bond piercing.

| Case | Failure | Interpretation |
|---:|---|---|
| 0031 | 2 close contacts, 5 short bonds, 15 bond-ratio failures | Severe local collapse after the 100-epoch budget. |
| 0045 | NaN gradients, close contact, 2 short bonds, 7 bond-ratio failures | Local collapse plus invalid gradient; final ring relations include mathematical uncertainty. |
| 0046 | NaN RMS/max gradient | Finite energy but invalid gradient; final ring relations include mathematical uncertainty. |
| 0047 | NaN gradients, close contact, 2 short bonds, 8 bond-ratio failures | Local collapse plus invalid gradient; final ring relations include mathematical uncertainty. |
| 0054 | NaN RMS/max gradient | Backend reports convergence, but the quality gate correctly rejects the invalid gradient. |
| 0061 | Two short Eu--donor bonds and two ratio failures | Distances 1.5193/1.5225 Å are below the 1.7485 Å lower threshold. |
| 0070 | One atom overlap | Hydrogen atoms 120 and 121 occupy the same coordinates after the 100-epoch budget. |

The aggregate failure signature is exactly the same as the baseline: 32 bond
ratio failures, 11 short bonds, four close contacts, four non-finite RMS
gradients, four non-finite maximum gradients, and one atom overlap.

### 5.4 CBond failures

Cases 0022, 0136, 0139, 0141, 0182, 0185, 0186, and 0187 have no candidate
above the raw score threshold -0.125. Case 0134 contains multiple sodium ions,
while CBond inference supports exactly one metal center. These failures occur
before forcefield construction.

### 5.5 Long-tail cases

The current within-run P95 threshold is 23.587 s.

| Case | Total (s) | Build (s) | Main cause |
|---:|---:|---:|---|
| 0070 | 71.749 | 29.235 | 143 atoms, 100 epochs, final H--H overlap. |
| 0089 | 55.283 | 34.200 | 154-atom ligand build dominates. |
| 0048 | 36.650 | 26.347 | Two ligand candidates; build dominates. |
| 0010 | 31.999 | 21.859 | 175 atoms; build and CBond are both relatively expensive. |
| 0062 | 30.459 | 13.363 | 100 global epochs and 131 main frames. |
| 0045 | 27.267 | 9.716 | 100 epochs, collapsed geometry, NaN gradient. |
| 0061 | 26.991 | 10.098 | 82 epochs; final Eu--donor bonds too short. |
| 0047 | 26.501 | 9.330 | 100 epochs, collapsed geometry, NaN gradient. |
| 0021 | 23.900 | 9.231 | 100 global epochs and 126 main frames. |
| 0140 | 23.711 | 14.746 | Two ligand candidates; build dominates. |

Across these ten cases, CBond consumes only 0.94% of total time. Six are
dominated by global optimization and four by ligand construction.

## 6. Coordinate comparison with the previous baseline

All 178 forcefield cases have identical atom order and topology and can be
compared after Kabsch alignment. The other nine are the CBond failures and have
no trajectory in either run.

| RMSD statistic | Value (Å) |
|---|---:|
| Median | 1.99e-15 |
| P90 | 1.92e-14 |
| P95 | 0.0350 |
| P99 | 0.5916 |
| Maximum | 1.3811 |

Fifteen cases have RMSD greater than 0.001 Å. All retain the same pass/fail
status as the baseline. The changes arise from altered terminal or selected
optimizer frames after the workflow correction; they do not arise from atom
reordering or topology mismatch.

## 7. Isolated performance evidence

The following measurements isolate the four requested optimizations. They are
microbenchmarks or direct call-count measurements and must not be multiplied
together to predict end-to-end speed.

| Optimization | Measurement | Result |
|---|---|---:|
| Strict AABB broad phase | Real cases 0001 and 0048; dense versus sparse scan | 2.59x and 4.96x faster; confirmed piercing counts identical |
| Frozen watch set | 101 full candidate pairs versus one watched pair, 200 repeats | 6.95x faster |
| Stage 2 shared workspace | 12 blocked candidates, 30 repeats | 0.562 s to 0.334 s; 1.68x faster |
| One OBBuilder call | Case 0010 ligand, 174 atoms with H, 20 calls | 0.04095 s/call; 49 avoided calls save about 2.01 s in a 50-attempt worst case |

The immutable topology-template and frame-kernel implementation was also
measured on real cases 0001, 0048, and 0070: 30.210 s decreased to 21.840 s
(1.38x), with identical state, count, finding-key, and evidence output.

## 8. Remaining limits

1. Exact segment--cycle classification still iterates over the AABB-surviving
   pairs. A batched exact kernel may improve performance, but it is not needed
   for the present correctness target and must preserve all three-state and
   tolerance semantics.
2. The standard 187 set did not trigger metal relocation. The branch is
   covered by focused single- and multi-metal tests, but needs a dedicated
   real-complex corpus for empirical success-rate measurement.
3. `UNDETERMINED` remains a warning. Forcefields repairs only mathematically
   confirmed `PIERCES` relations, as specified.
4. Trajectory schema is version 4. Older version-3 archives are rejected
   explicitly; this repository currently has no archive-compatibility
   requirement.
5. `converged` describes the selected frame, whereas `terminal_converged` and
   `termination_reason=converged` describe the terminal optimizer frame. Cases
   0004 and 0064 therefore legitimately have a converged terminal frame while
   their selected lower-energy frame is not marked converged.
