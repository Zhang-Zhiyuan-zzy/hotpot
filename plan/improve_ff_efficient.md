# Force-field ring-screening efficiency refactor and 187-molecule validation

Date: 2026-09-23  
Branch: `perf/forcefield-ring-scan`  
Implementation baseline: `5946f62`  
Validated implementation: `fa0ec9b`

## 1. Executive conclusion

The approved performance changes are implemented:

1. Full molecule bond--ring screening now uses a strict AABB broad phase.
2. Ring untangling performs a full scan only to establish or validate a repair
   checkpoint. Between checkpoints it watches only the currently confirmed
   piercing ring--bond pairs.
3. Incremental coordination restoration screens only the newly restored bond,
   with the same AABB broad phase. Previously accepted coordination bonds are
   not rescanned for every trial.
4. Each disconnected ligand component invokes `OBBuilder` exactly once.
   Additional candidate attempts start from the same built coordinates and use
   independent perturbation plus optimization.
5. The failed ring-repair path no longer retrospectively performs a full scan
   for every saved frame. It ranks the current watch batch by its active
   piercing count, fully scans one selected candidate, and compares it with the
   best already validated checkpoint.

On the same 187-input, 16-process Eu--extractant benchmark, wall time decreased
from **2317.87 s (38 min 37.87 s)** to **380.59 s (6 min 20.59 s)**: a
**6.09x speed-up** and **83.58% reduction**. Final quality passes increased from
168 to 171. All 178 CBond-success cases retained readable trajectories and
optimized structures, and all 178 final PNGs were generated.

This is a performance and workflow refactor, not a proof that generated
coordinates are chemically identical to the baseline. Five cases changed
quality status; four improved and one, case 0070, regressed to an atom-overlap
failure. That case is retained with its full trajectory for diagnosis.

## 2. Scope and implementation

### 2.1 Strict AABB broad phase

Relevant implementation:

- `hotpot/cheminfo/geometry/relation.py`
  - `SegmentCycleScreening`
  - `iter_segment_cycle_screenings()`
- `hotpot/cheminfo/geometry/convert.py`
  - `BondRingScreeningReport`
  - `screen_bonds_against_rings()`
  - `screen_bond_ring_relations()`
- `hotpot/cheminfo/forcefields/acceptance.py`
  - force-field acceptance consumes the sparse screening report

For segment endpoints \(\mathbf p_0,\mathbf p_1\) and cycle vertices
\(\mathbf v_i\), define component-wise bounds

\[
\mathbf s_{\min}=\min(\mathbf p_0,\mathbf p_1),\qquad
\mathbf s_{\max}=\max(\mathbf p_0,\mathbf p_1),
\]

\[
\mathbf c_{\min}=\min_i\mathbf v_i,\qquad
\mathbf c_{\max}=\max_i\mathbf v_i.
\]

After the ring surface has first been shown to be valid and complete, the
finite segment is proven not to pierce that surface when at least one Cartesian
axis satisfies

\[
s_{\max,k}+\epsilon_{\mathrm{AABB}} < c_{\min,k}
\quad\text{or}\quad
c_{\max,k}+\epsilon_{\mathrm{AABB}} < s_{\min,k}.
\]

Only this strict-separation case skips the complete relation kernel. Boundary
cases, invalid planar polygons, incomplete non-planar surface families, and
overlapping AABBs use the full geometric predicates. Cycle bounds and surface
preparation are cached once for an entire segment batch.

The sparse API intentionally records only actionable `PIERCES` and
`UNDETERMINED` relations while retaining complete scope counts. The original
dense API remains available when callers need line-extension, boundary-contact,
or complete per-pair evidence.

Important numerical contract: strict finite-segment AABB separation is itself
a proof of `DOES_NOT_PIERCE`. At extreme coordinate scales, the old dense
kernel can conservatively return `UNDETERMINED` because its pair-local tolerance
becomes large, while the sparse screen returns the stronger AABB-proven
`DOES_NOT_PIERCE`. The sparse path therefore preserves geometric safety and all
confirmed piercings, but is not promised to reproduce every conservative
numerical uncertainty emitted by the dense diagnostic API.

### 2.2 Active piercing-pair watch during ring repair

Relevant implementation:

- `hotpot/cheminfo/forcefields/repair.py`
  - `_BondRingPairKey`
  - `_WatchedRingPiercing`
  - `_scan_ring_piercing_watch()`
  - `_scan_confirmed_ring_piercings()`
  - `_untangle_ring_piercings()`

The repair loop now follows this sequence:

```text
full AABB-screened scan
        |
        +-- no confirmed piercing --> optional settling --> full acceptance scan
        |
        `-- confirmed piercings --> freeze stable ring/bond graph keys
                                      |
                                      v
                           open one eligible ring edge
                           perturb and short-optimize
                           restore the same ring edge
                                      |
                                      v
                           scan watched pairs only (AABB + exact fallback)
                                      |
                    +-----------------+------------------+
                    |                                    |
          watched piercing remains              watch clears/is invalid
                    |                                    |
          continue targeted repair                 full checkpoint scan
                                                         |
                                        +----------------+--------------+
                                        |                               |
                               new piercing batch                 no piercing
                               -> replace watch                  -> settle/exit
```

Ring and bond identities are stored as atom-index keys, rather than object
identity, so the watch survives coordinate changes and bond hide/restore
operations. Eligible opening edges are calculated at a full checkpoint and
remain tied to that batch.

If the attempt budget is exhausted, all trajectory frames remain serialized,
but they are not all rescanned. The implementation retains the frame with the
lowest active-watch count (latest frame wins a tie), performs one full scan on
that candidate, and compares it with the best previously full-scanned
checkpoint. A settling optimization is then accepted only if its final full
scan does not increase the confirmed piercing count.

This preserves a safe fallback while changing worst-case rescan cost from
approximately

\[
O(A\,R\,B\,K)
\]

to

\[
O(A\,W\,K)+O(C\,R\,B\,K),
\]

where \(A\) is the number of repair attempts, \(W\) the current watched-pair
count, \(C\) the small number of full checkpoints, \(R\) the ring count,
\(B\) the bond count, and \(K\) the exact relation-kernel cost.

### 2.3 Incremental coordination-bond restoration

Relevant implementation:

- `hotpot/cheminfo/forcefields/repair.py`
  - `_screen_coordination_bond_relations()`
  - `_restore_next_nonpiercing_coordination_bond()`
  - `_restore_coordination_bonds_incrementally()`

For each proposed metal--ligand bond, the workflow now:

1. temporarily restores that one bond;
2. screens only that bond against ligand-skeleton rings;
3. excludes the expected chelate-cycle closure when the tested ring contains
   both endpoints of the proposed coordination bond;
4. accepts or rolls back the trial;
5. performs a full-graph AABB-screened validation after restoration finishes.

Previously accepted bonds are not repeatedly rescanned. The per-trial scope is
reduced from all `ring x bond` pairs to `ring x 1 candidate bond`.

### 2.4 One OBBuilder call per ligand component

Relevant implementation:

- `hotpot/cheminfo/forcefields/ligand.py`
  - `_build_ligand_proxies()`

Each non-metal connected component is embedded once with `OBBuilder`. The built
coordinates are copied as an immutable proposal root. Attempt 1 evaluates that
geometry directly; later attempts independently reset to the root, perturb it,
and run the existing warm-up/scoring/untangling sequence. A builder failure is
reported immediately because repeating the same builder call with the same
input and seed does not add useful conformational diversity.

`max_attempts` remains the upper bound for perturbative candidate search. This
change removes repeated embedding; it does not remove the candidate-selection
or fallback policy.

## 3. Commits

| Commit | Change |
|---|---|
| `3bae2a5` | Add strict segment--cycle AABB screening primitives. |
| `ace544c` | Use sparse screening in force-field acceptance and repair. |
| `878c948` | Add explicit-bond screening and cache each ring AABB. |
| `1e8a437` | Track active piercing pairs between full checkpoints. |
| `fead9f2` | Screen only the proposed bond during coordination restoration. |
| `76d1297` | Reuse one ligand embedding for perturbative proposals. |
| `a1ac0d1` | Remove all-history full rescans from failed ring repair. |
| `fa0ec9b` | Update the geometry public-export regression fence. |

## 4. Verification

### 4.1 Automated regression tests

The current branch passed **625 relevant tests in 34.65 s** under Python
3.11.16. The suite covers:

- all `hotpot.cheminfo.geometry` tests;
- geometry/Core integration and export contracts;
- force-field API, package, optimizer, acceptance, and trajectory behavior;
- complex construction, hydrogen handling, hidden-bond restoration, and staged
  ring-untangling workflows;
- relevant-ring integration.

Focused complex construction and untangling tests independently passed
**80/80**. The watch-path regression test verifies that intermediate attempts
do not invoke full scans; the only scans on budget exhaustion are the initial
checkpoint, the selected terminal candidate, and optional post-settling
acceptance.

`compileall` passed for `geometry`, `forcefields`, Core, and the affected tests
under Python 3.11. A Python 3.9 interpreter also passed syntax compilation; a
full 3.9 runtime test was not possible because the available 3.9 environments
do not contain Open Babel.

The repository-wide test collection was not claimed as clean because the
current runtime lacks unrelated optional dependencies including
`torch_geometric`, `requests`, and `sklearn`. No failure remained in the
625-test change-focused suite.

### 4.2 End-to-end configuration

Input:

`molecules/extractant/extractants.smi` (187 non-comment records)

Final output:

`movie/extractants_eu_forcefield_efficiency_final_16c_20260923`

Execution used 16 worker processes pinned to CPUs 0--15 and one native thread
per worker. Both baseline and optimized runs used:

| Parameter | Value |
|---|---:|
| Metal | Eu |
| CBond threshold | -0.125 |
| Epochs / steps per epoch | 100 / 100 |
| Candidate maximum attempts | 50 |
| Candidate warm-up / score / refinement steps | 500 / 1000 / 3000 |
| Ligand / coordination / complex repair limits | 20 / 20 / 30 |
| Coordination relaxation steps | 100 |
| Perturbation sigma | 0.5 |
| Seed | 20260921 + sample index |
| Quality level | standard |
| Trajectory start | ligand build |
| ONNX provider | CPUExecutionProvider |
| Open Babel | 3.2.1 |

### 4.3 Performance comparison

Baseline: `movie/extractants_eu_forcefield_refactor_16c_20260923`  
Optimized: `movie/extractants_eu_forcefield_efficiency_final_16c_20260923`

| Metric | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Wall time | 2317.87 s | 380.59 s | **-83.58%, 6.09x faster** |
| Aggregate per-case time | 32473.68 s | 5669.52 s | **-82.54%, 5.73x faster** |
| Median case | 113.46 s | 20.70 s | **-81.76%, 5.48x faster** |
| P90 | 345.37 s | 60.95 s | **-82.35%, 5.67x faster** |
| P95 | 547.24 s | 81.50 s | **-85.11%, 6.71x faster** |
| P99 | 956.61 s | 132.45 s | **-86.15%, 7.22x faster** |
| Maximum case | 1738.32 s | 207.61 s | **-88.06%, 8.37x faster** |

Phase totals over the 178 CBond-success cases:

| Phase | Baseline | Optimized | Change |
|---|---:|---:|---:|
| CBond inference | 28.47 s | 28.84 s | +1.31%; effectively unchanged |
| Complex construction | 15126.61 s | 1445.62 s | **-90.44%, 10.46x faster** |
| Post-build/final optimization | 17309.62 s | 4185.31 s | **-75.82%, 4.14x faster** |
| Complete force-field workflow | 32436.23 s | 5630.93 s | **-82.64%, 5.76x faster** |

The measurements establish the combined speed-up, but do not uniquely assign
time to each individual code change because no full factorial ablation was
run. The strongest direct evidence for the dominant construction improvement
is the collapse of ligand proposal branches:

| Activity / artifact | Baseline | Optimized | Change |
|---|---:|---:|---:|
| Ligand proposal branches | 1208 | 284 | -76.49% |
| Ligand proposal frames | 7729 | 2150 | -72.18% |
| Maximum proposals for one case | 50 | 4 | -92.00% |
| Main trajectory frames | 9525 | 8823 | -7.37% |
| All logical trajectory frames | 17254 | 10973 | -36.40% |
| Serialized trajectory bytes | 233196904 | 143402449 | -38.51% |
| Complete result-file bytes | 322432250 | 230870465 | -28.40% |

### 4.4 Result quality and status transitions

| Outcome | Baseline | Optimized |
|---|---:|---:|
| Total inputs | 187 | 187 |
| CBond success | 178 | 178 |
| CBond failure | 9 | 9 |
| Final quality pass | 168 | **171** |
| Final quality failure | 10 | **7** |
| Force-field converged | 159 | 158 |
| Quality pass and converged | 155 | **156** |

The same nine inputs failed CBond inference: 0022, 0134, 0136, 0139, 0141,
0182, 0185, 0186, and 0187. These fail before force-field construction and are
not caused by this refactor.

Status transitions relative to the baseline were:

- 167 `passed -> passed`;
- 6 `failed_quality -> failed_quality`;
- 9 `failed_cbond -> failed_cbond`;
- 4 improvements: cases 0017, 0044, 0126, and 0140 changed from
  `failed_quality` to `passed`;
- 1 regression: case 0070 changed from `passed` to `failed_quality` because the
  final structure contains an exact atom overlap between atom indices 120 and
  121. It remains available with its entire trajectory and final PNG.

The seven final quality-failure cases are 0031, 0045, 0046, 0047, 0054, 0061,
and 0070. No final report contains a confirmed bond--ring piercing as an error;
remaining failures are non-finite gradients, short/abnormal bond lengths,
close atoms, or the case-0070 overlap.

The net pass-rate change is:

- all inputs: 168/187 (89.84%) -> 171/187 (91.44%);
- among CBond-success inputs: 168/178 (94.38%) -> 171/178 (96.07%).

Because the single-embedding candidate path deliberately changes conformer
generation, coordinate-level equality with the old run is neither expected nor
claimed. The quality gate, complete trajectory, and explicit status-transition
audit are the comparison contract.

### 4.5 Remaining long tail

The optimized P95 threshold is 81.50 s. The ten slowest cases are 0070, 0062,
0048, 0045, 0047, 0090, 0089, 0157, 0117, and 0094. Eight of these consume the
full 100 optimization epochs. Cases 0048 and 0089 additionally require 14 and
7 ligand ring-repair attempts, respectively.

The dominant remaining cost is therefore the final optimizer, not repeated
OBBuilder calls or full scans inside the ring-opening loop. The post-build
stage now accounts for 74.34% of total force-field time. A future optimization
could cache fixed-topology ring conversion and split the per-epoch acceptance
path into cheap frame checks plus less frequent full geometric evidence. That
change was deliberately not included here because per-epoch acceptance affects
best-frame selection and early stopping and therefore needs its own behavioral
review.

## 5. Artifact integrity

The final output contains:

```text
movie/extractants_eu_forcefield_efficiency_final_16c_20260923/
├── summary.json
├── results.csv
├── integrity.json
├── trajectory_integrity.json
├── passed.smi
├── failed_quality.smi
├── failed_cbond.smi
├── optimized_all.sdf
├── optimized_passed.sdf
├── cases/
│   └── NNNN/
│       ├── input.smi
│       ├── cbond.smi
│       ├── optimized.mol2
│       ├── optimized.sdf
│       ├── final.png
│       └── trajectory/
└── final_png/
    ├── rendered.json
    └── contact_sheet_001.png ... contact_sheet_012.png
```

The deep archive audit loaded every trajectory through the public
`ForceFieldTrajectoryArchive` API and checked frame/revision bounds, topology
indices, finite coordinates, SDF frame counts, selected-frame correspondence,
and image validity:

- 187 reports;
- 178 readable trajectory archives;
- 178 optimized MOL2 files;
- 178 optimized SDF files;
- 178 case-level final PNG files;
- 12 contact sheets;
- 10973 logical trajectory frames;
- zero global or case-level integrity issues;
- maximum selected-frame versus MOL2 coordinate difference:
  \(4.9996\times10^{-5}\) angstrom, within the configured
  \(10^{-4}\) angstrom serialization tolerance.

## 6. Remaining limitations and next priority

1. Case 0070 is a real final-quality regression and must not be counted as a
   successful structure. Its retained trajectory should be inspected before
   altering chemistry policy.
2. The final optimizer still runs a sparse full-scope AABB screen as part of
   each epoch's acceptance report. This is distinct from the ring-opening
   repair loop optimized here, but it is now the clearest remaining screening
   optimization opportunity.
3. Rings larger than the force-field scope limit of 16 remain excluded and
   generate coverage warnings. This follows the established policy: large,
   normally flexible rings are not actively opened by this workflow.
4. AABB screening is designed for the finite bond--ring piercing question. Use
   the dense geometry API for complete line-extension and boundary-contact
   diagnostics.

## 7. Reproduction commands

```bash
$ export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
$ export LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib
$ taskset -c 0-15 /home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python \
    movie/extractants_eu_forcefield_refactor_16c_20260923/run_validation.py \
    --input molecules/extractant/extractants.smi \
    --output movie/extractants_eu_forcefield_efficiency_final_16c_20260923 \
    --workers 16
$ taskset -c 0-15 /home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python \
    movie/render_final_png.py \
    movie/extractants_eu_forcefield_efficiency_final_16c_20260923 \
    --title 'Eu-extractant force-field final structures (optimized sparse scan)' \
    --workers 16
```

