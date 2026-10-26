# Stage 17 follow-up: O0 complex sampling and MCA conformer-engine assessment

## 1. Decision summary

O0 remains the complex construction and optimization default.

The existing O0 implementation can be reused as the kernel of a
complete-complex sampling engine, but the sampling unit must be one complete,
independent O0 workflow execution. `OpenBabelConformerAdapter` itself is only
a one-proposal ligand adapter and is not a multi-conformer complex sampler.

For MCA, native `rdWrappers` is technically viable and preserves every site
record and `is_Nuc_site` flag in the 256-molecule study. Its prediction rank
parity is high, but its values are not bit-identical because the current native
wrapper performs ETKDGv3 embedding without the current MCA route's subsequent
MMFF/UFF optimization. It produced no material end-to-end speedup: -0.09% on
the production-like one-A800 route and +0.99% on the eight-CPU route.

The native `obWrappers` MCA route also completed all 256 molecules and
preserved site semantics, but it was slower, less correlated with the current
predictions, and not bit-reproducible between fixed-seed repeats. It should not
replace the current MCA geometry engine.

## 2. What was compared

The paired input was the existing stratified MCA set:

- 256 molecules;
- 6,486 modeled atoms;
- 1,915 nucleophilic sites;
- input SHA-256
  `40eea8b16b8a9c451b918c0e7f3c49c0abb3c099d89493f3767d47ed044dcad5`;
- charged inputs admitted exactly as in the previous formal MCA benchmark; and
- conformer seed 42.

All routes used the same normalization, ordered site search, Uni-Mol v2
featurization, model artifact, batching, atom order and result join. Only the
coordinate proposal and its local force-field treatment changed.

| Route | Geometry implementation |
|---|---|
| `current_rdkit` | Python RDKit 2026.09.1: add H, ETKDGv3, MMFF when complete otherwise UFF, remove H |
| `native_rdwrapper` | pinned native RDKit 2026.09.1: materialized-H ETKDGv3 through the current `RDKitConformerAdapter`, no force-field optimization, remove H |
| `native_obwrapper` | native Open Babel 3.2.1: add H, OBBuilder, MMFF94s with UFF setup fallback, at most 200 optimization steps, remove H |

The native RDKit route deliberately measures the capability of the current
`rdWrappers`; adding an RDKit force-field API would be a separate component
extension. The Open Babel route measures organic MCA construction, not the
full O0 complex workflow.

Two execution configurations were run three times per route:

| Configuration | Resources | Batch |
|---|---|---:|
| CPU | CPUs 120-127 | 64 atom rows |
| GPU | A800 GPU2, 1 site CPU + 3 feature CPUs | 2,048 atom rows |

The GPU baseline reproduced the prior Plan-18 warm throughput (13.42-13.57
mol/s versus the recorded 13.52 mol/s), providing a cross-study performance
check. Python RDKit and native rdWrapper both reported 2026.09.1.

## 3. MCA business-semantics result

All three routes completed 256/256 inputs. Every paired comparison had:

- identical ordered `(atom_index, site_type)` records;
- identical `is_Nuc_site` for all 6,486 atoms; and
- identical site count, 1,915.

Geometry replacement therefore did not change MCA site-recognition business
semantics. It did change continuous MCA predictions because coordinates are
model inputs.

## 4. Prediction differences

The production-like GPU comparison is the primary numerical result. Each
candidate is paired against `current_rdkit` on the same backend.

### 4.1 Nucleophilic-site predictions

| Route | Pearson | Spearman | Mean signed error | MAE | RMSE | P95 abs. | Max abs. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native rdWrapper | 0.999721 | 0.999568 | +0.319 | 1.889 | 2.739 | 6.000 | 15.500 |
| Native obWrapper, repeat range | 0.999009-0.999045 | 0.998619-0.998709 | +0.730 to +0.813 | 3.505-3.563 | 5.053-5.158 | 10.500-10.750 | 29.250-37.500 |

Errors are in kJ/mol. The native rdWrapper result is high-rank-parity numeric
drift, not exact prediction parity. Its 15.5 kJ/mol worst site prevents calling
the change negligible for every individual prediction even though the
population correlation is excellent.

### 4.2 All atom predictions

| Route | Pearson | Spearman | Mean signed error | MAE | RMSE | P95 abs. | Max abs. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native rdWrapper | 0.999330 | 0.997974 | +0.231 | 1.878 | 2.702 | 5.500 | 17.000 |
| Native obWrapper, repeat range | 0.997562-0.997675 | 0.992791-0.992986 | +0.618 to +0.688 | 3.513-3.562 | 5.043-5.172 | 10.500-10.750 | 34.500-37.500 |

The CPU comparison reached the same engineering conclusion. At nucleophilic
sites, native rdWrapper gave Pearson 0.999683, Spearman 0.999519 and MAE 2.023
kJ/mol. The first native obWrapper repeat gave Pearson 0.998832, Spearman
0.998363 and MAE 3.893 kJ/mol.

## 5. Repeatability

`current_rdkit` and `native_rdwrapper` produced one identical MCA SHA-256
across all three repeats on both CPU and GPU.

The native obWrapper route produced three different hashes despite the fixed
seed. On GPU, repeat 1 versus repeats 2 and 3 had site-level:

| Comparison | Pearson | Spearman | MAE | P95 abs. | Max abs. |
|---|---:|---:|---:|---:|---:|
| Repeat 1 vs 2 | 0.999780 | 0.999697 | 1.109 | 5.038 | 22.500 |
| Repeat 1 vs 3 | 0.999813 | 0.999692 | 1.036 | 4.537 | 20.500 |

Site identity remained exact. The drift is in coordinates and continuous
predictions. O0 currently obtains its stronger seed lifecycle by starting a
fresh ligand-build process and setting Open Babel random state before the
process starts. Reusing persistent MCA feature workers does not reproduce that
lifecycle. A deterministic native Open Babel sampling contract would be
required before this route could satisfy MCA repeatability without paying for
one child-process startup per molecule.

## 6. Performance

### 6.1 Production-like single-A800 route

| Route | Median wall time | Median throughput | Throughput change |
|---|---:|---:|---:|
| Current RDKit | 19.077 s | 13.420 mol/s | reference |
| Native rdWrapper | 19.094 s | 13.407 mol/s | -0.09% |
| Native obWrapper | 20.365 s | 12.570 mol/s | -6.33% |

The native rdWrapper difference is operationally zero. Native binding does not
remove the cost of ETKDG itself, and the current integration still performs a
RDKit-to-Hotpot projection to preserve represented stereochemistry. Native
obWrapper adds enough construction and force-field work to lower throughput.

### 6.2 Eight-CPU route

| Route | Median wall time | Median throughput | Throughput change |
|---|---:|---:|---:|
| Current RDKit | 79.451 s | 3.222 mol/s | reference |
| Native rdWrapper | 78.673 s | 3.254 mol/s | +0.99% |
| Native obWrapper | 81.919 s | 3.125 mol/s | -3.01% |

The roughly one-percent CPU gain is too small to establish a useful speedup
and disappears on the production-like GPU pipeline.

A separate one-core transport microbenchmark reconstructed the RDKit molecule
from the current DTO and converted it to Hotpot 2,560 times in 10.177 seconds,
or 3.975 ms/molecule. Even the optimistic bound that all of this work vanished
and divided perfectly across three feature workers is about 0.34 seconds for
256 molecules, 1.8% of the measured 19.08-second GPU wall time. Direct packed
transport is still the correct implementation shape, but it does not conceal a
large conformer-native speedup in these results.

## 7. Can O0 provide a complex conformer-sampling engine?

Yes, but only at the complete-workflow level.

The current low-level behavior is narrower:

1. `OpenBabelConformerAdapter.sample()` ignores the requested conformer count
   and returns exactly one OBBuilder coordinate proposal.
2. The adapter is invoked independently for each non-metal component after
   metal-ligand bonds are hidden.
3. O0 then explores bounded perturbations around that proposal, applies ligand
   force fields and geometry checks, places the components, restores
   coordination, optimizes the complete complex and selects one final frame.
4. The internal attempts are recovery/candidate attempts. They are not exposed
   as an ensemble and should not be relabeled as independent complex
   conformers.

A correct O0 sampling engine would therefore execute the complete O0 workflow
independently for seeds `seed + ordinal`, retain only accepted optimized
structures, and optionally deduplicate accepted structures after the existing
gate. It must not call raw `obWrappers.build()` on the intact metal complex and
bypass Hotpot's coordination topology, restoration and acceptance stages.

This engine would be useful for producing several valid starting structures,
but it would remain a force-field proposal ensemble rather than a claim of a
thermodynamic conformer distribution. Its cost scales approximately linearly
with requested sample count. The formal Plan-17 O0 run measured 9.644 mean
seconds per attempted case and 1.493 attempted cases/s under its parallel
execution policy; requesting several accepted structures multiplies the work.

If this capability is implemented, it belongs above the current ligand
`ConformerSampler` seam, for example as a complete-complex sampling operation
that reuses the one O0 implementation. Expanding
`OpenBabelConformerAdapter` into a second complex workflow would mix abstraction
levels and duplicate O0 logic.

## 8. Native rdWrapper feasibility for MCA

The replacement is feasible in terms of data flow and business semantics, but
it is not presently justified as the default:

- 256/256 completion and exact site semantics prove functional viability;
- the native sampler is deterministic under the tested fixed seed;
- prediction correlations are high, but output is not exact and has individual
  site deviations up to 15.5 kJ/mol;
- the current native component omits the baseline MMFF/UFF step; and
- measured GPU throughput did not improve.

Native conformer generation alone also does not remove MCA's Python-RDKit
runtime dependency. `mol_to_unimolv2()` still reads RDKit atom, bond,
stereochemistry and conformer objects. The evaluated adapter additionally
reconstructs Hotpot state from the RDKit-bearing MCA DTO so that the native
wrapper receives complete stereochemistry. A real cutover should instead
carry the common packed Hotpot/stereo transport from normalization and should
avoid this reverse conversion.

The next performance-relevant native boundary would be the combined
conformer-plus-feature stage, or at least direct native coordinates into a
native/packed featurizer. Merely replacing `EmbedMolecule()` with the existing
rdWrapper changes scientific preprocessing without accelerating the complete
pipeline.

## 9. Recommendation

1. Keep O0 as the complex default, as decided.
2. Treat repeated complete O0 executions as the only valid current route to a
   complex conformer ensemble. Implement it only when multiple accepted
   starting structures are a concrete requirement.
3. Keep the current Python-RDKit MCA geometry route as default.
4. Do not switch MCA to native rdWrapper for performance. Reconsider only with
   a native force-field/parity decision or a larger native
   conformer-plus-featurizer cutover.
5. Do not use native obWrapper as the default MCA organic geometry engine. It
   is slower, less correlated, and not repeatable in persistent workers.

## 10. Evidence

Raw artifacts are under
`ai-14:~/mca_test/mca-conformer-study-6639754/results`. Primary hashes:

| Artifact | SHA-256 |
|---|---|
| Benchmark driver | `1608cf373d5d5d61b30c8ba0b60bcc7b17a999914016ece18bc61d200527b5e0` |
| CPU analysis | `5b98aceec61309d64969f4872894f470ff4f43d3f25ac5587e16cc0ec956da69` |
| GPU analysis | `b7edf25c05df6becd1ac3dd3537be82c9fdac443588c0973d0448a4273711f1a` |
| GPU current route | `9c61b24545d1cb02e1ef7118b540156714161c46592e1c7bbd3ef059da9adb2c` |
| GPU native rdWrapper route | `6f094b39e3774f2eb0a909528cf85e86000774d71ddde4040109492d2748208b` |
| GPU native obWrapper route | `c0cd98d39b1de70e2c72b16c6499ab8d011b97b38a0552629ee482593bb074de` |

## 11. Follow-up: random-seed sensitivity

The initial fixed-seed repeatability result must not be read as seed
independence. Repeating a fixed seed asks whether one request is reproducible;
changing the seed asks whether the sampler reaches different conformational
basins. The latter was measured separately with seeds 42, 43 and 44.

### 11.1 Pure-organic MCA molecules

The 256-molecule MCA corpus was generated three times and compared by stable
atom-row identity after Kabsch alignment. These are not symmetry-corrected
RMSDs, so a symmetry-equivalent atom permutation can inflate an individual
value; they nevertheless measure exactly the coordinate rows consumed by the
model.

| Geometry route | Pairwise heavy-atom RMSD P50 | P95 | Cases with maximum RMSD >= 1 A | >= 2 A |
|---|---:|---:|---:|---:|
| Native rdWrapper ETKDGv3, no FF | 1.666 A | 3.065 A | 236/256 (92.2%) | 115/256 (44.9%) |
| Current Python RDKit ETKDGv3 + MMFF/UFF | 1.701 A | 3.193 A | 236/256 (92.2%) | 124/256 (48.4%) |

The different seeds therefore produce materially different conformers. Local
MMFF/UFF optimization does not collapse them to one unique minimum; it only
relaxes each proposal into a nearby basin.

MCA remains population-stable but not value-invariant. Across the three seed
pairs, nucleophilic-site Pearson was 0.999411-0.999451 and Spearman was
0.999159-0.999214 for the native ETKDG-only route. Site MAE was 2.58-2.61
kJ/mol and the worst individual difference was 49.5 kJ/mol. The current
ETKDG+MMFF/UFF route was similar: Pearson 0.999429-0.999515, Spearman
0.999157-0.999264, site MAE 2.29-2.52 kJ/mol and maximum 39.125 kJ/mol.
Every ordered site record and every `is_Nuc_site` flag remained exact.

### 11.2 Complete O0 complex structures

All 182 CBond-positive Plan-17 cases were run through the complete O0 workflow,
not merely through the Open Babel ligand proposal. Because O0 applies the
one-based case index to its configured seed, seed bases 42, 43 and 44 mean
effective seeds `base + case_index`.

| Metric | Result |
|---|---:|
| Completed executions | 546/546 |
| Accepted for seed bases 42 / 43 / 44 | 181 / 180 / 181 of 182 |
| Accepted under all three seeds | 180/182 |
| Pairwise ligand-heavy RMSD P50 / P75 / P95 | 0.111 / 1.002 / 2.411 A |
| Cases with maximum ligand-heavy RMSD >= 0.5 / 1 / 2 A | 79 / 62 / 26 of 181 comparable |
| Metal-donor distance-vector RMSD P50 / P95 | 0.0015 / 0.0525 A |
| Donor-metal-donor angle-vector RMSD P50 / P95 | 0.091 / 14.442 degrees |

O0 therefore has a mixed response: many cases return to nearly the same local
minimum, while a substantial flexible subset reaches a different final
structure. Maximum ligand RMSD correlates with rotatable-bond count
(Spearman 0.609), although flexibility is not the sole cause. The coordination
distances are much more stable than the ligand conformation because the O0
restoration and optimization stages constrain that geometry. Case 96 failed
the final acceptance gate under all three seed bases; case 88 additionally
failed only for seed base 43.

## 12. Native RDKit UFF/MMFF feasibility

`MBFF94` is interpreted here as RDKit `MMFF94`. Supporting native UFF,
MMFF94 and MMFF94s inside rdWrapper is feasible and is the correct next step
if the native component is to reproduce the current MCA geometry semantics.

The implementation does not require a new third-party dependency. The locked
RDKit 2026.09.1 ETKDG closure already contains `ForceFieldHelpers_static`,
`ForceField_static`, `Optimizer_static` and `Trajectory_static`; ETKDG itself
links that closure. Direct force-field use would pull additional objects from
the already approved archives and requires refreshed binary-size and exported-
symbol evidence, but not a new dependency family or RDKit version.

The native operation should be one fused pipeline over one native RDKit
molecule:

1. reconstruct the pinned RDKit graph and materialize hydrogens;
2. run ETKDGv3 with the caller's explicit seed;
3. use MMFF94 only when `MMFFHasAllMoleculeParams()` is true, otherwise UFF;
4. optimize each requested conformer;
5. return mapped coordinates plus the force field selected, parameter-coverage
   result, RDKit optimization status, convergence status and finite final
   energy; and
6. project heavy-atom coordinates only at the MCA boundary.

The current MCA failure semantics must be retained exactly: a negative RDKit
optimization status is an error, while a positive status means the iteration
limit was reached and is not currently treated as a fatal error. UFF is a
parameter-coverage fallback, not a retry after an MMFF optimization failure.
For one requested conformer, the implementation should reproduce the current
`EmbedMolecule()` call path rather than assume that
`EmbedMultipleConfs(count=1)` is bit-equivalent.

The API work is medium difficulty even though the RDKit call itself is small.
It spans the standalone C API, C++ facade, Python facade, immutable result
contracts, exception/status translation, symbol allowlist, locked build proof,
wheel audit and standalone tests. The RDKit version remains exactly 2026.09.1;
no compatibility profile or multi-version branch is needed.

### 12.1 Expected benefit and replacement boundary

The primary benefits are semantic parity, one implementation shared by the C++
and Python facades, and removal of Python-to-RDKit geometry calls. A large MCA
throughput improvement should not be promised. Python RDKit already executes
ETKDG and its force fields in C++, and the existing native ETKDG-only route was
-0.09% on the one-A800 workflow and +0.99% on the eight-CPU workflow. Fusing
embedding and optimization avoids an extra coordinate round trip, but it does
not remove the cost of either algorithm.

Native UFF/MMFF is still worth implementing as an rdWrapper capability. The
first replacement target should be the current MCA organic geometry stage,
with the fixed MCA seed 42 passed explicitly. Default cutover should require:

- exact MMFF-versus-UFF routing and failure semantics;
- 256/256 completion on the existing corpus;
- exact ordered sites and `is_Nuc_site`;
- same-seed coordinate and MCA numerical parity against the pinned Python
  RDKit reference, with every discrepancy explained; and
- no material throughput regression on the CPU and one-A800 routes.

This does not by itself remove MCA's Python-RDKit runtime dependency because
`mol_to_unimolv2()` still consumes an RDKit molecule. Eliminating that
dependency requires a later packed/native Uni-Mol feature builder. That larger
boundary is medium-to-high difficulty and is the part with a plausible
performance payoff. Complex generation and optimization remain on O0; the
native RDKit force-field addition must not alter that decision.

## 13. Follow-up evidence

| Artifact | SHA-256 |
|---|---|
| O0 182-case, three-seed results | `9d7feadce9f4dc2e6506f49924d058fd6cd2a197e4d4524b7d4e6cbecb8b2ce1` |
| Native rdWrapper geometry diversity | `1b56cc4e051c09f144eb863d7f0db10fb9a4877ffca2f5978c31c80cc472e557` |
| Current Python RDKit geometry diversity | `3d5f1cf508895da190de875ba09b6d56a2662607a200ac9e709ba2480e5faf80` |

The O0 artifact is under
`ai-14:~/mca_study/plan19-o0-seed-diversity-6639754/`. The organic geometry
and MCA prediction artifacts are under
`ai-14:~/mca_test/mca-conformer-study-6639754/results/`.

