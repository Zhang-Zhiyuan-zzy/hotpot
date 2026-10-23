# Plan 17 formal failure and efficiency analysis

Analysis date: 2026-10-23

Authoritative automated source checkpoint:
`b0aa6c5224a97469359c53448ba48d24d8a46112`

Formal matrix root: `ai-14:~/mca_study/plan17-b0aa6c5/formal-matrix`

Derived analysis root:
`ai-14:~/mca_study/plan17-b0aa6c5/failure-analysis`

This report analyzes the 41 failed primary route-cases and the efficiency of
all five primary routes. It does not change any A01-A10 decision, convert a
rejected frame into an accepted structure, or add a failed coordinate to the
governed benchmark dataset.

## 1. Did the failed routes produce structures?

The 41 failures divide into three evidence classes:

| Evidence class | Count | Meaning |
| --- | ---: | --- |
| Nondegenerate rejected complex 3D structure | 10 | The route reached a finite complex coordinate frame, but the frame failed A08 or optimization could not recover a finite gradient |
| Degenerate recorded attempt only | 1 | F1 case 10 archived two events sharing one 174-atom all-zero coordinate revision; this is not a valid 3D proposal |
| No recorded route frame | 30 | Failure occurred before a route 3D proposal existed; only the input ligand graph can be displayed |

The 10 nondegenerate rejected structures are O0 cases 24 and 96; F1/F2 case
29; C1 cases 21 and 70; and C2 cases 21, 54, 63, and 70. The supplementary
gallery renders these exact rejected frames with red diagnostic labels.

F1 case 10 is displayed as a side-by-side view of its collapsed recorded frame
and input ligand graph. All other pre-generation failures display the input
ligand graph and state explicitly that it is not route-generated 3D output.

## 2. Failure-category summary

| Category | Route-cases | Count | Interpretation |
| --- | --- | ---: | --- |
| A08 standard-geometry rejection | O0: 24, 96; F1: 29; F2: 29; C1: 21; C2: 21, 54, 63 | 8 | A finite selected frame exists but fails named geometry checks |
| Unrecovered unavailable gradient | C1: 70; C2: 70 | 2 | A finite coordinate frame exists, but no finite optimization direction was restored |
| ETKDG returned no conformer | F1: 10; F2: 10; C1: 10, 22, 25, 28, 96, 101, 115, 117; C2: 10, 28, 96, 101, 115 | 15 | The fixed-seed ligand or whole-complex generation call returned zero conformers |
| C2 bounds failed triangle smoothing | C2: 22, 25, 64, 66, 67, 71, 72, 74, 75, 79, 80, 81, 103, 104, 105, 109 | 16 | The fixed coordination-bound hypothesis was internally inconsistent, so ETKDG did not start |

The formal native wrapper records `RDKit ETKDGv3 returned no conformers` but
does not expose RDKit's internal embedding-failure counters. Therefore the 15
zero-conformer cases can be localized exactly to ligand or whole-complex
ETKDG generation, but a more specific distance-geometry failure subtype is
not present in the frozen evidence. Assigning one from molecular appearance
would be speculation. Adding native RDKit failure-counter transport would be
a separate instrumentation change requiring a rebuilt native component and
new validation, not a reinterpretation of this matrix.

## 3. Per-failure inventory

`Frames` counts all archived trajectory frames for that failed route-case, not
only the optimization frames translated into A01-A10 evidence.

| Route | Case | Failure phase/category | Exact recorded result | Diagnostic structure | Frames |
| --- | ---: | --- | --- | --- | ---: |
| O0 | 24 | A08 geometry | atom_too_close x3; bond_length_ratio x41; short_bond x11 | rejected complex 3D | 142 |
| O0 | 96 | A08 geometry | bond_length_ratio; short_bond | rejected complex 3D | 131 |
| F1 | 10 | ligand ETKDG | single conformer-provider call returned no conformer | collapsed 174-atom all-zero attempt plus input graph | 2 |
| F1 | 29 | A08 geometry | atom_too_close x5; bond_length_ratio x5; short_bond x3 | rejected complex 3D | 125 |
| F2 | 10 | frozen ligand ETKDG | returned no conformer before replay | input ligand graph only | 0 |
| F2 | 29 | A08 geometry | atom_too_close x5; bond_length_ratio x5; short_bond x3 | rejected complex 3D | 125 |
| C1 | 10 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 21 | A08 geometry | bond_length_ratio; short_bond | rejected complex 3D | 114 |
| C1 | 22 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 25 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 28 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 70 | optimization | unrecovered unavailable gradient at `gradient:main:130` | rejected complex 3D | 132 |
| C1 | 96 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 101 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 115 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C1 | 117 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C2 | 10 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C2 | 21 | A08 geometry | atom_too_close | rejected complex 3D | 104 |
| C2 | 22 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 25 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 28 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C2 | 54 | A08 geometry | atom_too_close | rejected complex 3D | 87 |
| C2 | 63 | A08 geometry | atom_too_close x2 | rejected complex 3D | 40 |
| C2 | 64 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 66 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 67 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 70 | optimization | unrecovered unavailable gradient at `gradient:main:123` | rejected complex 3D | 125 |
| C2 | 71 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 72 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 74 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 75 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 79 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 80 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 81 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 96 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C2 | 101 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |
| C2 | 103 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 104 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 105 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 109 | bounds construction | triangle smoothing failed | input ligand graph only | 0 |
| C2 | 115 | whole-complex ETKDG | returned no conformer | input ligand graph only | 0 |

The ten nondegenerate rejected frames have the following selected numerical
evidence. Finite values do not override the geometry or continuation failure.

| Route/case | Selected energy | RMS gradient | Maximum gradient | Final reason |
| --- | ---: | ---: | ---: | --- |
| O0/24 | 229,288.081 | 71,836.950 | 431,788.115 | A08 geometry |
| O0/96 | 8,500.771 | 344.299 | 1,410.235 | A08 geometry |
| F1/29 | 34,798.976 | 14,213.810 | 46,298.793 | A08 geometry |
| F2/29 | 32,760.764 | 68,789.476 | 248,468.559 | A08 geometry |
| C1/21 | 15,752.453 | 98.738 | 488.912 | A08 geometry |
| C1/70 | unavailable | unavailable | unavailable | unrecovered gradient |
| C2/21 | 15,027.591 | 139.342 | 760.140 | A08 geometry |
| C2/54 | 2,281.952 | 2.345 | 9.233 | A08 geometry |
| C2/63 | 4,647.535 | 44.141 | 170.799 | A08 geometry |
| C2/70 | unavailable | unavailable | unavailable | unrecovered gradient |

## 4. Route efficiency

### 4.1 Reliable per-case elapsed comparison

All rows use the same 182-case denominator. `O0 ratio` is elapsed-sum divided
by the O0 elapsed sum; lower is faster. `s/accepted` is deliberately shown in
addition to mean seconds per attempted case so that failures remain visible.

| Route | Passed | Rate | Mean s/case | s/accepted | p50 s | p95 s | Max s | Elapsed sum s | O0 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O0 | 180/182 | 98.901% | 9.644 | 9.751 | 8.027 | 19.810 | 44.125 | 1,755.171 | 1.000 |
| F1 | 180/182 | 98.901% | 8.516 | 8.611 | 7.474 | 19.213 | 36.617 | 1,549.921 | 0.883 |
| F2 | 180/182 | 98.901% | 8.519 | 8.613 | 7.458 | 19.295 | 36.828 | 1,550.421 | 0.883 |
| C1 | 172/182 | 94.505% | 3.797 | 4.018 | 2.486 | 9.919 | 36.097 | 691.132 | 0.394 |
| C2 | 157/182 | 86.264% | 3.354 | 3.889 | 2.239 | 8.927 | 36.680 | 610.513 | 0.348 |

F1's elapsed sum and mean are 11.694% below O0. F2 is 11.666% below O0.
The F1 repeat produced 180/182 again with a 1,546.564-second elapsed sum and
8.498-second mean, 0.217% below the primary F1 run. This supports timing and
classification stability but does not repair F1 cases 10 and 29.

C1/C2 must not be presented as 2.5-2.9x production speedups. They enter at a
different workflow stage, omit ligand build and coordination restoration, and
10 or 25 cases terminate early. Their lower elapsed sums describe these exact
research experiments, including failures, rather than equivalent successful
throughput.

### 4.2 Stage sums

Parentheses show the number of route-cases that reached the stage.

| Route | Proposal/generation s | Ligand build s | Restoration s | Optimization s | Acceptance s |
| --- | ---: | ---: | ---: | ---: | ---: |
| O0 | 0.001 (182) | 584.991 (182) | 64.273 (182) | 845.121 (182) | 3.976 (182) |
| F1 | 0.001 (181) | 546.727 (181) | 62.991 (181) | 681.975 (181) | 3.864 (181) |
| F2 | 19.163 (181) | 529.111 (181) | 63.023 (181) | 682.573 (181) | 3.913 (181) |
| C1 | 51.819 (174) | not applicable | not applicable | 510.912 (174) | 1.588 (174) |
| C2 | 69.489 (161) | not applicable | not applicable | 478.484 (161) | 1.525 (161) |

The O0-to-F1 elapsed-sum reduction is 205.250 seconds. Of the named stage
differences, 163.146 seconds comes from optimization and 38.265 seconds from
ligand build; restoration changes by only 1.282 seconds. The formal
instrumentation does not time F1 sampling separately because it occurs inside
the ligand worker. Consequently this result supports “the complete F1 route
was faster in this run,” not “native ETKDG sampling alone is 11.7% faster.”

### 4.3 Observed 16-worker completion spans

These spans are reconstructed from each result's recorded completion time and
runner elapsed value. Adjacent routes overlap at worker-pool boundaries, so
the spans are useful scheduling observations but are not additive wall times.

| Route | Completion span s | Attempted route-cases/s | Accepted route-cases/s |
| --- | ---: | ---: | ---: |
| O0 | 121.898 | 1.493 | 1.477 |
| F1 | 112.151 | 1.623 | 1.605 |
| F2 | 114.142 | 1.595 | 1.577 |
| C1 | 64.023 | 2.843 | 2.687 |
| C2 | 55.625 | 3.272 | 2.822 |

The complete six-run matrix, including the F1 repeat, produced 1,092 results
in 506.107 wall seconds: 2.157 attempted route-cases/s and 2.073 admitted
structures/s. This is the only exact whole-matrix wall-throughput result.

### 4.4 CPU and memory interpretation

The recorded `user CPU` is `getrusage(RUSAGE_SELF)` for each matrix worker.
O0/F1/F2 perform ligand work in spawned child processes, and child CPU is not
included. It is therefore parent-worker CPU evidence, not total route CPU
consumption, and must not be used to claim CPU-efficiency superiority over
C1/C2. A future total-CPU comparison must add `RUSAGE_CHILDREN` or cgroup-level
accounting before and after every route-case.

The reported RSS is also a process high-water mark, not a per-case memory
delta. Later tasks inherit prior worker high-water state. The maxima were
227,148 KiB for O0/F1, 231,164 KiB for F2, and 236,616 KiB for C1/C2; these
values establish an observed ceiling for the run but do not measure route-only
memory allocation.

## 5. Diagnostic gallery and evidence identities

The original 910-cell acceptance gallery remains unchanged and continues to
show placeholders for failed results. This prevents rejected frames from being
mistaken for admitted selected structures. The supplementary failure gallery
is explicitly diagnostic and renders every one of the 41 failures.

| Artifact | SHA-256 |
| --- | --- |
| `failure-analysis.json` | feb795712f715ad3ecd8147a73fae845582caf6ba2ba69bacd40113498b2b224 |
| `failure-analysis.md` | cfc3189ad074635af71feb6b66a97f6e8ef08010c7588a95983e063271e58cfe |
| O0 failures | 75a823b70986247c52f9284fd73186534b09eb67045791fe0b0e924f9f430c8d |
| F1 failures | c8c66769941455994802c83bf9e322610975504eefea0a736be29eb5cd50000b |
| F2 failures | 7855b6330131be415486a070678aefb51f5eba00feb08bd4dcc95e436407575d |
| C1 failures | 517e814139bb5bf96f53fe68026ab00d505f58a639f6be38a87e2ea21213719a |
| C2 failures | 23936b762486372284a21d25d838efe84e3e684e624b5c581f795b84f8ab01be |
| Combined 41-case sheet | 492efa16a9d044ba744ad7b6879e7255fac09e8296360453b0e88e894b798dc5 |

Each diagnostic source and image has its own digest in
`failure-analysis.json`. The 10 rejected complex frames and the collapsed F1
attempt remain excluded from the 869 admitted coordinate assets.
