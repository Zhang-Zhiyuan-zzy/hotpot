# MCA Complex-Molecule GPU Efficiency Report

## Method

- Host: ai-14, one A800 GPU (logical GPU2).
- Input: the same first 64 stratified complex molecules used by the matched
  CPU scaling test: 1,515 atoms and 447 detected sites.
- Input SHA-256:
  `40eea8b16b8a9c451b918c0e7f3c49c0abb3c099d89493f3767d47ed044dcad5`.
- GPU batch size: 2,048 atom rows.
- Five calls through one persistent predictor; median throughput reported.
- Current GPU: native search and the approved 1 site + 3 feature workers.
- Old GPU: commit `13b836f`, before native search, with 9 site + 7 feature
  workers.
- Current CPU reference: synchronized 8 x 8 configuration using 64 CPUs and
  the same 64-molecule workload.

Per-process GPU monitoring showed zero non-zero SM or memory-engine samples
from the resident external GPU2 process during both GPU benchmarks.

## Results

| Workflow | Resources | Median wall time | Throughput |
|---|---|---:|---:|
| Current CPU | 64 CPUs (8 x 8) | 25.050 s for 512 molecules | 20.439 mol/s |
| Current GPU | 1 A800 + 4 CPUs (1 + 3) | 5.191 s for 64 molecules | 12.329 mol/s |
| Old GPU before native search | 1 A800 + 16 CPUs (9 + 7) | 19.863 s for 64 molecules | 3.222 mol/s |

On this workload:

- 64-CPU aggregate throughput is 1.66x the current single-GPU throughput.
- The current single-GPU workflow is 3.83x the old GPU workflow (+282.6%).
- Current GPU CPU reservation is 75% lower than the old GPU default.
- Throughput per reserved CPU is 3.082 mol/s/CPU for the current GPU workflow,
  0.201 for the old GPU workflow, and 0.319 for the 64-CPU workflow. The GPU
  figure includes A800 compute and is therefore a resource-efficiency measure,
  not a CPU-only hardware comparison.

The current GPU's first cold call took 10.582 seconds (6.048 mol/s); its four
warm calls were 5.111--5.243 seconds (12.206--12.522 mol/s). The old GPU's
first call took 25.780 seconds and warm calls remained at 19.457--19.868
seconds.

## GPU activity

One-second per-process sampling across all five calls found:

| Workflow | Consumer samples | Active samples | Active fraction |
|---|---:|---:|---:|
| Current 1 + 3 | 27 | 22 | 81.5% |
| Old 9 + 7 | 84 | 25 | 29.8% |

The old workflow leaves the GPU idle for most of the call despite allocating
more CPU workers. The result is consistent with removal of the Python
site-search supply delay and the subsequent workflow changes; the current
workflow keeps the GPU active much more consistently with only four assigned
CPUs. Because this is an end-to-end version comparison, the full 3.83x gain
must not be attributed to native search alone.

## Prediction compatibility

Current and old GPU runs produced identical digests in every repeat:

- MCA: `7095ae1f61ed39d624daf418efe6642a54f494f7cae98d231cdda8a574ad0e59`
- sites: `c1b85cfab87bb615de705b619ec2bd7828710b237d3e33701ab496c69bc6335e`
- joint: `0dd1293b63bc2a415742d4ce7c224f5eedd1e6e26cea34c1ac7bee8ca1313878`

The joint digest includes atom index, MCA bit pattern, `is_Nuc_site`, site
type, and exact site-to-atom MCA equality. Native search and the worker-ratio
change therefore introduced no observed prediction error or site-semantic
change.

## Conclusion

For this 64-molecule complex workload, use 64 CPU cores when maximum absolute
throughput is the priority and those cores are available. Use the current
single-GPU path when CPU conservation is important: it retains 60.3% of the
64-CPU throughput with 93.75% fewer assigned CPUs, and it is substantially
faster and more GPU-efficient than the pre-native-search 9+7 implementation.

Raw ai-14 artifacts are under `~/mca_test/results/gpu_vs_cpu_complex64`.
