# MCA Resource-Efficient Workflow Implementation Report

## 1. Outcome

The MCA workflow now exposes one bounded CPU budget and reuses it across
parallel preprocessing and ONNX inference. The final GPU default is one site
worker plus three conformer/feature workers per card, as approved after the
end-to-end ai-14 measurements.

No prediction semantics changed. CPU and GPU runs retained their respective
exact MCA, ordered-site, and joint hashes. The joint hash includes atom index,
MCA bit pattern, `is_Nuc_site`, site type, and the exact site-to-atom MCA join.

## 2. Implemented changes

### CPU workflow

- Added `MCACpuPredictor(cpu_count: int | None = None)` and CLI
  `--cpu-count`.
- `None` resolves to at most eight CPUs from the caller's affinity; an explicit
  request that does not fit fails before startup.
- Added a persistent spawn pool with one individually pinned process per
  selected CPU.
- Fused each molecule's worker task in the existing scientific order: site
  detection, conformer generation, feature extraction.
- Added a bounded window of at most 64 molecules. A single molecule executes
  directly without starting a pool.
- Configured ONNX Runtime with the same total as its intra-op thread count.
- Preprocessing and ONNX execute in phases on the same CPU IDs; they never own
  simultaneous independent budgets.
- Preserved the existing global `LogicalBatchPlanner`, atom-row batch
  boundaries, padding widths, input order, and ordered result join.
- Restored caller affinity after construction and after every prediction call.

### GPU workflow

- Reduced the approved default from 9 site + 7 feature workers to 1 site + 3
  feature workers: 16 to 4 reserved CPUs per GPU.
- Kept the existing direct resource controls and did not introduce a profile
  layer or a second backend implementation.

## 3. Verification

Local verification at final HEAD:

- `tests/mca`: 191 passed.
- MCA calculator integration: 3 passed.
- Focused import, GPU routing, and GPU pipeline suite: 26 passed.
- Python byte-compilation and `git diff --check` passed.
- Ruff was not available in the local environment.

ai-14 used the deployed source at
`~/mca_test/hotpot-resource-bc927fe`; raw results are under
`~/mca_test/results/resource_workflow_bc927fe`. CPU comparisons used the same
cores 120--127 in the same time window. GPU comparisons used A800 GPU2. Its
resident external process remained at zero SM and memory-engine utilization in
per-process monitoring; only the benchmark consumer showed non-zero activity.

## 4. ai-14 CPU benchmark

Each result is the median of three runs with an eight-CPU total budget and
batch size 64. The comparison is paired against the previous serial production
backend on the same cores.

| Workload | Previous | Optimized | Throughput change | Wall-time change |
|---|---:|---:|---:|---:|
| Piperidine, 256 molecules | 31.500 mol/s, 8.127 s | 39.177 mol/s, 6.534 s | +24.4% | -19.6% |
| Stratified complex set, 32 molecules | 1.963 mol/s, 16.303 s | 2.222 mol/s, 14.401 s | +13.2% | -11.7% |

Median process-tree CPU time fell from 46.441 to 41.096 CPU-s on the simple
set (-11.5%) and from 105.837 to 101.651 CPU-s on the complex set (-4.0%).

Exact semantic digests:

| Workload | MCA SHA-256 | Sites SHA-256 | Joint SHA-256 |
|---|---|---|---|
| Simple 256 | `c74d76e00df6972722603260d7331dce140dc241d25bb940f768046835a24f82` | `33519e36432b580e6c3e79500c4e906279c3f74536fe44232d3bab266c621813` | `1b46ff03680bf8631ea47a21910e9125c3f81334847ef42919de08e8dc19a5b5` |
| Complex 32 | `672a2093a319e3fff2298b4ce9fa990f70fd0f4d2414d199fa2af2e6bd703b65` | `5d35bdc499eda972148ee60629168cc602ef4905c4a4690d668293054544afc0` | `beddd228b6421538b4e267cdad3a5f59629cf5a36032616d3c407ff1e008b988` |

All hashes were identical between the previous and optimized CPU workflows and
across all repeats.

## 5. ai-14 GPU benchmark and accepted trade-off

The 256-molecule stratified input contains 6,486 atoms and 1,915 detected
sites. The GPU batch size was 2,048.

Feature-worker isolation established the lower inference-side bound:

| Site + feature workers | Median throughput | Median wall time |
|---|---:|---:|
| 1 + 2 | 14.273 mol/s | 17.936 s |
| 1 + 3 | 14.994 mol/s | 17.073 s |

Three feature workers are 5.1% faster and reach the previously measured
14.6--15.0 mol/s single-A800 inference ceiling, so two workers are not the
non-throttling minimum.

The complete production workflow, including routing and site work, measured:

| Default | Reserved CPU/GPU | Steady median | Change from previous |
|---|---:|---:|---:|
| Previous 9 + 7 | 16 | 14.712 mol/s | reference |
| Final 1 + 3 | 4 | 13.520 mol/s | -8.1% throughput, -75% reserved CPUs |

For the final 1+3 production run, the first cold call was 24.235 s
(10.563 mol/s); warm calls were 18.935 and 18.840 s (13.520 and 13.588 mol/s).
The lower full-workflow number is site/routing overhead, not an underfed ONNX
GPU branch. The user explicitly accepted this resource-efficiency trade-off;
1+3 is therefore the final default.

The exact GPU digests were unchanged across every worker comparison and
repeat:

- MCA: `fa8e55f55519fac1afca9874f1535de6ee936c89ee88f9bbcda298e31ad158e0`
- sites: `5d6150efed59cbf96cc83324c4c35c387630e71e25826bad5382378dfe4a10b3`
- joint: `8fe4ede2dad5e7caf1575432f348f9b10ec4f251eb14c88c41d2fb5b790f87e1`

## 6. Final assessment

The CPU optimization is a clear win in both throughput and CPU work while
retaining exact business semantics. The GPU change is primarily a resource
efficiency optimization: it removes 12 reserved CPUs per A800 at the accepted
cost of about 8% complete-workflow throughput on the measured 256-molecule
complex workload. The inference branch itself remains fully supplied with
three feature workers.
