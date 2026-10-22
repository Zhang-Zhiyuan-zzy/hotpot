# MCA Resource-Efficient Workflow Implementation Log

## 2026-10-22: ai-14 resource characterization

- GPU conformer/feature supply on 1,024 stratified molecules: one worker
  12.75 molecules/s, two 27.07, three 40.70.
- One native-search site worker on 4,096 molecules: 123.81 molecules/s.
- One A800 GPU consumption: 14.6--15.0 molecules/s.
- CPU ORT scaling on ai-14 peaked at eight threads; 16, 32, and 64 threads
  were slower, with 64 threads reaching only 9.45 molecules/s on the simple
  workload.
- Windowed phase reuse improved the simple workload by 22--30% and the
  stratified workload by 9--10% at four/eight CPU budgets.
- Exact MCA/site/joint digests were unchanged; the joint digest includes site
  type and `is_Nuc_site`.

## 2026-10-22: implementation start

- Created branch `perf/mca-resource-efficient-workflow` from `9640d90`.
- Frozen the implementation decisions in
  `mca_resource_efficiency_implementation.md`.

## 2026-10-22: resource API and execution core

- Initially changed the GPU defaults from 9/7 to 1 site worker and 2
  conformer/feature workers per card.
- Added the single CPU resource control `cpu_count`, resolving by default to at
  most eight affinity-visible CPUs and failing before startup on an explicit
  shortfall.
- Wired the same CPU count into ONNX Runtime's intra-op thread count.
- Replaced serial CPU preprocessing with a persistent, individually pinned
  spawn pool. Each per-molecule task preserves the ordered site -> conformer ->
  feature path.
- Added a bounded 64-molecule window. Preprocessing and ONNX reuse the same CPU
  IDs in separate phases; the existing global logical batch planner and ordered
  result join remain unchanged.
- Added `--cpu-count` and synchronized MCA API/CLI documentation.

## 2026-10-22: local validation

- Passed all 191 MCA tests with the real CPU model where applicable, including
  persistent-pool reuse, exact worker pinning, and caller-affinity restoration.
- Passed the three calculator integration tests and 26 focused import,
  GPU-routing, and GPU-pipeline tests.
- Confirmed exact golden MCA values, ordered site types, the site-to-atom value
  join, deterministic logical batch padding, bounded generator consumption,
  prepared-molecule reuse, spawn execution, and CLI resource routing.
- `git diff --check` passed. Ruff is not installed in the local environment;
  Python byte-compilation passed and changed lines were checked for the
  repository's formatting limits.

## 2026-10-22: reproducible benchmark harness

- Added a production-API benchmark runner for both backends. It records wall
  throughput, process-tree CPU time, resolved resources, caller-affinity
  restoration, and exact MCA/site/joint hashes including `is_Nuc_site`.

## 2026-10-22: GPU default corrected by production-path measurement

- Per-process GPU2 monitoring showed zero SM activity from the resident
  external process during both comparisons.
- On the same 256-molecule stratified workload, two feature workers reached
  14.273 molecules/s while three reached 14.994 molecules/s. All three exact
  semantic hashes matched.
- The three-worker result reaches the independently measured 14.6--15.0
  molecules/s A800 ceiling; two workers leave 5.1% throughput unused once
  complete pipeline overhead is included.
- Corrected the final GPU default to 1 site + 3 conformer/feature workers per
  card: three feature workers are the measured minimum that does not throttle
  the inference branch; the single-site-worker trade-off was evaluated and
  explicitly accepted.

## 2026-10-22: final ai-14 benchmark and closure

- Paired CPU tests on cores 120--127 improved the 256-molecule simple workload
  from 31.500 to 39.177 molecules/s (+24.4%) and the 32-molecule complex
  workload from 1.963 to 2.222 molecules/s (+13.2%).
- All CPU MCA/site/joint digests matched the previous workflow exactly across
  three repeats.
- Per-process GPU monitoring confirmed that the resident external GPU2 process
  used no SM or memory-engine capacity during the comparisons.
- The final 1+3 production workflow used four rather than sixteen reserved
  CPUs per GPU and delivered 13.520 molecules/s versus the old default's
  14.712 on the full 256-molecule stratified path. The user accepted this 8.1%
  throughput trade-off for a 75% reduction in reserved CPUs.
- The isolated inference path with 1+3 reached 14.994 molecules/s. GPU
  MCA/site/joint digests matched the historical exact values in every run.
- Wrote `implementation_report.md`; raw ai-14 output remains under
  `~/mca_test/results/resource_workflow_bc927fe`.

## 2026-10-22: 64-CPU 8 x 8 validation

- Added a synchronized runner for eight persistent predictors with eight
  disjoint CPUs each, including per-atom reference comparison.
- On the current server, aggregate throughput improved from 17.035 to 19.210
  molecules/s (+12.8%).
- On ai-14, aggregate throughput improved from 17.488 to 20.439 molecules/s
  (+16.9%).
- Each host compared 12,120 atom outputs against the old backend. Maximum
  absolute error, MAE, and RMSE were all zero; output order, ordered site type,
  and `is_Nuc_site` were identical.
- Documented the separate cross-host numerical difference observed under the
  unpinned host environments; it is not an old/new workflow regression.

