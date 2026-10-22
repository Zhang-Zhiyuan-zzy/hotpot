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

- Changed the measured GPU defaults from 9/7 to 1 site worker and 2
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

