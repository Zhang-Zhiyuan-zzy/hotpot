# Geometry performance evidence

The Segment–Cycle benchmark is an explicit evidence collector, not a default
unit-test gate. It records median and p95 timings without imposing a brittle
hardware-dependent threshold.

```bash
python -m tests.performance.test_segment_cycle_relation_benchmark \
  --output /tmp/hotpot-geometry-benchmark.json
```

The report covers planar and nonplanar 6/8-membered rings, the lazy relation
gate used before repair, the dense relation scan used for frame diagnostics,
surface-enumeration counters, and the tri-state distribution of a curated
geometry set.

## Initial baseline

The first quick run on 2026-09-18 used CPython 3.11.15, NumPy 2.3.5, Linux
5.15, and seven measured repetitions. Times are diagnostic evidence rather
than pass/fail thresholds.

| Scope | Median (ms) | p95 (ms) |
|---|---:|---:|
| Planar 6-membered ring, one segment | 1.334 | 1.401 |
| Planar 8-membered ring, one segment | 1.698 | 1.710 |
| Nonplanar 6-membered ring, one segment | 21.183 | 21.523 |
| Nonplanar 8-membered ring, one segment | 312.748 | 354.966 |
| Lazy frame relation gate, 14 atoms / 14 bonds / 2 rings | 8.027 | 8.142 |
| Dense frame relation scan, 16 candidate pairs | 16.186 | 16.262 |

The curated four-case distribution was one `PIERCES`, two
`DOES_NOT_PIERCE`, and one `UNDETERMINED`; the unresolved case recorded
`surface_construction`. The frame rows isolate the relation work used during
force-field frame observation; they do not include Open Babel minimization.
