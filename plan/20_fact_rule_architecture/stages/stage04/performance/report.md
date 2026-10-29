# Stage4 focused performance checkpoint

Status: measured regressions remain open. Stage4 is still in progress while a
bounded Sol/max optimization review examines its own view-creation and numeric
delegation costs. No policy, workload or exception budget changed.
[Evidence](evidence.json) retains exact commands, source/native/manifest identities,
all attempts and paired samples.

## Executed equivalent work

B20 `e6082f5` and current freeze-v3 used the same interpreter, inputs, CPU0–63,
one BLAS/OpenMP thread and three warmups plus ten measured samples. The driver
interleaved baseline/candidate order by workload after task tests/builds stopped.
All six workers and78 attempts completed;60 measured attempts have equal actual
work counters, stable production/native bytes and zero environment mismatches.
The host was shared; no exclusive allocation is claimed.

| Frozen workload | current/B20 | Paired95% CI | Peak RSS ratio |
|---|---:|---|---:|
| Graph/search/angles/torsions/pair-distance reads |4.053 |4.040–4.081 |0.993 |
| Numerical segment/cycle screening |0.997 |0.990–1.006 |0.856 |
| Crystal matrix conversion |1.103 |1.098–1.112 |0.856 |

The complete two-entry graph/search/geometry family is2.010, CI2.003–2.022.
The Crystal row is only one of three values/collections entries; that family
is incomplete. Geometry screening is parity, not a measured improvement.
Lower observed process RSS does not resolve either runtime regression or
Stage3's recorded retained-memory increases.

## Remaining work

Investigate the4.053 graph/view and1.103 Crystal costs without restoring
carrier caches, weakening source/stale/constraint/rollback behavior, changing
scientific algorithms or reducing benchmark work. The original measurements
remain visible. Global scientific, all71-workload, end-to-end and overall
memory acceptance remain open under the original policy.
