# Stage4 focused performance checkpoint

Status: graph/view performance acceptance remains open after the completed
bounded Sol/max optimization and formal follow-up. No policy, workload or
exception budget changed.
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

## Frozen optimization follow-up

The [local optimization](review.md) captures source/UID/geometry once per
locked sequence query while retaining fresh immutable views, exact member
bytes and the existing constraints. Crystal binds its one pure mathematical
function directly. The original measurements above remain visible.

After all new technical tests/builds finished, the same formal driver repeated
the original inputs, work counters, CPU0–63, threads and3-warmup/10-sample policy.
All six workers/78 attempts/60 measured samples completed without failed or
missing outputs; production/native bytes stayed stable and there were zero
comparator environment mismatches. The shared host was not reserved.

| Frozen workload | current/B20 | Paired95% CI | Peak RSS ratio |
|---|---:|---|---:|
| Graph/search/angles/torsions/pair-distance reads |2.054 |2.040–2.065 |0.994 |
| Numerical segment/cycle screening |1.006 |0.999–1.013 |0.855 |
| Crystal matrix conversion |0.994 |0.991–1.000 |0.853 |

The complete graph/geometry family is1.437, CI1.430–1.444: it remains a
regression. Screening and Crystal are parity under the unchanged neutral band;
no useful speedup is claimed. The values/collections family still lacks two
entries. Exact raw samples and artifact hashes are in the evidence's
`optimization_followup`; the original4.053/1.103 results are preserved.

## Remaining work

Resolve the remaining2.054 graph/view cost without restoring
carrier caches, weakening source/stale/constraint/rollback behavior, changing
scientific algorithms or reducing benchmark work. The original measurements
remain visible. Global scientific, all71-workload, end-to-end and overall
memory acceptance remain open under the original policy.
