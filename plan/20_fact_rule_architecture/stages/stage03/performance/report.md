# Stage3 storage, export and marking measurements

The migrated storage has measured gains and regressions. Final efficiency
acceptance remains **open**. [Evidence](evidence.json) binds all original
samples, commands, source/native/runner identities, work counters, confidence
intervals and memory observations. No benchmark or exception policy changed.

## Equivalent-work comparison

The final subset uses B20 `e6082f5`, the current frozen build8 source, the
Stage1 allocation CPU0–63, one BLAS/OpenMP thread, three warmups and ten measured
samples per fresh-process workload. All50 workers succeeded:650 total timing
attempts,500 measured attempts,25 equal-work comparisons. Production/native
bytes were stable, and the maintained comparator reports no environment
mismatch. Internal task tests/builds finished before measurement; the host was
shared and is not claimed to have been exclusive.

| Task; ratio=current/B20 | N8 | N64 | N256 |
|---|---:|---:|---:|
| Scalar fact read |0.915 |1.014 |0.916 |
| Scalar formal-charge write |1.833 |1.878 |1.872 |
| Complete formal-charge batch |2.339 |0.760 |0.563 |
| Full coordinate write/read |0.402 |0.122 |0.082 |
| Five-column float32 export |0.227 |0.033 |0.008 |
| Selected native transport |0.398 |0.126 |0.077 |
| Public factual construction |3.585 |1.343 |0.455 |
| Insert/delete retaining existing rows |7.088 |3.880 |2.867 |

The complete N64 atomic-number-write/required-implicit-H-read task remains
3.111×, with paired95% CI3.076–3.146. It includes the retained eager scientific
body and the new storage/transaction boundary. The result is unresolved; it
does not show that the entire overhead comes from that legacy body or that a
later stage will remove it.

The complete measured Atom-storage family has geometric-mean ratio1.037,
CI1.035–1.041; native export has0.079, CI0.079–0.080. The former remains slower.
Individual regression cases remain explicit. Most families and the complete
end-to-end matrix were not part of this subset. The maintained comparator
keeps those rows incomplete; neither export gains nor a subset aggregate
certifies final parity.

The candidate private selector installs the frozen five-column float32
binding only. The full public19-column float64 property remains a distinct
operation. Native export still uses the sole current implementation until
Stage10's physical move. Earlier unfavorable diagnostics, including the failed
v4 invocation that produced no new samples, remain in the native evidence.

## Actual scaling and memory

Supplemental new-only probes use N8/64/256/4096, CPU0 and the same3+10 sampling
counts. They are mechanism observations with no invented B20 counterparts:

| Operation | N8 | N64 | N256 | N4096 |
|---|---:|---:|---:|---:|
|20 sparse single-UID marks in one transaction, microseconds |48.1 |49.9 |50.6 |51.0 |
| One all-UID mark in one transaction, microseconds |10.7 |27.6 |87.6 |1294.6 |
|20 full19-column float64 exports, microseconds |59.4 |88.3 |183.9 |2107.6 |

The sparse probe touches the same bounded20 inputs as molecule size grows;
dense work and export bytes grow with N. These measurements, together with
physical-writer review and allocator tests, cover the actual mechanism rather
than an unconnected layout prototype. They do not prove every future workload
or substitute for the later lazy-evaluator measurements. The earlier array,
AoS and two-block experiments remain linked as historical selection evidence.

All25 paired worker peak-RSS values were lower for current source, by roughly
1–4MB in processes of roughly194–198MB. This includes interpreter/toolkit
memory and is not a measurement of table bytes alone. Untimed tracemalloc
probes retain their unfavorable temporary/retained allocation observations.

A separate six-process GC-controlled construction probe measures one live
molecule after one warmed construction. Complete19-column numeric projections
match B20 byte-for-byte at all three sizes:

| Atoms | B20 tracked retained bytes | Current tracked retained bytes |
|---|---:|---:|
|8 |7667 |17187 |
|256 |129203 |144419 |
|4096 |2095348 |2175556 |

Current retained memory increases are explicit. Numeric capacity, metadata,
seed-set overhead, post-release GC readings and RSS are recorded separately.
At N4096 the RSS increase is about7.09MB B20 versus5.12MB current, while
tracked retained bytes are higher for current source. RSS page granularity
and Python/NumPy tracking limits prevent treating one metric as a substitute
for the other. Final memory acceptance remains open.

## Stage boundary

Stage3 supplies actual selected-layout, scalar/batch/export, marking-scaling
and memory evidence, with one physical writer and no whole-table scalar
copy or per-Atom full-source digest. Its owning-stage gate is distinct from
the complete final efficiency gate, as documented in the [gate review](../gates/review.md).
The Stage5 eager transition and Stage7 owner changes remain at their assigned
stages. All71 final entries, family/end-to-end results, individual outliers
and memory still require their final reports under the existing policy;
no exception or future performance result is approved here.
