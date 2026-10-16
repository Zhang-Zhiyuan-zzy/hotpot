# Scientific-state pre-activation resolver baseline

Date: 2026-10-17

Code revision: `87f5296`

This report records a performance and representation baseline. It is **not** a
`COORDINATION_SCREENING_V1` activation report: the profile artifact remained
`planned`, its static activation gate remained enabled, and no code bypassed
that gate.

## 1. Reproduction

```bash
$ python -m tests.benchmarks.scientific_state \
    --structure-registry hotpot/dataset/data/structures/structures.csv \
    --structure-root hotpot/dataset/data \
    --output /tmp/hotpot-state-structures-full-v2
```

| Bound input | Value |
|---|---|
| Registry records | 2280 |
| Registry SHA-256 | `df41f588200f4cd7dd096696350cae7525dd4f7761aa5c541743d99fc8908362` |
| Profile ID | `coordination_screening_v1` |
| Profile version | `1.0.0` |
| Profile status | `planned` |
| Profile content SHA-256 | `574ed32272b9af9ac3f1f2ff879082bd37ba78e4d67f3305a46bf8091bff8ab5` |
| Python | 3.11.16 |
| Deterministic replay | enabled for every record |

The generated `manifest.json`, `records.jsonl` and `summary.json` occupied
approximately 3.3 MiB. They remained under `/tmp`; the tracked benchmark and
the source/profile digests above reproduce them.

## 2. Measured result

| Metric | Result |
|---|---:|
| Source records resolved | 2280 / 2280 |
| Source/resolution exceptions | 0 |
| Replay mismatches | 0 / 2280 |
| Total wall time, including full replay | 43.286 s |
| Primary records per wall second, including replay overhead | 52.67 |
| Resolver-only throughput, primary pass | 375.32 records/s |
| Resolver-only time, primary pass | 6.075 s |
| Maximum traced Python allocation | 221,728 bytes |
| Mean / maximum expanded paths | 0.0 / 0 |
| Resource-truncated records | 0 |
| Automatic selections | 0 |

`python_peak_bytes` is a `tracemalloc` measurement. It excludes Open Babel and
other native allocations.

## 3. Representation audit

| Format | Records | With explicit coordination bonds | With unknown bonds |
|---|---:|---:|---:|
| MOL2 | 190 | 0 | 190 |
| XYZ | 2090 | 0 | 2079 |
| Total | 2280 | 0 | 2269 |

Every record reported both:

- `PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE`; and
- `PROFILE.COORDINATION_BOUNDARY_REQUIRED`.

All 2280 terminal decisions were `INADMISSIBLE`; 2263 candidate searches were
`NOT_EVALUATED`, and 17 were `NO_CERTIFIED_DECISION`. These values are not a
measurement of active-profile accuracy or coverage. The input representations
do not supply the explicit `DATIVE` semantic boundary required by the profile,
and the inactive profile correctly prevents execution selection.

In particular, XYZ cannot preserve bond topology. The present MOL2 registry
also reconstructs metal-ligand edges as `unknown`. Promoting those edges to
coordination bonds merely because a metal is present would invent chemical
semantics and is prohibited.

## 4. Acceptance conclusion

The benchmark infrastructure passes its current acceptance points:

- every source and profile is digest-bound;
- all records are streamed rather than retaining scientific bundles;
- timings, Python allocation peak, path expansion, truncation and typed
  decision states are reported;
- topology preservation is reported by format; and
- a complete second pass reproduced every facts and decision-snapshot digest.

Final Phase 10 profile activation remains blocked. It requires a reviewed,
held-out corpus adapter or facts corpus that preserves explicit coordination
semantics, followed by expert approval of the profile rules. Only then can the
active DecisionKey compiler, automatic-selection coverage and preregistered
activation thresholds be measured legitimately.
