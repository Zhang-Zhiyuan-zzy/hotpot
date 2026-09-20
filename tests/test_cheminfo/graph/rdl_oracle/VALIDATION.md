# Relevant Cycles differential validation

## Result

The Hotpot implementation exactly matched RingDecomposerLib on every tested
graph:

| Measure | Result |
|---|---:|
| Graphs compared | 2,196 |
| Exact graph-level matches | 2,196 |
| Exact match rate | 100% |
| Reference Relevant Cycles compared | 11,791 |
| `max_size` comparisons | 138/138 matched |
| Incorrect partial result at `max_cycles` | None; the expected exception was raised |

The complete machine-readable result, including binary and source hashes, is
stored in `validation_results.json`. The validated Hotpot commit is
`c7f382e141a7e503336d39b2459cfa6551f68fec`; the report confirms that all three
native source files matched that commit when the extension was built.

## Oracle and corpus

The oracle is RingDecomposerLib commit
`3a7ff93de0d9c4f6a5661508549c6063573f39c7`. Tests call
`RDL_getRCyclesIterator()` and compare complete normalized cycle sets. They do
not substitute SSSR, one minimum cycle basis, RCF counts, or prototypes for
Relevant Cycles.

| Corpus | Graphs | Notes |
|---|---:|---|
| Handcrafted semantic cases | 7 | Triangle, diagonal square, equal-path theta, fused rings, $K_4$, disconnected rings and figure-eight |
| NetworkX named graphs | 11 | Includes cubical, dodecahedral, icosahedral, complete and bipartite graphs |
| PubChem connectivity graphs | 5 | Caffeine (2519), cholesterol (5997), erythromycin (12560), paclitaxel (36314), vancomycin (14969) |
| NetworkX Graph Atlas | 1,173 | Every graph with positive cycle rank in the atlas |
| Deterministic `G(n,p)` stress corpus | 1,000 | 3--12 vertices; fixed seed `20260920`; shuffled edge order and randomized edge direction |

The random corpus uses, for each $n=3,\ldots,12$, 100 samples with
$p\sim U(0.08,\min(0.55,4.5/(n-1)))$. It contributed 4,114 of the compared
Relevant Cycles. Graph Atlas results were also checked against the independent
exhaustive GF(2) oracle in the adjacent test package.

## Timing

The timing run used CPython 3.11.16 on Linux x86-64. After one untimed
correctness/warm-up call to each implementation, each of the 2,196 graph calls
was repeated ten times per implementation. Call order was alternated, producing
21,960 timing samples per implementation.

| End-to-end call | Median | 95th percentile |
|---|---:|---:|
| Hotpot public Python facade + C++ core | 45.667 us | 140.907 us |
| Python ctypes adapter + RDL C core | 64.783 us | 221.561 us |

The median Hotpot/RDL ratio was 0.705: Hotpot used about 29.5% less time, or was
about 1.42 times as fast, in this small-graph mixed corpus. These are
end-to-end Python-call measurements, not isolated kernel timings. They include
input normalization, the native boundary, cycle reconstruction, canonical
ordering and result conversion.

## Reproduction

From the repository root:

```bash
$ python tests/test_cheminfo/graph/rdl_oracle/build.py \
    --clone \
    --source /tmp/hotpot-rdl-source \
    --build /tmp/hotpot-rdl-build
$ python setup.py build_ext --inplace
$ export HOTPOT_RDL_ORACLE_LIBRARY=/tmp/hotpot-rdl-build/src/RingDecomposerLib/libRingDecomposerLib.so
$ python -m pytest tests/test_cheminfo/graph --run-slow
$ python tests/test_cheminfo/graph/rdl_oracle/benchmark.py \
    --library "$HOTPOT_RDL_ORACLE_LIBRARY" \
    --include-atlas \
    --include-random \
    --repeats 10 \
    --output tests/test_cheminfo/graph/rdl_oracle/validation_results.json
```

The verified run produced `44 passed`. The RDL source is not downloaded during
normal tests; live differential tests skip unless `HOTPOT_RDL_ORACLE_LIBRARY`
is explicitly set. Offline tests use `golden_relevant_cycles.json`.

## Scope

This evidence covers undirected, unweighted simple graphs and the published
Hotpot output contract. It is strong differential evidence, not a mathematical
proof for every graph. It does not characterize exponential-output cases near
the default cycle limit or weighted-cycle semantics.

RingDecomposerLib's BSD 3-Clause license is preserved in
`LICENSE.RingDecomposerLib`.
