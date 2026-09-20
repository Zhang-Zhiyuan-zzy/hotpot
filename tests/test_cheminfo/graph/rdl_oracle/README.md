# RingDecomposerLib differential oracle

This directory contains test-only infrastructure for comparing Hotpot's
Relevant Cycle implementation with RingDecomposerLib (RDL). It is not imported
by the production package and must never become a runtime fallback.

The oracle is pinned to RDL commit
`3a7ff93de0d9c4f6a5661508549c6063573f39c7`. The adapter calls
`RDL_getRCyclesIterator()` and compares the complete, expanded Relevant Cycle
set—not an SSSR, cycle basis, RCF count, or prototype set.

## Build and run

The normal test suite performs no network access. Build the optional oracle
explicitly in disposable directories:

```bash
$ python tests/test_cheminfo/graph/rdl_oracle/build.py \
    --clone \
    --source /tmp/hotpot-rdl-source \
    --build /tmp/hotpot-rdl-build
```

The final line printed by the command is an `export` statement. Apply it, then
run the live differential tests:

```bash
$ export HOTPOT_RDL_ORACLE_LIBRARY=/tmp/hotpot-rdl-build/src/RingDecomposerLib/libRingDecomposerLib.so
$ pytest tests/test_cheminfo/graph/test_rdl_differential.py --run-slow
```

Without `HOTPOT_RDL_ORACLE_LIBRARY`, live RDL comparisons are skipped. This is
intentional: a test run must not download or execute mutable upstream code.

The checked-in golden corpus remains available offline. It contains hand-built
semantic cases, symmetric named graphs, and five pinned PubChem connectivity
graphs. Live tests additionally cover all 1,173 cyclic graphs in NetworkX's
Graph Atlas and 1,000 deterministic `G(n, p)` stress graphs with 3--12
vertices. The PubChem records preserve CID and Connectivity SMILES in
`corpus.py`; they were retrieved through PUG REST on 2026-09-20.

Regenerate the golden file only with a verified pinned build:

```bash
$ python tests/test_cheminfo/graph/rdl_oracle/generate_golden.py \
    --library /tmp/hotpot-rdl-build/src/RingDecomposerLib/libRingDecomposerLib.so
```

For an accuracy and timing report after building Hotpot's native extension:

```bash
$ python tests/test_cheminfo/graph/rdl_oracle/benchmark.py \
    --library /tmp/hotpot-rdl-build/src/RingDecomposerLib/libRingDecomposerLib.so \
    --include-atlas \
    --include-random \
    --output /tmp/relevant-cycles-benchmark.json
```

## Provenance and license

RingDecomposerLib is Copyright (c) 2016 University of Hamburg, ZBH, Niek
Andresen, Florian Flachsenberg, and Matthias Rarey. It is distributed under the
BSD 3-Clause license reproduced in `LICENSE.RingDecomposerLib`.

The adapter contains no copied RDL implementation. Golden outputs, when used,
are generated with the pinned commit and record their graph input and upstream
commit. RDL itself remains an optional development oracle.

Relevant references:

- Vismara, P. *Union of all the Minimum Cycle Bases of a Graph* (1997).
- Flachsenberg, F.; Andresen, N.; Rarey, M. *RingDecomposerLib: An Open-Source
  Implementation of Unique Ring Families and Other Cycle Bases* (2017).
