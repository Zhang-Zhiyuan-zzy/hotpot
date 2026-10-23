# Plan 19 implementation log

## Recording policy

This log is updated during implementation, not reconstructed after the work.
Every stage records its intended result, files changed, verification commands,
results, unexpected findings, decisions, residual risks, and Git checkpoint.

The owner authorized implementation on 2026-10-23.  The final tree may contain
one deterministic RI search implementation, one private native topology type,
and no production NetworkX fallback or compatibility alias for a removed graph
or cycle-basis interface.

## Baseline

| Item | Result |
| --- | --- |
| Date | 2026-10-23 |
| Branch | `clean/networkx` |
| Worktree | `/tmp/hotpot-clean-networkx` |
| Starting revision | `2704f3c9e0ca449f0d9d1445d6320714d5dbbc0c` |
| Python | 3.11.16, GCC 14.3.0 runtime build |
| Platform | Linux 5.15.0-139-generic x86_64, glibc 2.31 |
| Native compiler | GCC 9.4.0, C++17 |
| NetworkX | 3.6.1 |
| NumPy | 2.4.6 |
| RDKit | 2026.09.1 |
| Open Babel | 3.2.1 |

## Stage record

### Stage 0 - freeze evidence before deletion

Status: complete

Planned:

- freeze graph, topology, search, MCA-site, cycle-basis, model-ring-tensor,
  dependency, timing, and packaging evidence before deleting any behavior;
- add an implementation-independent exhaustive small-graph cycle-basis oracle;
  and
- establish a source-built test environment tied to the exact starting
  revision.

Performed:

- recorded the environment and exact starting revision above;
- audited all seven production NetworkX imports, runtime dependency entries,
  graph-returning public surfaces, old updater calls, duplicated Atom adjacency
  state, old cycle-basis consumers, force-field rollback state, model/PyG ring
  feature consumers, and active tests/documents;
- added `capture_legacy_semantics.py`, which emits ordered legacy topology,
  DFS, Relevant Cycle, Paton cycle-basis, ring-tensor, RI mapping, and MCA-site
  evidence as JSON;
- added an independent pure-Python oracle for undirected simple-cycle
  enumeration, GF(2) rank, deterministic Paton traversal, and exhaustive
  simple-cycle-basis enumeration; and
- self-validated the oracle on acyclic, triangular, diagonal-square, K4, and
  disconnected graphs.  An additional one-off comparison covered all 1,253
  NetworkX graph-atlas entries with no Paton mismatch;
- rebuilt all four native extensions from the exact starting revision with
  CPython 3.11, GCC 9.4.0, C++17, pybind11 3.x, and setuptools 77-81;
- wrote the exact ordered baseline to `legacy_semantics.json`, including the
  five-cycle cubane Paton basis, six-cycle Relevant Cycle family, ring tensor,
  topology/traversal results, RI mapping order, MCA site order, isotope case,
  and the three accepted MCA prediction hashes already enforced by the test
  suite;
- wrote `baseline_dependency_inventory.md`, classifying every active
  production/import/API/model/test/document consumer as migration, deletion,
  or test-only oracle retention; and
- recorded fixed-input microbenchmarks and targeted retained duplicate-state
  size in `legacy_topology_benchmark.json`.

Verification completed so far:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/graph/test_cycle_basis_oracle.py
```

Result: 8 passed.

Additional focused baseline results:

```text
graph/topology/ring group: 62 passed in 1.22 s
native search/API group: 80 passed, 49 subtests passed in 4.70 s
MCA site/coordination/serialization group: 59 passed in 1.32 s
MCA accepted prediction-digest gate: 1 passed in 6.30 s
```

The frozen 512-atom chain/cubane/search/MCA microbenchmark medians were:

```text
Atom.idx                         9.359429 us
Atom.connectivity               3.315981 us
Atom.neighbours                 3.240479 us
Molecule.bond                   0.454011 us
topology refresh            15720.257000 us
Molecule.copy               61901.120000 us
legacy cubane Paton basis      60.786000 us
cubane Relevant Cycles         75.383000 us
12-carbon pair search         259.833000 us
MCA site detection           1557.816000 us
```

The targeted deep-size measurement for the legacy NetworkX projection plus
per-Atom neighbor and incident-Bond lists on the 512-atom chain was 653,181
bytes.  This is a controlled before/after metric for these specific derived
objects, not whole-process RSS.

Unexpected findings:

- copied native extensions from the main worktree did not match this source
  checkout (`NativeTarget` rejected a missing `degree` field), so they are
  invalid as baseline evidence;
- the first isolated `uv run` selected CPython 3.14 from project metadata; it
  was stopped and its exact `.venv` was removed; and
- authoritative baseline tests therefore require a fresh CPython 3.11 native
  build from this worktree.  That rebuild completed successfully; and
- the broad legacy `tests/test_cheminfo/test_core.py` module contains an
  unbounded `test_atompair` path that invokes 3D construction for a very large
  molecule and is unsuitable as a focused graph baseline.  The graph-relevant
  behavior is instead frozen by the targeted suites and semantic snapshot;
  final closure still runs the maintained repository suite with its normal
  exclusions/constraints.

Adjustment:

- do not record any mixed-source/native result;
- run semantic snapshots and focused suites only after the exact checkout has
  rebuilt its native extensions; and
- keep NetworkX only in test-oracle and historical evidence after production
  cutover.

Git checkpoint: pending commit for Stage 0 evidence.

### Stage 1 - establish single-source Python topology

Status: not started.

### Stage 2 - introduce the private base Graph

Status: not started.

### Stage 3 - cut native search over to the base Graph

Status: not started.

### Stage 4 - native MCA automorphism

Status: not started.

### Stage 5 - cycle algorithms and ordering

Status: not started.

### Stage 6 - remove Core graph state

Status: not started.

### Stage 7 - remove remaining public Graph-like surfaces

Status: not started.

### Stage 8 - direct-target consolidation and optional performance work

Status: not started.

### Stage 9 - dependency and packaging cutover

Status: not started.

### Stage 10 - final closure

Status: not started.
