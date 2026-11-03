# Stage7E — Canonical stereo storage

Root accepts the CP311 owning scope on 2026-11-03. Canonical isotope/radical/
current-H occurrences now live only in the Atom table, display occurrences only
on Bond, and Molecule retains schema/records plus source-only layout. Whole
`_stereo_payload` and sparse Atom isotope truth are removed. Getters assemble
transport without preparing chemistry or retaining an assembled mirror.

[Decisions](../contracts/decisions.md) define exact values/types/presence/order,
source-H versus current-H, opaque/unbound/duplicate handling and private Record24/
v5 transport. No new production/native type, public stereo schema/signature,
public19/native prefix, H algorithm, model encoding or scientific route was added.

## Actual validation

| Boundary | Actual outcome | Coverage |
| --- | --- | --- |
| CP311 source affected | 2397 passed, 1 skipped | 79 files, every changed test plus connected storage/lifecycle/H/stereo/toolkit/native/FF/MCA contracts |
| CP311 installed affected | Composite 2397 passed, 1 skipped | Same frozen wheel/runtime/tests: original2385passes/1skip plus all12support-load errors repaired and reexecuted |
| Source and installed actual FF | 11 passed each | Real Open Babel build/optimization, retained handles/topology/IDs and geometry |
| Source and installed actual pipeline | 3 passed each | Real CBond→FF→GFN-FF/GFN2, geometry/persistence and state projection |
| Source and installed actual MCA | 1 mixed CLI +3attachment tests each | Input1 succeeds, input2 locally rejected, input3 succeeds; denominator3 and actual values preserved |
| Source native selection | 6 passed, 2 skipped | Molecule-buffer/RDWrapper source contracts; installed SDK cases require external tier |
| Fresh factual native | Build RC0 | Changed fact source rebuilt once; other5unchanged CP311 binaries reused |
| Architecture/storage audit | Passed | 943-file ratchet/224preexisting debt; 458 production files parsed; zero new types or persistent retired owners |

Every run retains its exact commands, source/artifact identities, environment,
results, observations and limits in [the checkpoint](checkpoint.json),
[scientific observations](scientific_observations.json) and the concise
[evidence index](evidence_index.json). Hashes attribute bytes, not chemical truth.

## Failures and corrections

The original affected run retained2387passes/4failures/1skip. Three failures
were stale private-record assertions; the actual MCA isotope-only export failure
required a bounded registered RDKit map, with unknown H/radical conventions still
rejected. A separate actual reproduction showed detached Bond display loss;
removal now retains the exact detached owner and public outer rollback restores
membership/layout/display owners. A source-derived self-invalidation ordering
problem was also corrected and tested, without claiming an original runtime
reproduction. Targeted intermediate failures remain archived.

The first installed run had12load errors solely because its isolated test tree
lacked the Plan17 benchmark support script. Root copied that exact dependency,
verified unchanged product/tests/native/model bindings, reexecuted all12cases and
completed the four actual workflow suites. The original full collection matches
source in ordered membership. No second full installed run is claimed. The
initial cost script syntax error occurred before product imports/measurements;
its original bytes and failure are retained.

## Bounded cost and remaining work

N16/64/256 fixtures include duplicate, unbound and opaque transport. All18cost
cells/90samples complete. Scalar isotope reads are1.29–1.32µs; atN256, getter
0.482ms, same setter3.992ms, scalar-write rollback4.512ms, whole rollback4.289ms
and Atom transport packing2.794ms. Scalar/getter reads make zero molecular
identity capture/check/digest calls. Largest measured isolated Python peak is
155621bytes; this is not total RSS or native memory. Root accepts usable bounded
cost and leaves whole-transport scalar-write optimization in P20-D018. No
historical parity, speedup or global performance claim follows.

Stage7E ownership closes; Stage8A may enter after its own alignment. Other Python
versions, clean all-native/ABI/SDK/sdist,41measured package omissions, broad
scientific/GPU/performance and later Plan20/integrated Plan12 obligations remain
open. The effective goal remains active.
