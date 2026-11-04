# Stage14D/E — accepted Crystal/Data transport and delivery

Root accepts Stage14A–E within the declared useful general Molecule/Crystal
transport domain on 2026-11-05. Source, installed support, actual mixed Crystal
CUDA, documentation and explained bounded costs agree. Final combined13/14/15
union validation and delivery remain pending; overall Plan20 is open. Root owns
review/application of these records and the related coherent business commit.
The [checkpoint](checkpoint.json) records exact identities and per-version data.

The accepted source is HEAD `c0eb6c4c134806b2d96827d2fc5cf4352ec8ac69` plus
root's no-commit merge of accepted Stage15 `4407db0`, the reviewed14D repairs
and two maintained test-literal migrations. The943-file source guard, rather
than HEAD alone, identifies the validated bytes; source02 and cost guards agree.
The [current contract](../contracts/decisions.md) and packaged
[scientist-facing examples](../../../../../../hotpot/cheminfo/operations/pyg.md)
define the sole `hotpot.molecule/2` hard cut and ordinary PyG Data. Schema1 has
no loader. Historical14A/schema1 and molecular checkpoints retain their original
source/schema/evidence bounds and are not rewritten by this delivery.

The transport preserves useful supported facts, presence and provenance;
oriented cell, current directed images and occupancy; exact group/setting,
current-key joins and historical maps; stereo/H/constraints; and attributable
observations. Edited tensors remain authoritative. Unknown group differs from
P1, unavailable image differs from zero, and standalone Crystal differs from an
attached empty Molecule. The shared Stage15 codecs and molecular builder remain
the sole owners. No implicit chemistry, symmetry or model calculation runs.

| Qualification | Actual result | Evidence inside the archive |
|---|---|---|
| CP311 source02,49 affected modules | 1880 passed;0 failed/error/skipped;83.538339s; recipes passed3.215410s; guards stable | `delivery/source02/RESULT.json` |
| Outside-checkout installed01, CP310–314 | 1880 passed each;0 failed/error/skipped; recipes passed each;213 product origins each with0 violations; source/site/support/optional guards stable | `delivery/installed01/<abi>/RESULT.json` |
| Fresh sdist02 | 961 members;531 Hotpot Python/stub members verified; required data/docs/licenses/metadata present; no native compilation | `delivery/sdist02/RESULT.json` |
| Five wheels | RECORD verified; six accepted native payloads unchanged in every repack; no new native builds | `delivery/installed01/wheels/<abi>/IDENTITY.json` |
| Actual installed Crystal CUDA | A800 GPU6;5 graphs with3/16/0/4/0 sites;32 synthetic SGD steps; CPU/unbatch/fresh import/trusted save-load passed;172 correct origins and919 unchanged wheel-product files | `cuda/received/{RESULT.json,output/result.json,INSTALLED_ATTRIBUTION.json}` |

Exact test/build/install/docs/runtime/workload argv, cwd and environment are in
the corresponding archived `intent.json` files named by the checkpoint. The
49-module selection is `delivery/TESTS.json`. Source/installed tests cover the
accepted15F/14C affected union and current Crystal boundaries: oriented/skew and
generated cells, current versus historical joins, edited/reordered facts,
periodic/finite outcomes, source/destination isolation, standalone/empty values,
foreign paired images, named local rejection, stereo, frames/results,
constraints, ownership and affected package/import consumers. This is the
recorded affected selection, not a new whole-repository default-suite result.

Installed runtimes were Python3.10.20/3.11.16/3.12.13/3.13.15/3.14.7.
PyG2.6.1 was exercised throughout; Torch2.7.1+cpu on CP310–313 and2.9.1+cpu
on CP314. CUDA used Python3.11.14, Torch2.7.1+cu126 and CUDA12.6. The synthetic
training path establishes general transport usability, not deployed CBond/MCA
inference, scientific validity, convergence or exclusive-device performance.
The five wheel native payload sets are attributed reuse from the accepted
Stage12 native build, not new native builds or standalone SDK qualification. Fresh sdist
SHA256 is `6f92dc71462bbc491fa4a981be1e7dcdccc059205f085f478391d7e67b08f5e9`;
per-wheel hashes and native reuse bindings are in the checkpoint/archive.

Three independently reproduced product defects were repaired in the two
adapters: stereo diagnostic ordering, unnamed required context/equivalence joins,
and silent loss of present molecular declarations on standalone Crystal import.
Nineteen maintained independent boundary cases accompany the repair; its focused
qualification passed86. Original review failures and the reviewer's corrected
Batch-index witness remain in `review/`. Source01 then completed1862 passes and
18 failures, all attributable to two stale test literals:17 obsolete schema1
expectations and1 pre-SpaceGroup/SymmUnit public-export tuple. Those two literal
expectations were migrated without changing product bytes or chemical assertions;
source02 and sdist02 preserve the successful current selection. Source01 and
sdist01 remain archived with their original identities.

Root accepts the following usable current-only complete export→ordinary
Batch→CPU/unbatch→fresh import costs from `cost/attempt02/COST.json` and
`cost/attempt02/RESULT.json`. Each case completed1 warmup,5 timed calls and1
separate same-process memory call, with all postconditions passed and raw samples
flushed before assertions. No outliers were discarded.

| Case | Graphs / total sites | Median ms | Traced peak bytes | Cumulative process RSS KiB |
|---|---:|---:|---:|---:|
| mixed16 | 16 / 32 | 60.301725 | 2251280 | 373372 |
| crystal4x128 | 4 / 512 | 66.88821 | 3064387 | 378916 |
| generated1x128 | 1 / 128 | 201.867601 | 2764866 | 379180 |

The original cost assertion compared fresh `eq=False` MoleculeRecord identities.
Adjacent captures already compared unequal without any workflow, while all24
record fields, copied bond metadata and tracked source identities stayed equal.
No product mutation was found. The original failed script/logs/root result remain
under `cost/attempt01` and `cost/RESULT.json`. Its first case reached1 warmup,
5 timed calls and1 memory call by preserved control flow; original raw timings
and memory were never flushed and are unavailable. The corrected comparator
compares captured facts by value and retains ownership checks; the task-local
repair and successful new campaign do not overwrite that failure.

Costs were measured on the shared3995WX64-core/128-logical-CPU host, affinity
0–127, Torch/BLAS/OMP threads1. Load moved from0.32/1.59/1.42 to0.47/1.56/1.41;
full hardware/frequency/memory snapshots and all timing spikes are retained.
Tracemalloc is a separate phase in the same process and misses much native Torch
allocation; RSS is cumulative with all fixtures constructed up front. Exact
cache/TMP and tooling-PYTHONPATH deltas are recorded in `cost/attempt02/IDENTITY.json`.
No historical baseline, parity, speedup, leak or exclusive-host claim follows.

Foreign schemas require explicit field maps; unknown facts are not reconstructed.
Unsupported self/parallel quotient edges reject without collapsing them. The
qualified portable fields support training and review, not complete live-session
cloning or arbitrary-object persistence. Deployed model inputs remain separate.

The published archive is
`.plan20-artifacts/20261105/stage14/crystal-final-delivery/evidence.tar.gz`,
45869791 bytes, SHA256
`e8e61ed1cf23e63e5acc439a7e1b55c91ef1b9e445cacefadd71187a5e3621b4`.
Root read back and verified all373 payloads against adjacent `INDEX.json`.
Derivable sites/env/tmp/cache/pytest-temp are excluded; exact intents, results,
guards, wheels, sdist and original failures are retained. The successful cost
result is the attempt02 result; the root-level failed result deliberately remains.
This record-writing task reran no validation, imported no product and changed
only the four assigned acceptance/index files.
