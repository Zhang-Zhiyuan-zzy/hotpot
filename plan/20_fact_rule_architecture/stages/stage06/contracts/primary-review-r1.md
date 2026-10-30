REVIEW_RESULT
task_id=p20-s6-contract-review-704283-34
revision=1
verification_code=704283
task_hash=21ad8134d66f80c4f8df735a6f46e5c6f0f377a8bf3e370527ebf7b10df43698
source_commit=90ff36c2c0b415bb9cf27628265bb4528e425922
diff_or_tree_hash=1bed097223b51033a780f5dc1ae452ffcdc01761bb0e4e52e23dc7ccad33b39f
capture_identity_kind=SHA256_of_canonical_capture_manifest_raw_bytes_not_Git_tree
status=BLOCKED_WITH_REQUIRED_DESIGN_DELTAS_AND_ROOT_DECISIONS

## Severity-ranked blockers

1. **BLOCKER — The proposed whole-operation lock contract is not deadlock-safe as stated.**

   The design holds validate/assign’s owner lock across preparation, pure rules, native iteration and coordinate handling, and holds both alignment locks in increasing `id(molecule)` order. Fresh-lock sorting does not handle inherited locks:

   - `_FactTransaction` already acquires and retains the molecule RLock and joins reentrantly (`structure/_commit.py:81-85,105-160`).
   - Working-copy publication demonstrates reachable internal code holding two molecule locks in argument order, not global ID order (`forcefields/working_copy.py:235-280`).
   - A caller already holding the higher-ID molecule can call alignment, after which alignment attempts the lower-ID lock while another thread can hold the lower and wait for the higher.
   - Proposed lock scope would execute custom `atom_mapping.items()` behavior (`stereo.py:4084-4180`), coordinate `np.asarray`/`__array__` behavior, and `_NativeTarget.iter_mappings` or monkeypatched spies (`1792-1828`) while owner locks are held. These are reentrant-callback boundaries.

   The supplied source does not establish a current production path from working-copy commit into alignment, so that particular path is a risk pattern, not a proven present deadlock. Active outer `_FactTransaction` and custom public inputs are concrete reachable cases.

   Minimal safe delta:

   - Never internally hold two distinct molecule locks simultaneously.
   - Normalize/copy the user mapping and perform coordinate coercion outside owner locks.
   - For each molecule independently, acquire its lock, capture the immutable structural snapshot, run callback-free preflight, perform any required H preparation, capture prepared scalar tuples, then release.
   - Process reference before target to preserve issue order. If both inputs are the same object, capture once and reuse.
   - Run native mapping iteration, coordinate classifiers and report algebra after releasing the operation’s lock.
   - Detect `_current_transaction()` on either distinct alignment input and either fail before acquiring another owner or define and test an explicit transaction-safe route. Arbitrary private preheld RLocks cannot be made safe merely by sorting new acquisitions; no general deadlock-free promise should be recorded.
   - Add bounded tests for same-owner reentrancy, active outer transactions, reversed-ID pairs, custom Mapping/`__array__` callbacks and native-spy reentry.

2. **BLOCKER — Section 5.3 ownership remains unresolved without a concrete later gate.**

   Section 5.3 requires `StereoPayload` to be assembled from stored stereo records/source ledger plus canonical Atom/Bond facts, with `set_stereo_payload()` decomposing atomically (`implementation_plan.md:394-443`). Current storage instead retains the whole payload, including duplicated current-H/isotope/radical/direction transport, and current tests require exact object identity (`test_stereo_level1_storage.py:1-49`).

   The proposal correctly avoids silently changing this behavior during extraction, but “record remaining debt” is insufficient. Before Stage6 implementation release, root must assign an exact later owner—naturally the Stage7 removal-of-second-truth work, or another explicitly named pre-Stage12 gate—with:

   - owned files and source budget;
   - migration of stereo records and source-only ledger;
   - canonical Atom isotope/radical/current-H and Bond direction ownership;
   - atomic get/set decomposition and reassembly;
   - copy/component/merge/pickle/toolkit/native/working-copy migration;
   - lossless `set(get(mol))` and schema tests;
   - removal of whole `_stereo_payload` as stored truth;
   - a maintained gate blocking global architecture acceptance until completion.

   The current `get(mol) is payload` assertion must be classified as a bounded transitional Stage6 preservation test, not a permanent Section 5.3 contract. Stage6 rules should consume an access/capture seam that does not entrench whole-payload storage.

3. **HIGH — N09 is not presently an approved `_StereoSnapshot` reservation.**

   N09 currently reserves a cap-one `StereoUpdate` whose responsibility is “source binding plus existing protected StereoPayload” (`data_object_budget.json:514-535`). The proposed `_StereoSnapshot(payload, atom_keys, bonds)` is a read-only captured-input value and deliberately contains no `_SourceIdentity`.

   The snapshot is technically justified as a carrier-free rule input, but it does not satisfy N09’s current responsibility. Because it produces reports rather than commits, adding a source binding merely to resemble `StereoUpdate` would be unnecessary.

   Required root choice before code:

   - revise N09’s ledger responsibility explicitly to this exact three-field private snapshot, recording its consumers and that no `StereoUpdate` also exists; or
   - retain N09 for the future Section 5.3 atomic update and represent this capture with an already-budgeted ordinary immutable tuple/value or separately approved type budget.

   N09 repurposing is a root ledger decision, not an accepted reservation or retirement credit.

4. **HIGH — Pure remap ownership conflicts with the supplied normative mapping.**

   The design places `_remap_stereo_payload` in `structure/_stereo.py`. The current plan maps pure remap proposals with validation/assignment/alignment to `rules/chemistry/stereochemistry/`, while payload decomposition/remap commit belongs to the structure adapter plus chemistry operation (`implementation_plan.md:1360-1372`).

   The prior inventory called this boundary ambiguous, but the current normative excerpt governs. Root and Sol-max must either:

   - place the pure value transform in an existing rule owner, most naturally `algebra.py`, with lifecycle callers routed through an operation/adapter; or
   - record a specific normative ownership amendment explaining why this schema-preserving lifecycle transform is a structure primitive.

   Do not create duplicate structure/rule implementations or a facade bridge.

5. **HIGH — Exact public annotation preservation is underspecified by the declared DAG.**

   The design preserves five evaluated signatures containing `Molecule`, `NDArray[np.float64]` and `Mapping[int,int]`, but the facade’s declared dependency list omits the modules supplying those runtime annotation objects. Switching to postponed/string annotations would change `__annotations__`, `inspect.signature`, `typing.get_type_hints` behavior and potentially the stated exact contract.

   Specify how the facade safely imports the actual annotation objects after the connected Core rewire, and add that dependency to the designed DAG. Validate source and installed-wheel identity for:

   - full `inspect.signature`, including annotations, defaults and keyword-only parameters;
   - exact `__annotations__`;
   - `typing.get_type_hints`;
   - class `__module__`, `__qualname__`, pickle round trips and `_identity._canonical` tokens.

   The 18 classes may be physically defined once in `structure/stereo.py` while explicitly retaining `__module__ == "hotpot.cheminfo.stereo"`, provided the facade reexports those same objects and cold unpickling resolves them without a cycle.

6. **MEDIUM — Immutable graph exactness and packing cost remain evidence gates.**

   The existing projection builds a plain nested target dictionary with type-sensitive fields, tuple order, `None` objects and consecutive real/virtual labels (`stereo.py:1126-1402`). Replacing its interior with mapping proxies is acceptable only if `_pack_gate6_target()` reconstructs precisely:

   - top-level and nested key order;
   - tuple/list distinctions;
   - integer, Boolean and float types;
   - normalized orders/kinds/aromatic flags;
   - `object=None`;
   - row and virtual-H labels;
   - edge, atom and bond order.

   Dictionary equality alone is insufficient because it can hide type differences such as `True == 1`. Require a recursive type-and-value comparison, existing native acceptance/mapping results, and spies proving exactly one pack per constructed native target and no repack per carrier-pair query.

   Deep-freeze plus repacking adds at least one O(V+E) allocation pass. Later evidence must measure bounded projection/pack time and transient/retained allocations across representative graph sizes. No speedup or parity claim is currently supported.

## Other required design clarifications

- The native callable seam is deterministic only when the operation constructs the sole `_NativeTarget`, supplies a task-local bound callable, invokes it synchronously in frozen pair order, does not retain it, and executes it outside molecule locks. Preserve current tri-state accumulation and early return on the first equivalent mapping.
- Snapshot capture should preserve exact active → hidden-metal → hidden-covalent bond ordering. Endpoint object IDs remain transient lookup keys only; the operation must retain the molecule/atoms strongly until computation completes, and native labels must remain consecutive row/virtual-H integers.
- Preflight ordering is sound if malformed atom facts, missing endpoints, self-loops, parallel edges, nonfinite orders and unknown dative direction return before H preparation, aromatic reads or native construction. Negative prepared-H diagnostics must still precede native construction.
- One successful candidate graph should create one native target per input molecule; coordination-only projection may prepare H without constructing a target; ineligible payloads must perform neither H preparation nor native construction.
- Assignment and alignment must retain exact payload/record sharing and issue order. Moving computation outside locks is safe only after every consumed value has been captured immutably.

## Supported bounded conclusions

The supplied static evidence supports:

- A complete named partition of the original 116 top-level definitions and 23 assignments; this is a source-partition check, not implementation evidence.
- Exactly 18 public class contracts, five public functions and the existing ordered 23-name `__all__`.
- No proposed duplicate public DTOs, Protocols, enums, native ABI mirrors or fallback engines.
- An acyclic declared 14-node stereo-slice DAG.
- One proposed three-field carrier-free snapshot, rather than a general Molecule mirror.
- A coherent separation of contract definitions, structural capture, pure algebra/graph/coordinate rules, operation control, native packing and maintained facade.
- Correct planned spy migration to actual production lookup sites: operation projection, operation validation, existing `_NativeTarget.iter_mappings`, structure remap owner once decided, and adapter packing. No private facade compatibility aliases are needed.
- Preservation intent for current schema fields, enum values, native packing, hidden-bond participation, scientific thresholds and diagnostic ordering.

## Static limits and root/Sol-max decisions

No implemented DAG, class relocation, lock behavior, native packing, allocation result, chemistry result, wheel, matrix or performance evidence exists. The recorded static checks cannot prove runtime import order, pickle resolution, native acceptance or concurrency behavior.

Root and Sol-max must resolve, before ownership release:

1. N09 ledger disposition.
2. Lock/capture/release contract, including active transactions and callback boundaries.
3. Normative remap owner.
4. Exact runtime annotation/import mechanism.
5. Immutable target packing and bounded allocation evidence.
6. Named later Section 5.3 decomposition owner and blocking gate.

No production authorization, public/scientific contract approval, Stage6 acceptance or global ownership waiver follows from this review.