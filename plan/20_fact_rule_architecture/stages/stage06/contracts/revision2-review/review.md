REVIEW_RESULT
task_id=p20-s6-contract-review-704283-34
revision=2
verification_code=704283
task_hash=2af2b832a8ea05daf98b30d47db18cdb48f9464f171f7bdad806dca87f44579a
source_commit=979455ed8bada61f4d44038935e7f02c1b204c86
diff_or_tree_hash=3e2476a82e925f9c33683bdb1b10361cefff111b0d6a2ad8703fa2eaf238da82
capture_identity_kind=SHA256 canonical capture-manifest.json raw bytes; NOT Git tree
status=BLOCKED_PENDING_ROOT_KEY_ACCEPTANCE_AND_IMPLEMENTATION_EVIDENCE

## Six-findings delta

| Finding | Delta disposition |
|---|---|
| 1. Lock/capture contract | **Resolved statically, root decision pending.** Revision2 eliminates overlapping fresh owner scopes and defines a nonblocking route for every initial/deferred acquisition when distinct-owner alignment begins with any tracked outer transaction, including a third owner. Same-owner capture/projection is reused. This prevents stereo-introduced blocking on another molecular owner in the inspected route, but does not promise general deadlock freedom. See `review.md` §§“Exact capture” and “Lock scopes”, SHA `022afe…`; `decisions.md` S6-R2-04–07, SHA `dd34b6…`. |
| 2. Section 5.3 storage debt | **Resolved statically; implementation deferred to a mandatory gate.** Stage7E now has named owners, rename manifest, zero-new-type constraint, connected migrations and an explicit block before Stage8 and Stage12/global ownership acceptance. Main plan lines 1815–1889, range SHA `5ecae4…`, file SHA `57ede1…`. |
| 3. N09 | **Resolved statically, key acceptance pending.** N09 is now exactly the cap-one three-field `_StereoSnapshot(payload, atom_keys, bonds)`; no `StereoUpdate`, source field, duplicate reservation or retirement credit. Budget SHA `dac7cc…`. |
| 4. Remap owner | **Resolved statically.** `_bond_key`, `_axis_bond_keys` and `_remap_stereo_payload` have one algebra owner; lifecycle mutation enters `operations/chemistry.py` and invokes it once. No structure or facade duplicate remains. Evidence SHA `77988e…`. |
| 5. Public annotations | **Prior premise corrected and design resolved; runtime proof deferred.** Original `stereo.py` already uses postponed annotations and imports `Molecule` only under `TYPE_CHECKING` (`stereo.py:1–70`, range SHA `d954dd…`). Revision2 preserves exact annotation strings, globals, signatures and the actual default/explicit-namespace `get_type_hints` outcomes. It does not add a runtime Core import. |
| 6. Immutable graph/native packing | **Contract resolved; implementation and cost gates remain.** Recursive type, ordered-key, container, scalar and float-byte equality is required, together with one pack per target, no pair/query repacking and real native acceptance/mapping comparison. No such runtime evidence exists yet. |

## Remaining concrete blockers

1. **BLOCKER — Root must accept or revise the two new observable failure contracts.**

   These are the tracked-transaction contention `RuntimeError` and deferred raw-capture drift `RuntimeError`. Their exact triggers, messages, reference/target ordering and late side effects are now explicit, but remain proposed public behavior—not preservation evidence or reviewer-approved behavior.

2. **MEDIUM — The packing-cost gate lacks a pass/fail decision rule.**

   Revision2 specifies fixtures, 3 warmups, at least 10 samples, component timings and allocation/RSS limitations, but only requires observations. Before treating this as a gate, root should freeze either quantitative ceilings or an explicit adverse-result disposition rule. This does not require speedup or parity.

No other internal contradiction is established by the supplied revision2 documents.

## Capture, ordering and locking assessment

The owner-only binding is sound within the supplied source: `_capture_source(mol, ())` and `_check_source` bind only the owner and do not acquire the conformer lock (`_identity.py:247–277`, range SHA `663b65…`).

The proposed raw integrity check covers the consumed payload, atom identity/order, each active/hidden partition identity/order, endpoint identities and exact bond fields. Strong atom, partition and endpoint referents close transient `id()` reuse. Z, charge, aromaticity and H readiness are frozen only at their original first live consumption.

The stale failure is placed at a coherent boundary: after pure preflight and immediately before strict-H/live preparation. Thus malformed captured input still returns its deterministic preflight diagnostics without H, aromatic or native work; a passing snapshot that has drifted cannot be combined with later live fields.

Ordering is preserved as specified:

- Raw reference then target capture precedes mapping callbacks.
- Mapping callbacks run outside stereo-owned scopes.
- Reference projection/native validation remains before deferred noncoordination target preparation.
- `np.asarray`/user `__array__` remains after complete graph validation and global-issue suppression.
- Native mapping runs outside stereo-owned scopes.

Caller-owned outer transaction locks may still be held during callbacks and native work. Revision2 acknowledges this; it does not establish callback lock-freedom. In the inspected helper routes, H/commit work reacquires only the prepared molecular owner, while conformer locking remains that owner’s existing hierarchy. Arbitrary private locks, leaked contexts and monkeypatches remain outside the claim.

## Required future evidence

Before ownership release or Stage6 acceptance:

- Exact actual-turn audit for this revision2 result, followed by separate root key review.
- Runtime source and installed-wheel baselines for exact `__annotations__`, `inspect.signature`, default `get_type_hints` success-or-error behavior, explicit-namespace hints, class identity/module/qualname, canonical tokens and cold pickle/import order.
- Deterministic barrier/timeout tests for same-owner reuse, reentry, reversed-ID inputs, initial and late contention, tracked reference/target/third-owner transactions, mapping/`__array__`/native reentry and late-error side effects.
- Failure-injection proof of one snapshot, strong referents, exact partition drift detection and immutable consumed inputs.
- Recursive packed-target comparison against bound source/wheel fixtures, real native acceptance/mapping equivalence, one-pack counts and bounded cost evidence.
- Full affected Python 3.10–3.14 source/installed-wheel/current-native selection.
- Later complete Stage7E canonical-storage gate.

Revision1’s audit PASS proves only its historical receipt and exact-turn identity. Neither that audit nor this delta review approves the design, production release, implementation, scientific behavior, Stage6 completion or any later gate.