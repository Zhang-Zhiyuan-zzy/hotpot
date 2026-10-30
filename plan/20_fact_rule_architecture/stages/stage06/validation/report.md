# Stage6 executed validation

Recorded date: 2026-10-30. Root accepts frozen C61 technical and preservation evidence
in the [final root decision](../../../../../.plan20-artifacts/20261030/stage6/root-gate-decision/attempt1/evidence.json).
This report integrates that decision; validated Git delivery and root's final
record/source check and Stage6 closure remain pending. Stage7A has not entered.
The [machine-readable record](evidence.json) binds exact
commands, artifacts, outcomes and qualifications.

## Validated source and execution boundary

Validated source: `d46fbc1d1316a2d34d2ef4c2eb289edff5a81e2a` on
`implement/plan20-fact-rule-architecture`, tree `3a48d3bb2949f0f8aa11b6ea3dff35cbc97ed198`;
4,129-file snapshot value SHA256
`782466b4ab6f1c8091633a4a3458ea9b42dd19d591f1b607597239f6631cb4c3`.
Documentation-only delivery does not replace this runtime source identity. The
[SOURCE_READY release](../../../../../.plan20-artifacts/20261030/stage6/validation-execution-attempt1/SOURCE_READY.json)
and [original FREEZE](../../../../../.plan20-artifacts/20261030/stage6/validation-execution-attempt1/FREEZE.json)
remain unchanged. Root ended the runtime freeze after the final cost decision;
the original artifact is preserved.

Selected revision4 runner SHA256:
`aca4a00ee35c0188e80c5bd496bcf65595110ea575d169b62098732a8236de49`.
The original143-path native closure retains142 strict unchanged inputs and the
one qualified Python `hotpot/cheminfo/_native/buffers.py` import/accessor rewire:
historical `a916c72a5872c59e19a07de0443aca557f97adec287305682cdd45cd67c9a9d2`,
candidate `60202219c365993884b5b77fad11954d5fd89bf33b2fe4e2b0a5f644f2c2886f`.
Each wheel reuses13 native pieces from its accepted Stage5 wheel. These are
current Python assemblies/execution against bound binaries; no native rebuild.

## Focused and five-version technical results

Focused Python3.11 passes65 tests with zero failures/skips. Its three files are
`test_stereo_api.py`, `test_stereo_extraction.py` and
`structure/test_buffer_ownership.py` under `tests/test_cheminfo/`. The retained
[actual focused command](../../../../../.plan20-artifacts/20261030/stage6/validation-execution-attempt1/focused-cp311-attempt1/source-run-attempt1/command.json)
records a4.576901959-second pytest child. The
[execution index](../../../../../.plan20-artifacts/20261030/stage6/validation-execution-attempt1/execution-evidence-index-attempt1.json)
binds every exact argv, cwd, resolved interpreter, command receipt and elapsed
time. Order: focused3.11; matrix3.11,3.10,3.12,3.13,3.14; default3.11. All recorded
source snapshots remain unchanged.

| Python | Source:46 files | Installed:46 files | Native-source:3 files |
|---|---:|---:|---:|
| 3.10 | 1,397 passed | 1,396 passed | 6 passed /2 external SDK skips |
| 3.11 | 1,397 passed | 1,396 passed | 6 passed /2 external SDK skips |
| 3.12 | 1,397 passed | 1,396 passed | 6 passed /2 external SDK skips |
| 3.13 | 1,397 passed | 1,396 passed | 6 passed /2 external SDK skips |
| 3.14 | 1,397 passed | 1,396 passed | 6 passed /2 external SDK skips |
| Total | 6,985 passed | 6,980 passed | 30 passed /10 external SDK skips |

The [root matrix review](../../../../../.plan20-artifacts/20261030/stage6/root-runtime-review/attempt3/matrix-review.json)
checks source/installed origins, frozen node rosters, signatures, qualified
transport and per-version wheel/native provenance. Source and installed results
remain separate. Exactly one installed source-reading node is deselected by
the accepted policy; it executes in every source tier:

```text
tests/test_cheminfo/rdWrappers/test_coordination_embedding.py::test_native_sampler_has_no_business_forcefield_entrypoint_calls
```

No ordinary required skips occur in these technical tiers. The two native SDK
link/run nodes for ABI0/ABI1 skip in every version because the external SDK binding
is unavailable. Those10 skips remain missing evidence. The two earlier root
artifact-reader failures remain retained; the successful third reader supplies
artifact review, not a new test execution.

## Default integration and artifact-only recovery

The original Python3.11 pytest child returned0 in267.226480996 seconds:
4,998 passed,123 skipped,3 strict xfailed, plus49 passed subtests. There are5,124
ordinary XML testcase elements and5,173 JUnit suite-summary entries. The original
driver returned1 because its verifier required suite-summary count to equal the
ordinary frozen-node count. Its failure, logs, XML, observations and helper bytes
remain unchanged.

The [strict recovery](../../../../../.plan20-artifacts/20261030/stage6/default-evidence-recovery/attempt1/evidence.json)
reads retained artifacts only. Root's
[separate acceptance](../../../../../.plan20-artifacts/20261030/stage6/root-runtime-review/default-recovery/acceptance.json)
accepts the original execution after exact ordinary setup/call-or-skip/teardown,
XML identity/order and all1,397 required Stage6 passes reconcile. The49 additional
successful call reports belong to
`tests/test_cheminfo/test_search.py::TestSmartsSearch::test_smarts_search`.
The sole `unittest.subTest` loop emits49 successful subtest calls and one successful
parent call. JUnit stores one testcase reporter per ordinary node while counting
successful calls in the suite summary. Both denominators remain visible. No
pytest rerun or frozen-helper weakening occurred.

The original observer flattened individual subtest report classes and parameter
contexts. Recovery supports aggregate attribution rather than retrospective
per-context attestation. Installed pytest source and RECORD are bound at recovery
time, not before the original execution. These limits remain part of the accepted
technical evidence.

## Preservation, public contracts and workflow observations

All15 [supplementary commands](../../../../../.plan20-artifacts/20261030/stage6/validation-execution-attempt1/supplementary-evidence-index-attempt1.json)
returned0 with unchanged freeze identities. Their index binds362 outputs; the
[root supplementary reader](../../../../../.plan20-artifacts/20261030/stage6/root-runtime-review/supplementary/attempt2/review.json)
verifies719 bindings, all90 metric comparisons and zero exclusions. Each source
and installed mode covers TH,DB,AX,coordination-SP,virtual-H,hidden-bonds. Four cold
orders—facade-first,core-first,identity-first,unpickle-only—each preserve all18
public classes and40 values, class/instance pickle identities and canonical module
tokens. The earlier historical-origin rehash reader failure remains retained.

The18 public types have one definition; ordered23 exports and five facade
signatures retain keyword-only shape, defaults and postponed annotation strings.
Default `get_type_hints` still raises `NameError` for `Molecule`; an explicit
Molecule namespace resolves in source and installed mode. The root-approved
contention/drift RuntimeErrors retain exact triggers, order and late side effects
in the [accepted decisions](../contracts/decisions.md). Executed tests
cover stale capture, distinct/third-owner transactions, same-owner reuse, rollback
and callbacks outside fresh stereo-owned scopes. They do not establish general
callback or lock deadlock freedom.

Recursive native targets match in type identity, ordered keys, container kind,
scalar values and float bytes. Real native acceptance/mapping output match
retained inputs. Observations show one capture/check/projection/pack/native target
and query per actual workflow, complete mapping consumption, and no pair/query
repacking. Operation-local weak references release; fixture carriers deliberately
remain live in the harness. Timing/allocation and separate component records are
assessed in the [performance report](../performance/report.md).

## Coverage limits and remaining delivery

Scientific algorithms, Stage5 H defaults, transport schema/order/source contracts
and native ABI remain preserved within this extraction. These tests/fixtures are
technical preservation evidence; no new global scientific monitoring report or
success-count criterion is introduced. Cold success in selected existing
environments does not close final optional-dependency/lazy-import acceptance.

Stage7A needs its post-Stage6 source/import/test/wheel rebind before design or
implementation release. Whole `_stereo_payload` storage and `get(mol) is payload`
remain transitional until mandatory Stage7E; Stage8 and global ownership
acceptance remain blocked on that gate. Stage10 clean-native/ABI/SDK/sdist,
global science/performance/hardware/Plan12 acceptance, and the Stage0–12 `B_exp`
prerequisite for separately gated Stage13A/B remain open. Root owns coherent
validated Git delivery and final Stage6 closure after checking records against
C61. This integration ran no Hotpot import, test, build, model, benchmark or new
review/audit request.
