# Stage5 hydrogen-default implementation

Task `p20-s5-h-default-implementation-02`, 2026-10-30. Implementation is complete and production/test writes have stopped. The frozen final affected focus passed **304 tests**; final static/environment checks passed on the same nine owned file hashes. The parent owns final delta review, broad validation, global records and Git. This is implementation evidence, not Stage5 acceptance.

The existing Stage5 entry record, accepted hydrogen-default decisions and Stage5 contracts control this bounded change. The latest exact source, document, test, native and command identities are in [evidence.json](evidence.json). Earlier267/275-pass runs are preserved as superseded source evidence.

## Behavior and invariants

`Atom.implicit_hydrogens` remains `int`, with the existing setter/int codec, field14 and source encodings. Explicit/source values, including0 and characterized raw float64 bytes, win over defaults. Missing required inputs display the existing scalar/permissive zero placeholder without creating H presence or CURRENT; status remains UNRESOLVED. Stale inferred values are hidden. CONFLICT retains RuntimeError; strict consumers retain the existing unresolved ValueError.

The sole finite rule implements the approved elemental/formal-charge/aromatic/Bond/radical/bracket/no-implicit defaults. Metal raw0 and real-H graph inventory exceptions precede ordinary no-implicit handling. The ninety named scientific table cases pass. Metal classification, the other valence/formal-charge algorithms and aromatic/Kekule assignment are preserved.

The effective payload binding consumes UID, radical electrons, current bracket H and current no-implicit. Current transport implicit count and immutable source history are not additional count owners. Consumed-input payload changes, dative direction and changed retained-install bindings seed the existing groups atomically; history/isotope/descriptor-only changes do not rerun H.

Automatic inferred ordinary nonmetal addition captures all selected `(T,E,J,producer,eligibility,inputs)` before any graph write and validates the original complete source. Both public entries share `A=max(T-E,0)` and the existing current-H update path. Existing current implicit/bracket transport becomes0 even when A=0; exact no-implicit None/False/True, raw inferred T/producer/presence, source-ledger bytes, isotope and radicals remain. Real H/metals, supplied counts and explicit `num` retain their characterized paths. No current-H record is synthesized solely to hold zero.

The outer factual transaction covers graph creation, payload replacement, target/source/presence, source checks, dependency state and final topology. Tests exercise later-row, final-topology and zero-residual cleanup rollback while retaining normal burned-UID semantics.

## Changed ownership

| File | Assigned change |
| --- | --- |
| `hotpot/cheminfo/rules/chemistry/bonding/hydrogen.py` | One approved finite default; private rule now takes eleven inputs. |
| `hotpot/cheminfo/structure/_identity.py` | Minimal effective H transport binding, shared H extraction and retained-install bindings. |
| `hotpot/cheminfo/operations/chemistry.py` | Display/readiness boundary, proposal binding, strict consumer preparation and shared inferred expansion/current-H cleanup. |
| `hotpot/cheminfo/structure/_commit.py` | Existing payload/Bond writers seed consumed H inputs with rollback. |
| `hotpot/cheminfo/core.py` | Record installer invalidation, Atom outer add-H transaction, detached validity and strict aggregate/chemical consumers. |
| `hotpot/cheminfo/stereo.py` | Existing graph projection prepares strict counts once. |
| `tests/test_cheminfo/structure/test_stage5_hydrogen_defaults.py` | Scientific, binding/display, writer/installer, expansion, source-preservation and rollback regressions. |
| `tests/test_cheminfo/structure/test_stage5_bonding.py` | Retained isolated-H test now asserts the approved real-H0 CURRENT fact and H2 retry. |
| `tests/test_cheminfo/test_plan20_carrier_cutover.py` | Two private spies accept eleven inputs; retained endpoint rollback assertions use the actual lazy-preparation transaction boundary. |

No native source/binary changes, carrier schema, public DTO/type/group or misspelled alias were introduced by this owner. The two maintained records in this directory document this implementation. Other dirty workspace files belong to shared Stage5 work.

## Complete post267 delta

The sole `_hydrogen_inputs` implementation moved from chemistry to structure identity. The retained-dependency installer now compares complete local H inputs, Bond/neighbour UID adjacency and H presence/source/raw float64 bytes for retained UIDs. Eight regression cases cover six changed-input families, unresolved-to-supplied0 and source-only history. That source passed275 tests.

The reviewed zero-residual defect was then reproduced: before production repair, six Atom/Molecule cleanup cases retained current implicit91, and two rollback injections never entered cleanup; twenty-one exception-preservation cases passed. The narrow repair captures ordinary nonmetal eligibility and runs current-H cleanup independently of positive graph additions. Twenty-nine new cases cover both entries and all policy states, direct real-H/metal exceptions, supplied/legacy/explicit-num0 preservation, idempotency, exact source-ledger bytes and whole-entry rollback. The existing batch dirty-injection spy is restored before its repeat call; its original first-pass and repeat graph-count assertions remain.

Only `core.py`, `operations/chemistry.py`, `structure/_identity.py` and the new hydrogen-default test changed after267. Only chemistry and that test changed after275. The full review patches are `.plan20-artifacts/20261030/stage5/hydrogen-implementation/final-post267.diff` and `final-post275.diff`; exact patch and final-file hashes are recorded in the handoff manifest/evidence. The finite rule, commit writer, stereo projection and two historical test files did not change after267.

## Executed evidence

All attempts retain exact argv/environment/cwd, source/test/native hashes, source snapshots and logs under `.plan20-artifacts/20261030/stage5/hydrogen-implementation/`. Focused attempts additionally retain the selected test inputs; the final snapshot includes the shared fixture file. Every listed execution reported `frozen_unchanged=true`.

| Attempt | Actual result | Recorder elapsed |
| --- | --- | --- |
| `focused-attempt1` | 17 failed,222 passed; fixture kwargs, source-seed sampling, detached/import behavior and historical private tests corrected as recorded. | 4.526503955945373s |
| `focused-attempt2` | 4 failed,258 passed; detached freeze timing, explicit reference-toolkit policy and lazy transaction expectation corrected. | 2.7709376104176044s |
| `focused-attempt3` | 267 passed; superseded by retained-install completion. | 2.6406648978590965s |
| `focused-attempt4` | 275 passed; still contained the reviewed A=0 defect. | 2.67697836086154s |
| `zero-residual-regression-before-fix` | Expected red:8 failed,21 passed,167 deselected. | 1.6041052285581827s |
| `focused-attempt5` | 1 failed,303 passed; all29 new cases passed; existing first-pass spy still asserted dirty state on repeat. | 3.0313545875251293s |
| `focused-attempt6` | **304 passed in2.22s**, no exclusions, final source. | 2.824779950082302s |
| `final-static-and-environment-attempt4` | AST all9, new-file whitespace and assigned `git diff --check` PASS; same final file hashes. | 0.8581587597727776s |

The initial static helper failed because it checked unrelated legacy whole-file whitespace in core.py. Only the helper scope was corrected. Subsequent static attempts2/3 passed on superseded267/275 source; attempt4 is the final source evidence. All failures and results are retained.

The final focus was executed through the unchanged recorder:

```text
/tmp/hotpot-plan20-env/bin/python .plan20-artifacts/20261029/stage5/local/independent-checkpoint-attempt4/recording_runner.py .plan20-artifacts/20261030/stage5/hydrogen-implementation/focused-attempt6.request.json
```

Its exact test command was:

```text
taskset -c 24-31 /tmp/hotpot-plan20-env/bin/python -m pytest -q tests/test_cheminfo/structure/test_stage5_hydrogen_defaults.py tests/test_cheminfo/structure/test_stage5_bonding.py tests/test_cheminfo/test_plan20_carrier_cutover.py tests/test_cheminfo/kekulize/test_legacy_api.py tests/test_cheminfo/structure/test_atom_fact_table.py::test_raw_numeric_domain_is_preserved_separately_from_scalar_codecs
```

Environment overrides: `LD_PRELOAD=/tmp/hotpot-rdkit-toolchain/lib/libstdc++.so.6`, `PYTHONPATH=/home/zhangzhiyuan/hotpot`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=0`, and OMP/OpenBLAS/MKL/NumExpr threads1. Actual modules: Python3.11.16, NumPy2.4.6, RDKit2026.09.1, Open Babel3.2.1. Existing cp311 fact native SHA256 is `77473313ae381af6818a5d4b5c2e940d54a44cc0edf36855a68cdf03aa80327a`.

The final request, logs, production/test snapshots, exact nine-file SHA map and matching-input confirmation are recorded in `evidence.json` and `.plan20-artifacts/20261030/stage5/hydrogen-implementation/final-handoff.manifest.json`. All five formerly excluded historical function IDs are included in the final run. Strict scientific xfails were not removed.

## Remaining gates and limits

The parent will obtain delta-only independent re-review of the frozen source, including the reviewed A=0 repair, then perform broad default/native/five-ABI/wheel/matrix/science/performance gates and update global records. The task12 audit match does not turn its BLOCKED content into source approval; final re-review remains open. This owner has no remaining local functional blocker.

Toolkit/model field14 normalization remains an unapproved separate proposal. Inherited total-versus-omitted H inconsistencies in broader toolkit/SMARTS/stereo consumers and existing aromatic/Kekule limitations remain characterized and unnormalized. This bounded source focus does not claim those broader consumers or any native/wheel/matrix/performance gate are accepted.
