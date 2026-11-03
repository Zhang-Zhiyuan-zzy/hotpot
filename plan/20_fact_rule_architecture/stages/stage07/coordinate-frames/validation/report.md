# Coordinate-frame ownership: CP311 acceptance

Root accepted Stage7D's applicable owning scope on 2026-11-03. The [completion](../completion_20261103.json), [checkpoint](checkpoint.json), [business observations](scientific_observations.json) and [evidence graph](evidence_index.json) record the tested boundary. Stage7E and the remaining Plan20/Plan12 gates stay open.

Current Cartesian coordinates have one owner: the Atom table. Explicit source frames use immutable N16; generated ensembles use N17 with the existing sealed ConformerBatch; N18 holds shared, attributed observations. A bound Conformers stores only its owner, while existing standalone collections retain their independent domain. Public constructor/read/add/clear spelling, signatures and ordinary values remain. No assembled frame arrays or scientific attribute shadows remain on the bound projection.

Source insertion preserves current coordinates and the selected member. Loading selects coordinates and the corresponding observations atomically; generated applicability excludes starting geometry, with historical calculation inputs retained separately. Actual geometry/representation/cell/constraint changes invalidate only subscribed current results. Copy/components preserve attributable source values and drop generated results; fragment whole-system observations become retained source evidence, not fragment energies. Record25/tagged-v4 carries canonical source/generated/selection state and explicitly rejects oldv3. All affected publication, UID mapping, copy, pickle and rollback routes use those owners.

## Executed evidence

Exact commands, cwd, environment, interpreter, dependency/native/model identities, elapsed time and origin checks are in the linked receipts. Counts below describe actual executions rather than coverage targets.

| Selection | Source passed / failed / skipped | Installed passed / failed / skipped |
| --- | ---: | ---: |
| Final focused10files | 363 / 0 / 1 | Included in affected selection |
| Complete affected65files | 2105 / 0 / 1 | 2105 / 0 / 1 |
| Actual Open Babel FF | 11 / 0 / 0 | 11 / 0 / 0 |
| Actual CBond→FF→GFN-FF/xTB | 3 / 0 / 0 | 3 / 0 / 0 |
| Actual MCA mixed-input CLI | 1 / 0 / 0 | 1 / 0 / 0 |
| Actual MCA attachment | 3 / 0 / 0 | 3 / 0 / 0 |

Final freeze `2399687c…` binds1411 source/test/resource files and six CP311 native artifacts. Source and installed affected ordered node IDs agree. The outside-checkout wheel contains458 Python sources; its SHA256 is `8d08e16f1520764e2d64850162b3d4c2369233c3dd002b699414c00ab9fc9324`. Product/test/model/native bindings remained unchanged and recorded origin checks found no product leak. One fact-native was freshly rebuilt for F3-C1, with exact source/build/binary receipts; five binaries were reused. Forty-one measured non-Python package omissions and shared system-site dependencies remain explicit limits. This proves neither a clean all-native/ABI/SDK/sdist build nor other Python versions or full-default coverage.

The final architecture audit passes across940 Python files, with224 preexisting debt entries and no new ratchet, layer or carrier-definition findings. Production census783classes/360dataclasses/8Protocols/2TypedDicts adds exactly reserved N16–N18 and moves Conformers once; no retirement credit or native schema addition is claimed.

## Original failures and repairs

All original attempts and freezes remain. The first focused run exposed a real native constraint-only callback gap. F3-C1 repairs effective Boolean change detection and callback preparation before the chemistry-mask early return, through existing native Undo. It introduces no ABI/schema, tensor-encoding or scientific-algorithm change. A later affected run exposed empty-ensemble callback work and replacement of a retained bound view; both were repaired. Tests also migrated exact recursive array snapshots, private Record-v4 expectations and shared owner-lock checks. A renamed-node selector failed with RC4 before execution; it was corrected without a business change.

Installed attempt01 retained six cold-pickle failures caused by an omitted copied-test root in child PYTHONPATH. The test now supplies that path alongside the exact installed product root, retaining the product-origin assertion. All25record fields, UID joins, presence/provenance and nested bytes remain checked.

A final independent read-only review found two further actual defects, reproduced by root before repair: normalizing a Bond's raw truthy constraint invalidated unchanged Boolean applicability; negative-index source selection followed by append caused whole copy to lose a current observation. The repairs compare effective Boolean constraint values and rebase cursors when inserting source rows or dropping generated rows during source-only copy. Thirty added regression cases cover noops/actual flips, positive/negative source/generated selections, absent observations and exact outer rollback. Final source and installed selections passed those cases. No broad catch, retry, fallback or additional result store was added.

## Chemical observations and limits

Every pipeline boundary retains three cases and one final record per case. FF and xTB standard configured geometry checks passed. Scientific-state resolution remains `unresolved`; method-specific backend projection does not determine physical spin. Zn uses charge2/GFN2 UHF0; Eu uses charge3/GFN2 UHF0. GFN-FF uses charge-only projection with UHF unavailable. The explicit Zn FF seed is7; no Eu seed is inferred.

| Case | Source UFF energy, kJ/mol | Source xTB energies, hartree | Installed UFF / xTB observations |
| --- | ---: | --- | --- |
| Zn/NCCO, GFN2 singlepoint | 69.14934437776031 | −14.26981398 | Same measured values |
| Zn/NCCO, GFN-FF→GFN2 singlepoint | 69.14934437776031 | 1.298971744181724; −14.27320288 | Same measured values |
| Eu/acetate, GFN-FF→GFN2 optimization | 4.508910798742344 | 3.237789742459551; −13.53210129 | 4.508188304189914; 3.237789662524237; −13.53209991 |

These are actual method/state-specific observations, not accuracy claims or exact-equality gates. Small Eu differences remain visible under existing numerical/stochastic contracts. Both GFN-FF logs retain `Runtime exception occurred` / `gfnff_setup: Could not read topology file.` together with fresh topology initialization, convergence and normal termination. Root reuses the accepted predecessor control-flow interpretation of an optional restart read, without inferring the exact prior-file cause. No warning suppression, historical loader-trace claim or new Hotpot fallback follows.

MCA retains denominator3 and original order: CN prediction, local charged `C[NH3+]` rejection, CO prediction. Printed CN values are C308.75/N489.50 and CO C283.50/O318.25kJ/mol. Outer pytestRC0 verifies the expected CLI RC1; the inner diagnostic/RC was not separately persisted. Attachment tests retain actual prediction JSON, Atom/site joins, the exact selected payload and independent returned dictionaries. Values remain local FP16 CPU model observations without invented confidence labels.

Gaussian existing-log parser/import/update passed on source and installed boundaries. The updater publishes complete observations after geometry/declarations and explicitly selects complete supplied Mulliken charges through the sole charge owner, closing P20-D016's tested producer gap. Earlier unobserved charges remain unavailable, explicit zero survives, and unequal histories retain raw values with named unmapped findings instead of fabricated row associations. This addresses only D011-B4; B1–B3 remain. The official Gaussian executable test was skipped because it requires `--run-external`; no engine execution or installation-status claim is made. Parser assertion targets are identified separately from persisted business DTOs.

## Bounded costs and later work

Six fixtures use16/64Atoms and8/64/256frames, five samples per measurement:60measurements/300samples, zero failures. Scalar medians remain about12microseconds across these sizes. At64Atoms/256frames the larger observed source/generated medians are3.59ms for all energies,6.39ms for coordinates and7.33ms for complete publication/rollback. Reads make zero source capture/check/digest calls. The bound view retains no arrays; source observations are shared, generated coordinates have one batch owner, and selection shares its observation payload. One largest coordinate projection returns about395KB with about837KB Python-traced peak; this is not total RSS/native-memory accounting.

Root accepts bounded usable work and observed scaling on the declared shared host. There is no paired historical speed/parity result or global performance pass. P20-D017 indexes row-projection optimization for consolidated Stage12 work. Stage7E remains mandatory beforeStage8. Other Python versions, clean all-native/ABI/SDK/sdist/platform delivery,41asset omissions, broader scientific/GPU/default and global-efficiency evidence remain open.

The ignored finite local archive retains1255exact-byte files/35,638,261bytes, including original failed receipts/freezes, business text/numeric output and cold pickles. Git stores this compact evidence graph; environments, model/extension/wheel binaries and whole pytest trees are excluded.
