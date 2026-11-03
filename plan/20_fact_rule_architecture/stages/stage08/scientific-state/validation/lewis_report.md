# A11 Lewis applicability evidence — 2026-11-03

Root accepted A11's CP311 complete-H applicability correction and closed whole
Stage8A in its CP311 source and installed owning scope. Status is
`ROOT_ACCEPTED_CP311_LEWIS_APPLICABILITY`; root acceptance and scoped Stage8A
closure are true, while global gate closure remains false. Parent
`c5e4bb02ea8741c6a8f95ca29368d9ba11367f6d` is the committed A10 checkpoint.
Final A11 source freeze SHA-256 is
`1f46ae6eb56a6b9820c83e1248326849841a07876b090e228ce2321755944544`.
Root independently checked 1426 product/test/resource, six native and three build
identities (1435 total); the [root final check](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw/release/root-final-check.json)
binds those checks. Release-time decisions, validation selection and
provider-method documentation are retained exactly in the
[archived release inputs](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw/release/release-inputs); their hashes are indexed.

The existing fixed-bond-main-group-v2 provider now abstains before enumeration
when a non-H site's total-H slot is None, after retaining earlier element/bond
checks. It returns existing NOT_EVALUATED/OUT_OF_DOMAIN with no candidates and
`PROVIDER-001.INCOMPLETE_HYDROGEN_INVENTORY`, naming actual rows. This corrects
P20-D021's documented applicability discrepancy; it does not change counted facts,
charge evidence, arithmetic, enumeration/budgets, method version or optional policy.

First source focus:55passed. First affected:997passed/1failed/5skipped of1003.
The architecture ratchet correctly rejected the new test's legacy `core.Molecule`
import. Root changed that one import to the existing public export; the provider,
checker and allowlist stayed unchanged. Candidate patch/hash and both test hashes
are retained. The wrapper stopped before actual workflows. Original JUnit/logs and
exact debt finding remain in the [original failed attempt01 receipt](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw/source/affected-lewis-attempt01/execution.json) and [architecture failure record](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw/release/architecture-failure-attempt01.json). The failure was an import architecture issue.

Attempt02 source focus:55passed; affected:998passed/5skipped of1003; actual pipeline:
3passed; actual MCA:3passed. The eight new cases verify exact named abstention and
input nonmutation, knownH0 unique(-3,-3), directed positiveH unique(0,-1), explicit-H
neighbor versus unknown site-total, allowed H2 own-None slots and earlier-domain
finding precedence. The five skipped direct-backend tests require `--run-external`
and are not passes; actual pipeline selections use their released flags.

The [final source affected receipt](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw/source/affected-lewis-attempt02/execution.json) records the 1003-node source selection. The [installed attempt02 final receipt](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw/installed/lewis-attempt02/execution.json) confirms the same ordered affected nodes: 998 passed/5 skipped, actual_CBFF_xtb: 3 passed/0 skipped, actual_MCA_attachment: 3 passed/0 skipped. The affected, pipeline and MCA origin checks observed 10, five and one participating Python processes respectively, with no observed leaks; all recorded source, test, native, model, installed, support and context invariants remained true.

| Boundary | Case | CBond donors | UFF final (kJ/mol) | xTB energies (Hartree) |
|---|---|---|---|---|
| source | official-coordination-False | [0, 3] | 69.14934437776031 | gfn2=-14.26981398 |
| source | official-coordination-True | [0, 3] | 69.14934437776031 | gfnff=1.298971744181724; gfn2=-14.27320288 |
| source | official-readme-europium | [2] | 4.510532456453502 | gfnff=3.237789908678307; gfn2=-13.53209753 |
| installed | official-coordination-False | [0, 3] | 69.14934437776031 | gfn2=-14.26981398 |
| installed | official-coordination-True | [0, 3] | 69.14934437776031 | gfnff=1.298971744181724; gfn2=-14.27320288 |
| installed | official-readme-europium | [2] | 4.5084105970772885 | gfnff=3.237790496689996; gfn2=-13.53203276 |

The new actual workflow reports retain scientifically unresolved state separately
from GFN2 backendUHF0 execution default; GFN-FF projection is charge-only. Named
geometry checks pass for these observed runs, with zero ligand-skeleton piercing
and undetermined counts. Compact distances and atom/site MCA observations are in
[lewis_observations.json](lewis_observations.json); full checks, native files and numerical variation remain
in the raw attempts. No fixed historical numerical equality, confidence label or
experimental/ground-state truth is claimed.

A10's counted-inventory and complete/partial/ionic OS observations are reused by
[link to diagnostics_checkpoint.json](diagnostics_checkpoint.json) and its committed archive. This
slice does not rerun or duplicate that probe or the A10 report/archive.

CP311 source/installed scope, shared-site dependencies, six unchanged reused
Stage7E natives and 41 measured package omissions remain explicit. No clean
all-native/ABI/SDK/sdist/manylinux/all-version, global science/GPU, controlled
performance or speed claim follows. Optional activation and later/global gates
remain open. Root acceptance closes the A11 applicability discrepancy and whole
Stage8A only in the CP311 owning scope described above; it grants no optional,
all-version or global qualification.

[lewis_evidence_index.json](lewis_evidence_index.json) provides selective navigation. Root ran
`assemble.py` and archived exact A11 evidence under
`.plan20-artifacts/20261103/stage8a/lewis`. The completed
[raw_manifest.json](../../../../../../.plan20-artifacts/20261103/stage8a/lewis/raw_manifest.json) binds 1603 entries; its SHA-256 is
`07be599fe635f147d4a396376924088c8d46185bd3f344cc5b34f3a22f615e59`.
The original raw-input inventory contains 1596 entries; the completed manifest
also binds the seven archived preparation files. Archived drafts and receipts
retain their original bytes; acceptance is recorded in these maintained records.
