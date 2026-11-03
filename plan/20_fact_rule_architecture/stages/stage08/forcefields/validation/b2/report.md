# B2 acceptance separation validation — 2026-11-03

Root accepts the B2 pure acceptance/capture separation within its tested CP311
source/installed scope. The completed `attempt02` evidence and independent archive
review support this checkpoint; whole Stage8B and global gates remain open.

The change moves existing acceptance contracts and pure comparisons into rules,
carrier reads into scoped structure capture, and the sole metal/donor iterator to
that adapter. The operation retains finite early returns, count-first topology,
conditional radii reads, check order, one-scan/checkpoint reuse, late normalization
and fresh mutable public metric dictionaries. Current moved classes/instances
have successful source and installed canonical cold-pickle evidence. There are
zero new/replacement classes and zero net class additions. Static production census:
{"classes": 779, "dataclasses": 357, "protocols": 8, "typed_dicts": 2}.

Selections derive from the released manifest: 12
focused files; 64 affected files, including
all 33 mapped existing Stage8B files.
All reported counts below derive from actual JUnit and agree with runner receipts.

| Boundary | Selection | Actual counts |
|---|---|---|
| source | focused | 283 passed / 0 failures / 0 errors / 0 skipped of 283 |
| source | affected | 990 passed / 0 failures / 0 errors / 7 skipped of 997 |
| source | actual_CBFF_xtb | 3 passed / 0 failures / 0 errors / 0 skipped of 3 |
| source | actual_MCA_attachment | 3 passed / 0 failures / 0 errors / 0 skipped of 3 |
| installed | affected | 990 passed / 0 failures / 0 errors / 7 skipped of 997 |
| installed | actual_CBFF_xtb | 3 passed / 0 failures / 0 errors / 0 skipped of 3 |
| installed | actual_MCA_attachment | 3 passed / 0 failures / 0 errors / 0 skipped of 3 |

Exact ordered source/installed membership comparisons are retained for
affected, actual_CBFF_xtb, actual_MCA_attachment. Measured installed Python modules:
473; reused native binaries:
6; non-Python omissions: 41.
Omission paths and native/source/interface/setup provenance remain in raw evidence.
The current omission count is measured, not copied from B1.
Installed startup/final process observations: {"actual_CBFF_xtb": {"final_processes": 5, "startup_processes": 5, "zero_leaks": true}, "actual_MCA_attachment": {"final_processes": 1, "startup_processes": 1, "zero_leaks": true}, "affected": {"final_processes": 98, "startup_processes": 98, "zero_leaks": true}}.
Source origins cover the main pytest interpreter; installed origin coverage is
limited to participating Python site hooks, excluding site-bypassing/native processes.

B2 runtime attempt history:

- attempt01 / focused-attempt01: 282 passed / 1 failures / 0 errors / 0 skipped of 283; returncode 1. tests.test_cheminfo.test_forcefield_package::test_specialized_apis_are_reexported_without_wrapping: TypeError: cannot pickle code objects

Root's narrow original-parent/current alias diagnosis and the test-only correction
are retained separately from runtime receipts. The generic API test keeps identity
checks for every export and pickle-checks actual function/class exports; it does not
change alias representation or swallow exceptions. Canonical moved-class and instance
cold-pickle checks remain in source and installed validation.

The initial B2 static ratchet failure and exact correction are retained separately:
{"allowlist_growth": 0, "exceeds_frozen_inventory": 0, "new_debt": 1, "unretired_allowlist_entries": 1}.
The correction uses supported string bond kinds with stored-value assertions and
retires only the actually removed coordination module's core.Atom allowance;
its Molecule allowance remains. The post-fix ratchet passed with
220 remaining debt, without checker relaxation or
new exceptions. Original failed and corrected bytes/receipts remain archived.
Any subsequent root runtime repair/release receipts are preserved as their own
actual evidence. B1 loader failure and EOF adjustments remain historical references.

| Boundary | Case | CBond donors | FF final energy (reported unit in observations) | xTB energies (Hartree) |
|---|---|---|---|---|
| source | official-coordination-False | [0, 3] | 69.14934437776031 | gfn2=-14.26981398 |
| source | official-coordination-True | [0, 3] | 69.14934437776031 | gfnff=1.298971744181724; gfn2=-14.27320288 |
| source | official-readme-europium | [2] | 4.50858892791417 | gfnff=3.237789711014277; gfn2=-13.53209914 |
| installed | official-coordination-False | [0, 3] | 69.14934437776031 | gfn2=-14.26981398 |
| installed | official-coordination-True | [0, 3] | 69.14934437776031 | gfnff=1.298971744181724; gfn2=-14.27320288 |
| installed | official-readme-europium | [2] | 4.50829228107412 | gfnff=3.237790497123862; gfn2=-13.53210008 |

There are 6 recorded
source/installed pipeline case manifests. Full per-stage scientific/backend state,
FF and xTB geometry metrics, coordinate-commit behavior and persistence evidence
remain in observations and original reports. Final/stage SDF and listed native
artifact hashes inspected during finalization match their recorded hashes.
Scientific spin remains distinct from the backend UHF execution default; GFN-FF
projection is charge-only. Actual source/installed numerical variation is preserved,
without forced historical equality, ground-state or experimental-truth claims.

MCA observations retain all actual attached/equivalent/empty-site/multiple-site
prediction objects. Exactly equal complete JSON predictions are pooled only in this
compact evidence record; 6 boundary/case/label
raw references remain. No product schema or confidence grade is introduced.

Actual affected-selection skip reasons remain separate by boundary:

| Boundary | Case | Reason |
|---|---|---|
| source | tests.test_cheminfo.obWrappers.test_bridge::test_all_extractant_phosphorus_centers_build_and_optimize_finitely | requires --run-scientific-boundary |
| source | tests.test_cheminfo.test_complexes_build_full::test_readme_europium_complex_full_standard_workflow | requires --run-high-compute |
| source | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_wrapper_matches_direct_cli_energy | requires --run-external |
| source | tests.test_plugin.test_xtb.test_official_integration::test_official_gfnff_wrapper_matches_direct_cli_energy | requires --run-external |
| source | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_optimization_matches_direct_cli_result | requires --run-external |
| source | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_wrapper_matches_direct_cli_for_charged_and_open_shell_states[ionic-chloride] | requires --run-external |
| source | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_wrapper_matches_direct_cli_for_charged_and_open_shell_states[hydrogen-radical] | requires --run-external |
| installed | tests.test_cheminfo.obWrappers.test_bridge::test_all_extractant_phosphorus_centers_build_and_optimize_finitely | the governed extractant dataset is not installed locally |
| installed | tests.test_cheminfo.test_complexes_build_full::test_readme_europium_complex_full_standard_workflow | requires --run-high-compute |
| installed | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_wrapper_matches_direct_cli_energy | requires --run-external |
| installed | tests.test_plugin.test_xtb.test_official_integration::test_official_gfnff_wrapper_matches_direct_cli_energy | requires --run-external |
| installed | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_optimization_matches_direct_cli_result | requires --run-external |
| installed | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_wrapper_matches_direct_cli_for_charged_and_open_shell_states[ionic-chloride] | requires --run-external |
| installed | tests.test_plugin.test_xtb.test_official_integration::test_official_gfn2_wrapper_matches_direct_cli_for_charged_and_open_shell_states[hydrogen-radical] | requires --run-external |

Separately released actual business selections do not convert governed skipped
cases into passes. Equal membership/counts do not imply identical skip reasons.

The checkpoint/index bind the exact freeze, wheel, release inputs, source/installed
receipts, ordered nodes, model/backend/native provenance, raw business outputs and
frozen changed-source capsule. Raw originals are copied and hash-verified. Large
assets, environments/build trees, outside test-source copies and current symlink
aliases are omitted while their identities/provenance remain.

No speed/performance claim, native rebuild, clean-native/ABI/SDK/sdist/manylinux,
all-version, optional activation, broad science/GPU, whole-Stage8B or global
acceptance follows. Root independently reviewed key business evidence and accepts only this B2
checkpoint. Its 1,642 raw files (23,373,111 bytes) and all draft references were
independently verified; original failed-manifest paths resolve to their exact
archived attempt01 bytes. Scientific interpretation and all stated limits remain
as recorded.
