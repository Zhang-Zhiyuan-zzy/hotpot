# Stage9B bounded cutover evidence

Root accepts the bounded sole-owner carrier cutover at the tested CP311 source and installed boundaries. Selected checks completed without failures. Stage9 remains open pending MolBundle; later and overall gates remain open.

The source parent is `958bdd9b4757160aa57c4dc8eef91a8bb71587d5`. Final R5 freeze SHA-256: `de1779e072f764cab4c7cda87335d5cce70ac24578de823bd1977dfbdcc01ed5`.
R3 production/support bytes are unchanged through R5. Four test files received installed-path/scope fixes; R5 additionally binds the already packaged pipeline README. Full revision differences and retained overrides are archived.

| Selected existing run | Freeze | Passed | Ordinary skips | Known xfails | Seconds |
|---|---|---:|---:|---:|---:|
| installed-connected-attempt02 | R5 | 1535 | 3 | 3 | 33.07 |
| installed-consumers-attempt01 | R4 | 12 | 0 | 0 | 3.92 |
| installed-focused-attempt02 | R4 | 192 | 0 | 0 | 14.55 |
| source-connected-attempt03 | R3 | 1536 | 2 | 3 | 34.11 |
| source-consumers-attempt02 | R4 | 12 | 0 | 0 | 4.01 |
| source-contracts-attempt02 | R3 | 1 | 0 | 0 | 1.06 |
| source-default-attempt02 | R3 | 6533 | 128 | 3 | 315.37 |
| source-focused-attempt03 | R4 | 192 | 0 | 0 | 16.04 |
| source-path-regressions-attempt01 | R5 | 18 | 0 | 0 | 2.08 |

All selected runs report zero failures/errors, verified before/after freeze checks, and zero product-origin leaks. Exact commands, interpreter paths, environments, JUnit, resource receipts, raw ordered node IDs and all origin receipts are archived.
Focused source/installed selections contain the same 192 ordered node IDs; connected selections contain the same 1,541 ordered node IDs. The 12 consumer raw prefixes differ by pytest root; the explicit support filename, function and parameter order match exactly. Both raw lists are retained.
Connected source has two model-artifact skips. Installed connected adds only the explicit source-only private sampler.cpp architecture-fixture skip; real native tests remain in the installed selection. Three known kekulization xfails are separate from ordinary skips.
The R3 full-default result is 6,533 passed, 128 ordinary skips and three known xfails. It is reused with targeted R4/R5 path revalidation; it is not a fresh whole R5 source/default run.

Retained failures: source-connected01 (one), source-default01 (nine), source-consumers01 (four missing-torch imports), installed-focused01 (seven path assumptions), and installed-connected01 (two source-tree path assumptions). Raw failures, original freezes, prior passing attempts and retained source/test/support overrides remain archived.
R4 consumer checks use a separate CP311 environment with real torch 2.7.1+cpu and torch-geometric 2.6.1. Acquisition, import/native checks, exact interpreter-only wrappers, package versions/file manifests and installation reports are retained; the venv and caches are excluded.

The private CP311 wheel has 883 product members, all matching the installed site. It contains pipeline/README.md and excludes legacy core.py and private sampler.cpp. Exactly six accepted CP311 native binaries match the Stage8 baseline and source/wheel bytes; no native rebuild is claimed. The other frozen version-specific binaries receive no acceptance.
The final R5 architecture receipt passes its declared static ratchet/layer scope, with 24 remaining debt entries and no new debt or layer/core/carrier findings. Runtime/native/global acceptance remains separate.
Final static census over the 508 frozen Python product files: 770 classes, 355 dataclasses, eight historical bare Protocols and two TypedDicts. The older 507-file census receipt is preserved with its scope difference documented.

Next Stage9 work remains MolBundle rows 9.2-026 and 9.6-006. Stage9 whole and overall gates are false. This archive supplies no clean-native/SDK/sdist/other-version/science/performance/global acceptance evidence.

Archive: `.plan20-artifacts/20261104/stage9b/cutover`.
Deterministic manifest: `.plan20-artifacts/20261104/stage9b/cutover/manifest.json` (SHA-256 `18a8e6e3a08faa3a68ac9f14cd155133a69455c318252604f3e3267dc4b4a7fc`); 2,396 files, 109,512,531 bytes. Every copied file was verified by byte count and SHA-256.
The archive preserves relative source paths under long/ and medium/, plus the native-baseline receipt and derived static comparisons. Generated runtime/cache/temp trees and duplicate wheel build/site/outside/egg-info/staging trees are excluded.


[Root checkpoint](checkpoint.json) · [Evidence graph](evidence_graph.json).
