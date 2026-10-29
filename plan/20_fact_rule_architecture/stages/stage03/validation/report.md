# Stage3 frozen build8 validation

The complete default suite and the affected Python3.10–3.14 source/package
matrix pass on the frozen build8 source. [Evidence](evidence.json) binds the
source and native identities, exact commands, test selections, raw logs,
JUnit results, elapsed times and explicit limits. Stage3 storage/scaling and
its closure are recorded separately; this report does not certify the final
scientific or global efficiency matrix.

## Executed checks

| Check | Actual outcome |
|---|---|
| Complete source default |4432 passed,123 skipped,3 xfailed,49 passed subtests;257.20s pytest /259.05s command wall time |
| Affected source, each of3.10–3.14 |1023 passed,8 skipped |
| `PYTHONMALLOC=debug` native tests, each version |75 passed |
| Installed current wheel runtime, each version |941 passed,8 skipped |
| Each current wheel content audit and installed core smoke | Passed |
| Actual current sdist build and member audit | Passed; current CPP/stub/PSF license bytes match, no prebuilt `.so` |
| Architecture ratchet against `ba662e3` |281 remaining debt entries; no new debt, allowlist growth, layer or carrier-definition findings |

The default command is `/tmp/hotpot-plan20-env/bin/python -m pytest -q
--import-mode=importlib tests`, pinned to CPU24–31, with the exact JUnit,
cache and environment arguments in the linked command record. The matrix
entry command was `PYTHONDONTWRITEBYTECODE=1 /tmp/hotpot-plan20-env/bin/python
.plan20-artifacts/20261029/stage3/fact-native/helpers/run_matrix_final.py`.
Every version's command records contain its complete actual interpreter path,
argv, cwd, environment overrides and source/native/tests/runner hashes.

The canonical Stage3 selection contains37 targets; adding import safety,
toolkit import order and changed-build static checks produces43 source targets.
Installed runtime uses38 targets. The architecture scanner and four packaging
source scans remain source-only checks. All375 debug-allocator tests ran;
the eight skipped nodes in each source/installed run require external xTB.
Default-tier skips retain their actual tier meanings.

Production and native bytes remained unchanged throughout the default run.
The matrix also verifies before/after hashes for tests and recording helpers,
and its whole frozen before/after manifest matches. Installed tests use copied
unchanged test/support bytes, an isolated `pip --target` install and resolved
Hotpot/MCA module origins. No installed check imports current checkout Python
through the MCA test-path helper.

## Packaging and native boundary

The new `_fact_native` was actually rebuilt against each matching CPython
minor version. Its allocator/atomicity, raw-value, permissions, source/member,
bound/unbound signature and lifetime contracts run in all five source and
installed runtime selections. The separate debug runs use the same matching
binaries and the actual allocator rejection probes.

Current wheel assembly uses current `build_py` and `bdist_wheel --skip-build`,
the rebuilt fact extension, and the unchanged B20 native dependencies. Their
source closure, build recipes, original wheel/member hashes and installed
member identities are explicitly compared and retained. The fact stub and
unmodified PSF license are audited in each wheel and the actual sdist.

This establishes current Python plus new-native source/package integration.
Stage10 still owns all-component clean-source rebuilding, native path/build
migration and standalone SDK/symbol evidence. The real licensed CCDC tier was
not run; the factual fake-CCDC importer contracts were. No scientific success
count, model-inference result or performance verdict is inferred from this
technical matrix.
