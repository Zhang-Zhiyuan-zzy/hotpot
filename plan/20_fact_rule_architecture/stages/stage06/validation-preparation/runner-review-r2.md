REVIEW_RESULT
task_id=p20-s6-runner-review-704283-33
revision=2
verification_code=704283
task_hash=2def81c065c71577a253c91f188d15d11fef7a6634d4ce2328b108f6f59fffd2
source_commit=90ff36c2c0b415bb9cf27628265bb4528e425922
diff_or_tree_hash=ec97b93f89677771ab63b56fda01f0bc47ff2640e5d58ae0304564406660bf19
capture_identity_kind=SHA256 of frozen revision2 review-capture.json raw bytes; NOT Git tree
status=SIX_FINDINGS_RESOLVED_STATICALLY_FREEZE_EXECUTION_REMAINS_HELD

| # | Prior severity | Resolution | Evidence and bounded conclusion |
|---|---|---|---|
| 1 | BLOCKER | RESOLVED | `historical_source()` requires exactly the 13 named canonical `third_party/rdkit` paths, rejects overlap with the immutable Stage5 source map, and compares the supplement against both the bound cp312 historical source record and the matching proof record. The runner hard-binds supplement `02f81aeb…`, proof `ce903f4b…`, and source identity `d5ecea19…`; surplus, missing, overwrite, changed-hash and noncanonical fixtures reject. The supplied history distinguishes the independent RD-native build commit `e6082f5f…` from the Stage5 fact-native build commit `cac32af6…`, and records the five B20→Stage3→Stage5 RD-native reuse chains. This establishes the supplied historical map only; it is not a fresh Git or native-build verification. |
| 2 | HIGH | RESOLVED | `required_nodes()` now selects `source` paths for source, installed and default tiers, and `native-source` paths only for the native tier. SDK nodes are removed only from an unbound native tier. Default never enables high-compute/external flags, so gated native/SDK skips are not misclassified as required Stage6 failures. Synthetic cases cover both SDK-bound and unbound default behavior. |
| 3 | HIGH | RESOLVED | `relative_path()`, `regular_file()`, `test_file()` and `support_plan()` reject absolute paths, traversal, noncanonical spellings, symlink sources, non-regular files, native-package injection and destination escape. Selected tests must be `tests/**/test_*.py` outside `native_cpp`; support comes from the frozen eight-file allowlist plus `pyproject.toml`. The full copy plan is validated during `prepare()` and again before packaging. Synthetic absolute, traversal, source-symlink, destination-symlink and unallowlisted cases reject before copying. |
| 4 | HIGH | RESOLVED | `SDK_NODES` freezes exactly ABI0 and ABI1. Native collection requires exactly those two prefix-matching nodes, external metadata and no xfail marker. Unbound execution requires both to be ordinary skips; bound execution requires both to be required call passes without skip, failure or xfail. Missing, surplus, unexpected-pass, xfail, missing-report and bound-skip fixtures reject. Default has no SDK qualification policy and remains `not_qualified_in_this_tier`. Actual current SDK evidence remains missing until a bound native-tier execution succeeds. |
| 5 | MEDIUM | RESOLVED | `release_paths()` runs at the start of `prepare()`, before snapshotting or output creation. It requires canonical absolute roots, evidence beneath `ROOT/.plan20-artifacts`, runtime outside and non-ancestral to the checkout, mutual output/runtime nonoverlap, and a canonical freeze path strictly inside the released evidence root. `destination()` and `assert_freeze()` revalidate these constraints. Invalid-root preparation was retained as a rejection with no freeze directory created. |
| 6 | MEDIUM | RESOLVED | The suffix heuristic is replaced by the explicit 143-path inventory. `inventory_paths()` also reconstructs the expected set from the named owner-tree prefixes, so additions or deletions within them reject. The inventory includes the previously omitted `exports.map`, eight native owner trees, RDWrapper CMake/templates, six stubs, packaging inputs, and all 13 lock/fetch/probe files. Hash, missing-file and new suffixless owner-file mutations reject on both historical and candidate maps. This is bounded inventory/reuse policy evidence, not Stage10 clean-native, SDK or sdist acceptance. |

Evidence limits:

- Supplied verifier execution:  
  `PYTHONDONTWRITEBYTECODE=1 python .../revision2/check_revision2.py .../correctness-attempt1`
- Bound result: `checks.json` SHA-256 `1de7553c70090039f4c3f3bd0005ac80e617ca80dcc829e25db53244feea4aad`; 53 deterministic synthetic/content checks passed.
- The check imported neither pytest nor Hotpot and launched no subprocess.
- The historical 123 checks and Git-command records are supplied prior evidence; this reviewer and revision communicator did not rerun them.
- No Stage6 collection, pytest execution, build, installation, wheel validation, SDK execution, current closure comparison, scientific test, benchmark, or statistical result is established.

No remaining code delta was identified within the six task33 findings. This does not accept the runner or authorize freeze/execution. Root must still obtain the matching revision2 exact-turn read-only audit, issue a completed `STAGE6_SOURCE_READY` release with actual new-test/support identities, and then evaluate the resulting runtime evidence. Task34 remains independent.