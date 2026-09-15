# SMARTS conformance tests

This directory implements the test layers required by
`plan/test_search_and_SMARTS.md`. It exercises only the active NetworkX-backed
parser and matcher. Production code is intentionally not changed by this test
audit.

## Profiles

Run from the repository root with an environment in which Hotpot's project
dependencies are installed:

```bash
# Existing focused baseline
python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/test_search.py \
  tests/test_cheminfo/test_search_mapping.py \
  tests/test_cheminfo/test_smarts.py \
  tests/test_smart_parser.py

# Fast, deterministic supported-contract checks
python -m pytest -q -p no:cacheprovider \
  -m smarts_smoke tests/smarts_conformance

# Full core, including intentional strict failures for documented defects
python -m pytest -q -p no:cacheprovider \
  -m smarts_core tests/smarts_conformance

# Supported-behavior subset (useful while the strict defects remain open)
python -m pytest -q -p no:cacheprovider \
  -m 'smarts_core and not smarts_known_failure' tests/smarts_conformance

# Corpus schema and count gates
python -m pytest -q -p no:cacheprovider \
  tests/smarts_conformance/test_corpus_validator.py
python -m tests.smarts_conformance.audit_corpus \
  --output /tmp/hotpot-smarts-corpus-audit.json

# Active parser/matcher line and branch coverage
env COVERAGE_FILE=/tmp/hotpot-smarts.coverage \
  python -m coverage run --branch --source=hotpot/cheminfo/search \
  -m pytest -q -p no:cacheprovider \
  -m 'smarts_core and not smarts_known_failure' tests/smarts_conformance
env COVERAGE_FILE=/tmp/hotpot-smarts.coverage \
  python -m coverage report -m \
  hotpot/cheminfo/search/smarts.py hotpot/cheminfo/search/search.py

# Optional/heavy evidence collectors
python -m tests.smarts_conformance.differential.runner --output /tmp/smarts-differential.json
python -m tests.smarts_conformance.fuzz.run_deterministic --seed 20260915
python -m tests.smarts_conformance.benchmarks.benchmark_smarts --output /tmp/smarts-benchmark.json
```

`smarts_known_failure` tests encode the intended contract and therefore fail
until the corresponding production defect is fixed. They are not xfailed,
skipped, or weakened. The smoke profile deliberately contains only supported
behavior so it can diagnose failures in the test infrastructure itself.
The supported core also executes the entire generated corpus and compares its
mismatches to `corpus/known_mismatches.json`. Thus excluding strict failures
does not skip the corpus: any new mismatch, changed classification, or repaired
mismatch fails the baseline gate and requires an explicit review of the
snapshot.

The differential and benchmark programs are evidence collectors, not ordinary
CI dependencies. They never rewrite golden expectations. Run the fuzz command
under an external timeout for resource isolation, for example:

```bash
timeout 60s python -m tests.smarts_conformance.fuzz.run_deterministic --seed 20260915
```

## Corpus governance

Cases are source-controlled or deterministically generated from reviewed truth
tables. Every case has an ID, classification, feature tags, source, and license
record. `curated-hotpot` means an original minimal example authored for this
repository under its MIT license. Reference-engine outputs are supporting
evidence, not automatically accepted truth.

To add a regression:

1. minimize the SMARTS/target pair;
2. classify syntax, unsupported feature, target preprocessing, semantic,
   enumeration, dialect, or robustness behavior;
3. add a unique issue/case ID and evidence;
4. add the intended assertion to `test_regressions.py`;
5. update `feature_matrix.yaml` and the conformance report;
6. run smoke, core, the validator, and the relevant differential case.

Golden data must be reviewed as a diff. No command in this directory silently
updates expected output from the current Hotpot implementation.
