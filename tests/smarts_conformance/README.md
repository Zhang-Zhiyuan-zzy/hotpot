# SMARTS conformance tests

This directory implements the test layers required by
`plan/test_search_and_SMARTS.md`. It exercises only the active NetworkX-backed
parser and matcher, including the `FULL_GRAPH` and `LIGAND_SKELETON`
coordination semantics profiles.

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

# Strict core regression gate
python -m pytest -q -p no:cacheprovider \
  -m smarts_core tests/smarts_conformance

# Corpus schema and count gates
python -m pytest -q -p no:cacheprovider \
  tests/smarts_conformance/test_corpus_validator.py
python -m tests.smarts_conformance.audit_corpus \
  --output /tmp/hotpot-smarts-corpus-audit.json

# Active parser/matcher line and branch coverage
env COVERAGE_FILE=/tmp/hotpot-smarts.coverage \
  python -m coverage run --branch --source=hotpot/cheminfo/search \
  -m pytest -q -p no:cacheprovider \
  -m smarts_core tests/smarts_conformance
env COVERAGE_FILE=/tmp/hotpot-smarts.coverage \
  python -m coverage report -m \
  hotpot/cheminfo/search/_smarts_syntax.py \
  hotpot/cheminfo/search/errors.py \
  hotpot/cheminfo/search/search.py \
  hotpot/cheminfo/search/semantics.py \
  hotpot/cheminfo/search/smarts.py

# Optional/heavy evidence collectors
python -m tests.smarts_conformance.differential.runner --output /tmp/smarts-differential.json
python -m tests.smarts_conformance.fuzz.run_deterministic --seed 20260915
python -m tests.smarts_conformance.benchmarks.benchmark_smarts --output /tmp/smarts-benchmark.json
```

The historical `smarts_known_failure` marker remains attached to regression
tests that originally exposed defects, but those tests now pass and are part of
the strict gate. The smoke profile is a smaller infrastructure check. The core
gate also executes the entire generated corpus and compares its mismatches to
`corpus/known_mismatches.json`, which is currently empty. Any new mismatch or
changed classification fails the gate and requires explicit review.

Coordination coverage has three layers: perception-free Hotpot graphs,
Open Babel-backed MOL2/SDF fixtures, and repository CIF examples. The fixture
manifest records that Open Babel 3.1 collapses MOL2 `du`, `un`, and `nc` to
indistinguishable order-zero edges; Hotpot records these as `BondKind.UNKNOWN`
rather than inventing `ZERO` or `DATIVE` semantics.

The profile switch does not rerun hydrogen perception. Tests intentionally
freeze the observed difference for the same coordinated-amine topology:
Open Babel 3.1 supplies zero donor implicit H from MOL2 and one from SDF, so
`LIGAND_SKELETON` yields `X3/v3` and `X4/v4`, respectively.

At revision `7b262a9`, the strict command above reports **252 passed** on both
Python 3.9 and 3.14. The corpus audit reports **1,332 passed, 0 failed**. These
are scoped SMARTS results and do not imply a repository-wide test pass.

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
