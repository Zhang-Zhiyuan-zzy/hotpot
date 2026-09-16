#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
coverage_dir="$repo_root/tests/coverage"

mkdir -p "$coverage_dir"
export COVERAGE_FILE="$coverage_dir/.coverage"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hotpot-coverage-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

cd "$repo_root"
python -m coverage erase

test_targets=(
  tests/mca
  tests/cbond
  tests/test_cheminfo/test_mca_calculator.py
  tests/test_cheminfo/test_molecule_conversion.py
  tests/test_cheminfo/test_ob2chem_compat.py
  tests/test_cheminfo/test_bond_metadata.py
  tests/test_cheminfo/test_ligand_rings.py
  tests/test_cheminfo/test_search.py
  tests/test_cheminfo/test_search_mapping.py
  tests/test_cheminfo/test_smarts.py
  tests/test_smart_parser.py
  tests/smarts_conformance
)

set +e
python -m coverage run --branch --source=hotpot -m pytest \
  -q \
  -p no:cacheprovider \
  --junitxml="$coverage_dir/junit.xml" \
  "${test_targets[@]}"
test_status=$?
set -e

python -m coverage report --show-missing | tee "$coverage_dir/coverage.txt"
python -m coverage xml -o "$coverage_dir/coverage.xml"
python -m coverage html -d "$coverage_dir/html"

printf 'Test results: %s\n' "$coverage_dir/junit.xml"
printf 'Coverage XML: %s\n' "$coverage_dir/coverage.xml"
printf 'Coverage report: %s\n' "$coverage_dir/coverage.txt"
printf 'Coverage HTML: %s\n' "$coverage_dir/html/index.html"

exit "$test_status"
