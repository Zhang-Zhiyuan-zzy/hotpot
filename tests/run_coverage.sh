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
  tests/test_cheminfo/test_relevant_ring_integration.py
  tests/test_cheminfo/test_hidden_bond_restoration.py
  tests/test_cheminfo/geometry
  tests/test_cheminfo/test_geometry.py
  tests/test_cheminfo/test_geometry_core_integration.py
  tests/test_cheminfo/test_geometry_quality.py
  tests/test_cheminfo/test_forcefield_acceptance.py
  tests/test_cheminfo/test_forcefield_api.py
  tests/test_cheminfo/test_forcefield_package.py
  tests/test_cheminfo/test_forcefield_optimizer.py
  tests/test_cheminfo/test_forcefield_integration.py
  tests/test_cheminfo/test_complexes_build.py
  tests/test_cheminfo/test_complex_hydrogens.py
  tests/test_cheminfo/test_complex_untangling_workflow.py
  tests/test_cheminfo/test_topology_determinism.py
  tests/test_cheminfo/test_search.py
  tests/test_cheminfo/test_search_mapping.py
  tests/test_cheminfo/test_smarts.py
  tests/test_cheminfo/test_import_safety.py
  tests/test_smart_parser.py
  tests/smarts_conformance
  tests/test_works/test_convert.py
)

set +e
python -m coverage run --branch --source=hotpot -m pytest \
  -q \
  --import-mode=importlib \
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
