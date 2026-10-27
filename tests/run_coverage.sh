#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
coverage_dir="$repo_root/tests/coverage"

mkdir -p "$coverage_dir"
export COVERAGE_FILE="$coverage_dir/.coverage"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hotpot-coverage-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

work_dir=${HOTPOT_TEST_WORKDIR:-$repo_root}
mkdir -p "$work_dir"
cd "$work_dir"
if [[ -n "${HOTPOT_TEST_INSTALLED_PACKAGE:-}" ]]; then
  test_support="$work_dir/python-test-support"
  mkdir -p "$test_support"
  ln -sfn "$repo_root/tests" "$test_support/tests"
  export PYTHONSAFEPATH=1
  export PYTHONPATH="$test_support"
fi
python -m coverage erase

pytest_options=(
  -q
  --import-mode=importlib
  -p no:cacheprovider
  --junitxml="$coverage_dir/junit.xml"
)
if [[ "${HOTPOT_RUN_MODEL_ARTIFACT_TESTS:-}" == "1" ]]; then
  pytest_options+=(--run-model-artifacts)
fi

set +e
python -m coverage run --branch --source=hotpot -m pytest \
  "${pytest_options[@]}" \
  "$repo_root/tests"
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
