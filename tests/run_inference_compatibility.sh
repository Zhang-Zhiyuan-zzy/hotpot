#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# )); then
    versions=("$@")
else
    versions=(3.9 3.10 3.11 3.12 3.13 3.14)
fi

for version in "${versions[@]}"; do
    echo "Python $version"
    (
        cd "$repo_root"
        uv run --no-project --python "$version" \
            --with-requirements tests/requirements-inference.txt \
            python -m pytest -q -p no:cacheprovider \
            tests/mca tests/cbond \
            tests/test_cheminfo/test_mca_calculator.py \
            tests/test_cheminfo/test_molecule_conversion.py \
            tests/test_cheminfo/test_ob2chem_compat.py \
            tests/test_cheminfo/test_bond_metadata.py \
            tests/test_cheminfo/test_ligand_rings.py \
            tests/test_cheminfo/test_search.py \
            tests/test_cheminfo/test_search_mapping.py \
            tests/test_cheminfo/test_smarts.py \
            tests/test_smart_parser.py

        uv run --no-project --python "$version" \
            --with-requirements tests/requirements-inference.txt \
            python -m pytest -q -p no:cacheprovider \
            -m smarts_core tests/smarts_conformance
    )
done
