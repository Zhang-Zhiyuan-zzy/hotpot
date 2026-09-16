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
            tests/test_cheminfo/test_geometry.py \
            tests/test_cheminfo/test_geometry_quality.py \
            tests/test_cheminfo/test_forcefield_api.py \
            tests/test_cheminfo/test_forcefield_optimizer.py \
            tests/test_cheminfo/test_forcefield_integration.py \
            tests/test_cheminfo/test_complexes_build.py \
            tests/test_cheminfo/test_complex_hydrogens.py \
            tests/test_cheminfo/test_topology_determinism.py \
            tests/test_cheminfo/test_search.py \
            tests/test_cheminfo/test_search_mapping.py \
            tests/test_cheminfo/test_smarts.py \
            tests/test_works/test_convert.py \
            tests/test_cheminfo/test_import_safety.py \
            tests/test_smart_parser.py

        uv run --no-project --python "$version" \
            --with-requirements tests/requirements-inference.txt \
            python -m pytest -q -p no:cacheprovider \
            -m smarts_core tests/smarts_conformance
    )
done
