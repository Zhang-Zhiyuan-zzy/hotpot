#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if (( $# )); then
    versions=("$@")
else
    versions=(3.10 3.11 3.12 3.13 3.14)
fi

for version in "${versions[@]}"; do
    echo "Python $version"
    (
        wheel=${HOTPOT_TEST_WHEEL:-}
        uv_options=(
            run
            --no-project
            --python "$version"
            --with-requirements "$repo_root/tests/requirements-inference.txt"
        )
        if [[ -n "$wheel" ]]; then
            wheel=$(realpath "$wheel")
            uv_options+=(--with "$wheel")
        else
            cd "$repo_root"
            uv run --no-project --python "$version" \
                --with-requirements tests/requirements-inference.txt \
                --with 'setuptools>=77,<82' \
                --with 'pybind11>=3,<4' \
                python setup.py build_ext --inplace --force
        fi

        work_dir=${HOTPOT_TEST_WORKDIR:-$repo_root}
        mkdir -p "$work_dir"
        cd "$work_dir"

        uv "${uv_options[@]}" \
            python -c '
from openbabel import openbabel as ob

actual = tuple(map(int, ob.OBReleaseVersion().split(".")[:2]))
expected = (3, 2)
print(f"Open Babel {ob.OBReleaseVersion()} (expected {expected[0]}.{expected[1]}.x)")
assert actual == expected
'

        uv "${uv_options[@]}" \
            python -m hotpot models install --all

        export HOTPOT_MODEL_SOURCE=local

        test_targets=(
            "$repo_root/tests/mca"
            "$repo_root/tests/cbond"
            "$repo_root/tests/test_cheminfo/calculator"
            "$repo_root/tests/test_cheminfo/test_calculator.py"
            "$repo_root/tests/test_cheminfo/test_mca_calculator.py"
            "$repo_root/tests/test_cheminfo/test_molecule_conversion.py"
            "$repo_root/tests/test_cheminfo/test_ob2chem_compat.py"
            "$repo_root/tests/test_cheminfo/test_bond_metadata.py"
            "$repo_root/tests/test_cheminfo/test_ligand_rings.py"
            "$repo_root/tests/test_cheminfo/test_relevant_ring_integration.py"
            "$repo_root/tests/test_cheminfo/test_hidden_bond_restoration.py"
            "$repo_root/tests/test_cheminfo/kekulize"
            "$repo_root/tests/test_cheminfo/obWrappers"
            "$repo_root/tests/test_cheminfo/forcefields"
            "$repo_root/tests/test_cheminfo/geometry"
            "$repo_root/tests/test_cheminfo/test_geometry.py"
            "$repo_root/tests/test_cheminfo/test_geometry_core_integration.py"
            "$repo_root/tests/test_cheminfo/test_geometry_quality.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_acceptance.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_api.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_checkpoint_acceptance.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_cli.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_package.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_optimizer.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_stage_scan_boundaries.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_trajectory.py"
            "$repo_root/tests/test_cheminfo/test_forcefield_integration.py"
            "$repo_root/tests/test_cheminfo/test_complexes_build.py"
            "$repo_root/tests/test_cheminfo/test_complex_hydrogens.py"
            "$repo_root/tests/test_cheminfo/test_complex_untangling_workflow.py"
            "$repo_root/tests/test_cheminfo/test_topology_determinism.py"
            "$repo_root/tests/test_cheminfo/test_search.py"
            "$repo_root/tests/test_cheminfo/test_search_mapping.py"
            "$repo_root/tests/test_cheminfo/test_search_native_contract.py"
            "$repo_root/tests/test_cheminfo/test_search_native_kernel.py"
            "$repo_root/tests/test_cheminfo/test_search_native_compiler.py"
            "$repo_root/tests/test_cheminfo/test_search_native_production_kernel.py"
            "$repo_root/tests/test_cheminfo/test_search_native_predicate_parity.py"
            "$repo_root/tests/test_cheminfo/test_search_native_cutover.py"
            "$repo_root/tests/test_cheminfo/test_search_native_prepared.py"
            "$repo_root/tests/test_cheminfo/test_smarts.py"
            "$repo_root/tests/test_cheminfo/graph"
            "$repo_root/tests/test_works/test_convert.py"
            "$repo_root/tests/test_cheminfo/test_import_safety.py"
            "$repo_root/tests/test_smart_parser.py"
            "$repo_root/tests/test_cli/test_run_cli.py"
            "$repo_root/tests/test_cli/test_xtb_cli.py"
            "$repo_root/tests/test_main/test_command_loading.py"
            "$repo_root/tests/test_pipeline"
            "$repo_root/tests/test_plugin/test_harness"
            "$repo_root/tests/test_plugin/test_xtb"
            "$repo_root/tests/benchmarks/coordination_complexes/test_xtb_benchmark.py"
        )

        uv "${uv_options[@]}" \
            python -m pytest -q -p no:cacheprovider \
            --exclude-warning-annotations \
            --import-mode=importlib \
            "${test_targets[@]}"

        uv "${uv_options[@]}" \
            python -m pytest -q -p no:cacheprovider \
            --exclude-warning-annotations \
            --import-mode=importlib \
            -m smarts_core "$repo_root/tests/smarts_conformance"
    )
done
