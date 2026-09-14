#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

ln -s "$repo_root/hotpot/cheminfo/AImodels/mca" "$stage/mca"
ln -s "$repo_root/hotpot/cheminfo/AImodels/cbond" "$stage/cbond"

if (( $# )); then
    versions=("$@")
else
    versions=(3.9 3.10 3.11 3.12 3.13 3.14)
fi

for version in "${versions[@]}"; do
    echo "Python $version"
    (
        cd "$stage"
        uv run --no-project --python "$version" \
            --with 'numpy>=1.24,<3' \
            --with 'onnxruntime>=1.19,<2' \
            --with 'rdkit>=2023.9' \
            --with 'pytest>=7,<10' \
            python -m pytest -q -p no:cacheprovider mca/tests cbond/tests
    )
done
