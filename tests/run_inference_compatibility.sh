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
            --with 'numpy>=1.24,<3' \
            --with 'onnxruntime>=1.19,<2' \
            --with 'rdkit>=2023.9' \
            --with 'pytest>=7,<10' \
            python -m pytest -q -p no:cacheprovider tests/mca tests/cbond
    )
done
