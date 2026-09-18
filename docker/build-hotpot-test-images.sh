#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_dir="${HOTPOT_DOCKER_IMAGE_DIR:-${HOME:?}/docker/image}"
image_name="${HOTPOT_TEST_IMAGE_NAME:-hotpot-test}"
supported_versions=(3.9 3.10 3.11 3.12 3.13 3.14)

usage() {
    cat <<'EOF'
Usage: docker/build-hotpot-test-images.sh [VERSION ... | all]

Build Hotpot conda test images and export them outside the repository.
With no VERSION, Python 3.13 is built.

Examples:
  docker/build-hotpot-test-images.sh
  docker/build-hotpot-test-images.sh 3.11
  docker/build-hotpot-test-images.sh 3.9 3.13 3.14
  docker/build-hotpot-test-images.sh all
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# == 0 )); then
    versions=(3.13)
elif (( $# == 1 )) && [[ "$1" == "all" ]]; then
    versions=("${supported_versions[@]}")
else
    versions=("$@")
fi

image_dir="$(realpath -m "$image_dir")"
case "$image_dir" in
    "$repo_root"|"$repo_root"/*)
        echo "Refusing to store Docker image archives inside the Hotpot repository: $image_dir" >&2
        echo "Use the default ~/docker/image or set HOTPOT_DOCKER_IMAGE_DIR outside the repository." >&2
        exit 2
        ;;
esac

mkdir -p "$image_dir"

for python_version in "${versions[@]}"; do
    case "$python_version" in
        3.9|3.10|3.11|3.12|3.13|3.14) ;;
        *)
            echo "Unsupported Python version: $python_version" >&2
            echo "Use 'all' by itself to build every supported version." >&2
            echo "Supported versions: 3.9 3.10 3.11 3.12 3.13 3.14" >&2
            exit 2
            ;;
    esac

    python_tag="${python_version/./}"
    image_tag="${image_name}:py${python_tag}-conda"
    archive_path="$image_dir/${image_name}-py${python_tag}-conda.tar"

    echo "Building $image_tag with Python $python_version"
    docker build \
        --build-arg "PYTHON_VERSION=$python_version" \
        --tag "$image_tag" \
        --file "$repo_root/docker/test/Dockerfile" \
        "$repo_root"

    docker save "$image_tag" -o "$archive_path"
    echo "Saved $image_tag to $archive_path"
done
