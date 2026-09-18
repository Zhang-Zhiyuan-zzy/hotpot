#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_version="${PYTHON_VERSION:-3.13}"
image_dir="${HOTPOT_DOCKER_IMAGE_DIR:-${HOME:?}/docker/image}"
image_name="${HOTPOT_TEST_IMAGE_NAME:-hotpot-test}"

usage() {
    cat <<'EOF'
Usage: docker/run-hotpot-test-container.sh [--python VERSION] [COMMAND ...]

Run Hotpot tests in the selected conda image. Python 3.13 is the default.
With no COMMAND, tests/run_coverage.sh is executed.

Examples:
  docker/run-hotpot-test-container.sh
  docker/run-hotpot-test-container.sh --python 3.11
  docker/run-hotpot-test-container.sh --python 3.14 python -m pytest -q
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--python" ]]; then
    if (( $# < 2 )); then
        echo "--python requires a version." >&2
        exit 2
    fi
    python_version="$2"
    shift 2
fi

case "$python_version" in
    3.9|3.10|3.11|3.12|3.13|3.14) ;;
    *)
        echo "Unsupported Python version: $python_version" >&2
        echo "Supported versions: 3.9 3.10 3.11 3.12 3.13 3.14" >&2
        exit 2
        ;;
esac

python_tag="${python_version/./}"
image_tag="${image_name}:py${python_tag}-conda"
archive_path="$image_dir/${image_name}-py${python_tag}-conda.tar"
container_name="${HOTPOT_TEST_CONTAINER_NAME:-hotpot-test-py${python_tag}}"

if ! docker image inspect "$image_tag" >/dev/null 2>&1; then
    if [[ -f "$archive_path" ]]; then
        docker load -i "$archive_path"
    else
        echo "Missing image $image_tag and archive $archive_path" >&2
        echo "Build it with: docker/build-hotpot-test-images.sh $python_version" >&2
        exit 2
    fi
fi

if (( $# == 0 )); then
    docker run --rm \
        --name "$container_name" \
        --volume "$repo_root:/workspace/hotpot" \
        --workdir /workspace/hotpot \
        --env MPLCONFIGDIR=/tmp/hotpot-matplotlib \
        "$image_tag"
else
    docker run --rm \
        --name "$container_name" \
        --volume "$repo_root:/workspace/hotpot" \
        --workdir /workspace/hotpot \
        --env MPLCONFIGDIR=/tmp/hotpot-matplotlib \
        "$image_tag" "$@"
fi
