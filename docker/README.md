# Hotpot Docker Test Environment

This configuration installs and tests Hotpot in an isolated Conda environment.
It supports Python 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14, with Python 3.13
as the default.

The Docker build process automatically performs the following tasks:

1. Initializes a Linux image based on Miniforge and installs system dependencies.
2. Creates a Conda environment named `hp` using `conda-forge`.
3. Installs the specified Python version, Cairo, pip, setuptools, and wheel in `hp`.
4. Installs `hotpot-zzy` in editable mode along with its `dev` test dependencies.
5. Runs `pip check`, Hotpot/Open Babel import checks, and packaging metadata tests.
6. Runs `tests/run_coverage.sh` by default when the container starts.

Images are managed by the Docker daemon. Exported `.tar` archives are stored in
`~/docker/image` by default and are not written to the Hotpot repository. The
build script also rejects archive directories located inside the repository.

## Build and Install in One Step

Run the following command from the root of the Hotpot repository:

```bash
./docker/build-hotpot-test-images.sh
```

By default, this command builds `hotpot-test:py313-conda` and exports it to:

```text
~/docker/image/hotpot-test-py313-conda.tar
```

Select one, multiple, or all supported Python versions:

```bash
# One version
./docker/build-hotpot-test-images.sh 3.11

# Multiple versions
./docker/build-hotpot-test-images.sh 3.9 3.13 3.14

# All versions from 3.9 through 3.14
./docker/build-hotpot-test-images.sh all
```

To store the archive in another directory outside the repository:

```bash
HOTPOT_DOCKER_IMAGE_DIR=/data/docker-images \
  ./docker/build-hotpot-test-images.sh 3.13
```

## Use Docker Commands Directly

The following `docker build` command initializes the system, creates the Conda
environment, installs Hotpot, and runs the build-time checks:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.13 \
  --tag hotpot-test:py313-conda \
  --file docker/test/Dockerfile \
  .
```

To export the image, run:

```bash
mkdir -p "$HOME/docker/image"
docker save \
  --output "$HOME/docker/image/hotpot-test-py313-conda.tar" \
  hotpot-test:py313-conda
```

To build another version, update both `PYTHON_VERSION` and the version in the
tag. For example, Python 3.10 uses `PYTHON_VERSION=3.10` and
`hotpot-test:py310-conda`.

## Create a Container and Run Tests

Run the maintained coverage tests in a Python 3.13 container by default:

```bash
./docker/run-hotpot-test-container.sh
```

Specify a Python version or a custom test command:

```bash
./docker/run-hotpot-test-container.sh --python 3.11

./docker/run-hotpot-test-container.sh --python 3.13 \
  python -m pytest -q tests/test_packaging_metadata.py
```

The script mounts the current Hotpot source tree at `/workspace/hotpot`. If the
corresponding image is unavailable in the local Docker daemon but its archive
exists in `~/docker/image`, the script automatically runs `docker load` first.

Run the default tests directly with Docker:

```bash
docker run --rm \
  --name hotpot-test-py313 \
  --volume "$PWD:/workspace/hotpot" \
  --workdir /workspace/hotpot \
  --env MPLCONFIGDIR=/tmp/hotpot-matplotlib \
  hotpot-test:py313-conda
```

Create a persistent interactive test container:

```bash
docker run -it \
  --name hotpot-test-py313-shell \
  --volume "$PWD:/workspace/hotpot" \
  --workdir /workspace/hotpot \
  hotpot-test:py313-conda bash
```

The image entrypoint runs commands through `conda run -n hp`, so the default
tests, custom Python commands, and interactive shell all run in the `hp` Conda
environment.

## Files

- `docker/test/Dockerfile`: Initializes the system, creates the Conda environment,
  and installs and verifies Hotpot.
- `docker/build-hotpot-test-images.sh`: Builds and exports images for one,
  multiple, or all supported Python versions.
- `docker/run-hotpot-test-container.sh`: Loads the selected image, mounts the
  current source tree, and creates a test container.
