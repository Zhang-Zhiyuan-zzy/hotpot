FROM condaforge/miniforge3:latest

ARG PYTHON_VERSION=3.13

LABEL org.opencontainers.image.title="Hotpot conda test image" \
      org.opencontainers.image.description="Conda-based development and test environment for Hotpot" \
      org.opencontainers.image.source="https://github.com/Zhang-Zhiyuan-zzy/hotpot"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/hotpot-matplotlib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/hotpot

COPY . .

RUN conda config --set channel_priority strict \
    && conda create -y -n hp -c conda-forge \
        "python=${PYTHON_VERSION}" \
        "pip>=24" \
        "setuptools>=77,<82" \
        "wheel>=0.43" \
        "cairo>=1.18" \
    && conda clean -afy

RUN conda run --no-capture-output -n hp python -m pip install --upgrade pip \
    && conda run --no-capture-output -n hp python -m pip install -e '.[dev]'

# Fail the image build immediately if the environment or Hotpot installation is broken.
RUN conda run --no-capture-output -n hp python -m pip check \
    && conda run --no-capture-output -n hp python -c \
        "import hotpot; from openbabel import openbabel; print(f'hotpot={hotpot.version()} openbabel={openbabel.OBReleaseVersion()}')" \
    && conda run --no-capture-output -n hp python -m pytest \
        -q -p no:cacheprovider tests/test_packaging_metadata.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hp"]
CMD ["bash", "tests/run_coverage.sh"]
