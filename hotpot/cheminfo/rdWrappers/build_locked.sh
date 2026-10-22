#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 SOURCE_CACHE EMPTY_WORK_DIR TOOLCHAIN_PREFIX [PYTHON]" >&2
    exit 2
fi

source_cache=$(realpath -m "$1")
work_dir=$(realpath -m "$2")
toolchain=$(realpath "$3")
python_command=${4:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(realpath "$script_dir/../../..")
lock_dir="$repository_root/third_party/rdkit"

if [[ -e "$work_dir" ]] &&
   [[ -n $(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "rdWrappers work directory must be absent or empty: $work_dir" >&2
    exit 1
fi
mkdir -p "$work_dir/source" "$work_dir/build"

cmake="$toolchain/bin/cmake"
ninja="$toolchain/bin/ninja"
cc="$toolchain/bin/x86_64-conda-linux-gnu-gcc"
cxx="$toolchain/bin/x86_64-conda-linux-gnu-g++"

[[ $($cmake --version | head -1) == "cmake version 3.31.8" ]]
[[ $($ninja --version) == "1.12.1" ]]
[[ $($cc -dumpfullversion -dumpversion) == "14.3.0" ]]
[[ $($cxx -dumpfullversion -dumpversion) == "14.3.0" ]]
[[ $(uname -m) == "x86_64" ]]

HOTPOT_RDKIT_OFFLINE=1 "$lock_dir/fetch_locked_sources.sh" "$source_cache"

extract_gzip() {
    local archive=$1
    local destination=$2
    mkdir -p "$destination"
    tar -xzf "$archive" -C "$destination" --strip-components=1
}

extract_bzip2() {
    local archive=$1
    local destination=$2
    mkdir -p "$destination"
    tar -xjf "$archive" -C "$destination" --strip-components=1
}

extract_gzip \
    "$source_cache/rdkit-Release_2026_09_1.tar.gz" \
    "$work_dir/source/rdkit"
extract_bzip2 \
    "$source_cache/boost_1_86_0.tar.bz2" \
    "$work_dir/source/boost"
extract_gzip \
    "$source_cache/Catch2-v3.4.0.tar.gz" \
    "$work_dir/source/catch2"
extract_gzip \
    "$source_cache/better-enums-c35576bed0295689540b39873126129adfa0b4c8.tar.gz" \
    "$work_dir/source/better-enums"
extract_gzip \
    "$source_cache/eigen-3.4.0.tar.gz" \
    "$work_dir/source/eigen"
extract_gzip \
    "$source_cache/RingDecomposerLib-v1.1.3_rdkit.tar.gz" \
    "$work_dir/source/rdkit/External/RingFamilies/RingDecomposerLib"

build_sdk=${HOTPOT_RDWRAPPERS_BUILD_SDK:-ON}
build_python=${HOTPOT_RDWRAPPERS_BUILD_PYTHON:-ON}
python_output=${HOTPOT_RDWRAPPERS_PYTHON_OUTPUT_DIR:-}
install_prefix=${HOTPOT_RDWRAPPERS_INSTALL_PREFIX:-$work_dir/install}

configure_args=(
    -S "$work_dir/source/rdkit"
    -B "$work_dir/build"
    -G Ninja
    -DCMAKE_MAKE_PROGRAM="$ninja"
    -DCMAKE_C_COMPILER="$cc"
    -DCMAKE_CXX_COMPILER="$cxx"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$install_prefix"
    -DCMAKE_C_STANDARD=99
    -DCMAKE_C_STANDARD_REQUIRED=ON
    -DCMAKE_C_EXTENSIONS=OFF
    -DCMAKE_CXX_STANDARD=20
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DCMAKE_CXX_EXTENSIONS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -march=x86-64 -mtune=generic'
    '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -march=x86-64 -mtune=generic'
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
    -DCMAKE_DISABLE_FIND_PACKAGE_Catch2=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_TBB=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$work_dir/source/catch2"
    -DFETCHCONTENT_SOURCE_DIR_BETTER_ENUMS="$work_dir/source/better-enums"
    -DBoost_DIR="$lock_dir/probe/cmake"
    -DHOTPOT_RDKIT_BOOST_SOURCE="$work_dir/source/boost"
    -DEigen3_DIR="$lock_dir/probe/cmake"
    -DHOTPOT_RDKIT_EIGEN_SOURCE="$work_dir/source/eigen"
    -DCMAKE_PROJECT_INCLUDE="$script_dir/cmake/attach_rdwrapper.cmake"
    -DHOTPOT_RDKIT_SOURCE_LOCK_VERIFIED=ON
    -DHOTPOT_RDWRAPPERS_BUILD_SDK="$build_sdk"
    -DHOTPOT_RDWRAPPERS_BUILD_PYTHON="$build_python"
    -DHOTPOT_RDWRAPPERS_PYTHON_OUTPUT_DIR="$python_output"
    -DBUILD_TESTING=OFF
    -DRDK_BUILD_PYTHON_WRAPPERS=OFF
    -DRDK_BUILD_BOOST_PYTHON_WRAPPERS=OFF
    -DRDK_BUILD_NANOBIND_WRAPPERS=OFF
    -DRDK_BUILD_SWIG_WRAPPERS=OFF
    -DRDK_BUILD_CPP_TESTS=OFF
    -DRDK_BUILD_LONG_RUNNING_TESTS=OFF
    -DRDK_BUILD_CONTRIB=OFF
    -DRDK_BUILD_COMPRESSED_SUPPLIERS=OFF
    -DRDK_BUILD_INCHI_SUPPORT=OFF
    -DRDK_BUILD_AVALON_SUPPORT=OFF
    -DRDK_BUILD_PGSQL=OFF
    -DRDK_BUILD_COORDGEN_SUPPORT=OFF
    -DRDK_BUILD_MAEPARSER_SUPPORT=OFF
    -DRDK_BUILD_MOLINTERCHANGE_SUPPORT=OFF
    -DRDK_BUILD_CHEMDRAW_SUPPORT=OFF
    -DRDK_BUILD_FREESASA_SUPPORT=OFF
    -DRDK_BUILD_YAEHMOP_SUPPORT=OFF
    -DRDK_BUILD_XYZ2MOL_SUPPORT=OFF
    -DRDK_BUILD_STRUCTCHECKER_SUPPORT=OFF
    -DRDK_BUILD_PUBCHEMSHAPE_SUPPORT=OFF
    -DRDK_BUILD_DESCRIPTORS3D=OFF
    -DRDK_BUILD_FREETYPE_SUPPORT=OFF
    -DRDK_INSTALL_COMIC_FONTS=OFF
    -DRDK_BUILD_SLN_SUPPORT=OFF
    -DRDK_BUILD_MINIMAL_LIB=OFF
    -DRDK_BUILD_CFFI_LIB=OFF
    -DRDK_BUILD_FUZZ_TARGETS=OFF
    -DRDK_BUILD_RPATH_SUPPORT=OFF
    -DRDK_INSTALL_INTREE=OFF
    -DRDK_INSTALL_STATIC_LIBS=ON
    -DRDK_INSTALL_DEV_COMPONENT=OFF
    -DRDK_BUILD_THREADSAFE_SSS=ON
    -DRDK_TEST_MULTITHREADED=OFF
    -DRDK_USE_BOOST_SERIALIZATION=OFF
    -DRDK_USE_BOOST_IOSTREAMS=OFF
    -DRDK_USE_BOOST_STACKTRACE=OFF
    -DRDK_OPTIMIZE_POPCNT=OFF
    -DRDK_USE_FLEXBISON=OFF
    -DRDK_USE_UNITY_BUILDS=OFF
    -DRDK_LTO_MODE=OFF
)

if [[ "$build_python" == ON ]]; then
    python=$(command -v "$python_command")
    python_version=$($python -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$python_version" in
        3.10|3.11|3.12|3.13|3.14) ;;
        *)
            echo "rdWrappers requires Python 3.10-3.14, got $python_version" >&2
            exit 1
            ;;
    esac
    pybind11_dir=$($python -m pybind11 --cmakedir)
    configure_args+=(
        -DCMAKE_DISABLE_FIND_PACKAGE_Python=OFF
        -DPython_EXECUTABLE="$(realpath "$python")"
        -Dpybind11_DIR="$pybind11_dir"
    )
else
    configure_args+=(-DCMAKE_DISABLE_FIND_PACKAGE_Python=ON)
fi

export PATH="$toolchain/bin:/usr/bin:/bin"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH

"$cmake" "${configure_args[@]}" 2>&1 | tee "$work_dir/build/configure-output.txt"
if grep -E '^Downloading ' "$work_dir/build/configure-output.txt"; then
    echo "configure performed an unpinned network download" >&2
    exit 1
fi

targets=()
if [[ "$build_sdk" == ON ]]; then
    targets+=(hotpot_rdwrappers)
fi
if [[ "$build_python" == ON ]]; then
    targets+=(hotpot_rd_native)
fi
if [[ ${#targets[@]} -eq 0 ]]; then
    echo "at least one rdWrappers artifact must be enabled" >&2
    exit 1
fi
"$cmake" --build "$work_dir/build" --target "${targets[@]}" --parallel

printf 'rdWrappers build passed; artifacts: %s\n' "$work_dir/build"
