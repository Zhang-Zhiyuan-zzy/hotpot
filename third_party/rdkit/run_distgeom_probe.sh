#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 CACHE_DIR EMPTY_WORK_DIR TOOLCHAIN_PREFIX" >&2
    exit 2
fi

cache_dir=$(realpath -m "$1")
work_dir=$(realpath -m "$2")
toolchain=$(realpath "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -e "$work_dir" ]] && [[ -n $(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "probe work directory must be absent or empty: $work_dir" >&2
    exit 1
fi
mkdir -p "$work_dir"/{source,build}

cmake=$toolchain/bin/cmake
ninja=$toolchain/bin/ninja
cc=$toolchain/bin/x86_64-conda-linux-gnu-gcc
cxx=$toolchain/bin/x86_64-conda-linux-gnu-g++

[[ $($cmake --version | head -1) == "cmake version 3.31.8" ]]
[[ $($ninja --version) == "1.12.1" ]]
[[ $($cc -dumpfullversion -dumpversion) == "14.3.0" ]]
[[ $($cxx -dumpfullversion -dumpversion) == "14.3.0" ]]
[[ $(uname -m) == "x86_64" ]]

"$script_dir/fetch_locked_sources.sh" "$cache_dir"

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

extract_gzip "$cache_dir/rdkit-Release_2026_09_1.tar.gz" "$work_dir/source/rdkit"
extract_bzip2 "$cache_dir/boost_1_86_0.tar.bz2" "$work_dir/source/boost"
extract_gzip "$cache_dir/Catch2-v3.4.0.tar.gz" "$work_dir/source/catch2"
extract_gzip \
    "$cache_dir/better-enums-c35576bed0295689540b39873126129adfa0b4c8.tar.gz" \
    "$work_dir/source/better-enums"
extract_gzip "$cache_dir/eigen-3.4.0.tar.gz" "$work_dir/source/eigen"
extract_gzip \
    "$cache_dir/RingDecomposerLib-v1.1.3_rdkit.tar.gz" \
    "$work_dir/source/rdkit/External/RingFamilies/RingDecomposerLib"

export PATH="$toolchain/bin:/usr/bin:/bin"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH

"$cmake" -S "$work_dir/source/rdkit" -B "$work_dir/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$ninja" \
    -DCMAKE_C_COMPILER="$cc" \
    -DCMAKE_CXX_COMPILER="$cxx" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_STANDARD=99 \
    -DCMAKE_C_STANDARD_REQUIRED=ON \
    -DCMAKE_C_EXTENSIONS=OFF \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_VISIBILITY_PRESET=hidden \
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
    -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG -march=x86-64 -mtune=generic' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -march=x86-64 -mtune=generic' \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Catch2=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_Python=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_TBB=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_SOURCE_DIR_CATCH2="$work_dir/source/catch2" \
    -DFETCHCONTENT_SOURCE_DIR_BETTER_ENUMS="$work_dir/source/better-enums" \
    -DBoost_DIR="$script_dir/probe/cmake" \
    -DHOTPOT_RDKIT_BOOST_SOURCE="$work_dir/source/boost" \
    -DEigen3_DIR="$script_dir/probe/cmake" \
    -DHOTPOT_RDKIT_EIGEN_SOURCE="$work_dir/source/eigen" \
    -DCMAKE_PROJECT_INCLUDE="$script_dir/probe/attach_probe.cmake" \
    -DBUILD_TESTING=OFF \
    -DRDK_BUILD_PYTHON_WRAPPERS=OFF \
    -DRDK_BUILD_BOOST_PYTHON_WRAPPERS=OFF \
    -DRDK_BUILD_NANOBIND_WRAPPERS=OFF \
    -DRDK_BUILD_SWIG_WRAPPERS=OFF \
    -DRDK_BUILD_CPP_TESTS=OFF \
    -DRDK_BUILD_LONG_RUNNING_TESTS=OFF \
    -DRDK_BUILD_CONTRIB=OFF \
    -DRDK_BUILD_COMPRESSED_SUPPLIERS=OFF \
    -DRDK_BUILD_INCHI_SUPPORT=OFF \
    -DRDK_BUILD_AVALON_SUPPORT=OFF \
    -DRDK_BUILD_PGSQL=OFF \
    -DRDK_BUILD_COORDGEN_SUPPORT=OFF \
    -DRDK_BUILD_MAEPARSER_SUPPORT=OFF \
    -DRDK_BUILD_MOLINTERCHANGE_SUPPORT=OFF \
    -DRDK_BUILD_CHEMDRAW_SUPPORT=OFF \
    -DRDK_BUILD_FREESASA_SUPPORT=OFF \
    -DRDK_BUILD_YAEHMOP_SUPPORT=OFF \
    -DRDK_BUILD_XYZ2MOL_SUPPORT=OFF \
    -DRDK_BUILD_STRUCTCHECKER_SUPPORT=OFF \
    -DRDK_BUILD_PUBCHEMSHAPE_SUPPORT=OFF \
    -DRDK_BUILD_DESCRIPTORS3D=OFF \
    -DRDK_BUILD_FREETYPE_SUPPORT=OFF \
    -DRDK_INSTALL_COMIC_FONTS=OFF \
    -DRDK_BUILD_SLN_SUPPORT=OFF \
    -DRDK_BUILD_MINIMAL_LIB=OFF \
    -DRDK_BUILD_CFFI_LIB=OFF \
    -DRDK_BUILD_FUZZ_TARGETS=OFF \
    -DRDK_BUILD_RPATH_SUPPORT=OFF \
    -DRDK_INSTALL_INTREE=OFF \
    -DRDK_INSTALL_STATIC_LIBS=ON \
    -DRDK_INSTALL_DEV_COMPONENT=OFF \
    -DRDK_BUILD_THREADSAFE_SSS=ON \
    -DRDK_TEST_MULTITHREADED=OFF \
    -DRDK_USE_BOOST_SERIALIZATION=OFF \
    -DRDK_USE_BOOST_IOSTREAMS=OFF \
    -DRDK_USE_BOOST_STACKTRACE=OFF \
    -DRDK_OPTIMIZE_POPCNT=OFF \
    -DRDK_USE_FLEXBISON=OFF \
    -DRDK_USE_UNITY_BUILDS=OFF \
    -DRDK_LTO_MODE=OFF \
    2>&1 | tee "$work_dir/build/configure-output.txt"

if grep -E '^Downloading ' "$work_dir/build/configure-output.txt"; then
    echo "configure performed an unpinned network download" >&2
    exit 1
fi

"$cmake" --build "$work_dir/build" --target hotpot_distgeom_probe --parallel
"$work_dir/build/hotpot_distgeom_probe" | tee "$work_dir/build/probe-output.txt"
"$ninja" -C "$work_dir/build" -t commands hotpot_distgeom_probe \
    > "$work_dir/build/probe-commands.txt"
find "$work_dir/build" -type f -name '*.a' -printf '%f\n' \
    | sort > "$work_dir/build/built-static-archives.txt"
cmp \
    "$script_dir/probe/expected-static-archives.txt" \
    "$work_dir/build/built-static-archives.txt"
cmp \
    "$script_dir/probe/expected-target-link-map.txt" \
    "$work_dir/build/distgeom-target-link-map.txt"
ldd "$work_dir/build/hotpot_distgeom_probe" \
    | tee "$work_dir/build/probe-ldd.txt"

if grep -Ei '(^|[[:space:]])lib(RDKit|boost)' "$work_dir/build/probe-ldd.txt"; then
    echo "probe unexpectedly depends on a shared RDKit or Boost library" >&2
    exit 1
fi
grep -F -- '-std=c++20' "$work_dir/build/compile_commands.json" >/dev/null
grep -F -- '-fPIC' "$work_dir/build/compile_commands.json" >/dev/null
grep -F -- '-march=x86-64' "$work_dir/build/compile_commands.json" >/dev/null

printf 'probe passed; artifacts: %s\n' "$work_dir/build"


