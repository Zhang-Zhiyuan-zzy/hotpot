#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 CACHE_DIR" >&2
    exit 2
fi

cache_dir=$1
offline=${HOTPOT_RDKIT_OFFLINE:-0}

mkdir -p "$cache_dir"

fetch_one() {
    local filename=$1
    local url=$2
    local sha256=$3
    local size=$4
    local destination="$cache_dir/$filename"
    local partial="$destination.part.$$"

    if [[ -f "$destination" ]]; then
        if [[ $(wc -c < "$destination") -ne $size ]] ||
           ! printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check --status; then
            echo "locked source has the wrong size or SHA-256: $destination" >&2
            exit 1
        fi
        printf 'verified  %s\n' "$destination"
        return
    fi

    if [[ "$offline" == 1 ]]; then
        echo "offline cache is missing locked source: $destination" >&2
        exit 1
    fi

    trap 'rm -f "$partial"' RETURN
    curl --fail --location --proto '=https' --tlsv1.2 \
        --retry 3 --retry-delay 2 \
        "$url" --output "$partial"
    if [[ $(wc -c < "$partial") -ne $size ]] ||
       ! printf '%s  %s\n' "$sha256" "$partial" | sha256sum --check --status; then
        echo "downloaded source failed its locked size or SHA-256: $url" >&2
        exit 1
    fi
    mv "$partial" "$destination"
    trap - RETURN
    printf 'fetched   %s\n' "$destination"
}

fetch_one \
    rdkit-Release_2026_09_1.tar.gz \
    https://codeload.github.com/rdkit/rdkit/tar.gz/refs/tags/Release_2026_09_1 \
    86711e340759a3a0678a2e62f54f26f7ff564c3a02de0460f60b1ce89ac1bfe9 \
    84507361

fetch_one \
    boost_1_86_0.tar.bz2 \
    https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.bz2 \
    1bed88e40401b2cb7a1f76d4bab499e352fa4d0c5f31c0dbae64e24d34d7513b \
    126220652

fetch_one \
    Catch2-v3.4.0.tar.gz \
    https://codeload.github.com/catchorg/Catch2/tar.gz/refs/tags/v3.4.0 \
    122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb \
    1112790

fetch_one \
    better-enums-c35576bed0295689540b39873126129adfa0b4c8.tar.gz \
    https://codeload.github.com/aantron/better-enums/tar.gz/c35576bed0295689540b39873126129adfa0b4c8 \
    9b78dcef7f88d1345b6f25335bfbcba5f024b08990c7d6dc605b833f4128b8dd \
    137228

fetch_one \
    eigen-3.4.0.tar.gz \
    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
    8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72 \
    2705005

fetch_one \
    RingDecomposerLib-v1.1.3_rdkit.tar.gz \
    https://codeload.github.com/rareylab/RingDecomposerLib/tar.gz/refs/tags/v1.1.3_rdkit \
    944b5816712a48bbf88aa25d4300ce11871ddf6e971218eac08f90ed2192f715 \
    137931

