# Locked native RDKit inputs and closure probe

This directory freezes the native source identity used by Stage 17. It does
not use a system, Conda, or Python-wheel RDKit SDK. `rdkit.lock.json` is the
machine-readable identity: RDKit 2026.09.1, Boost 1.86.0, every configure-time
source required by that RDKit release, the accepted Phase-1 probe toolchain,
the complete CMake configuration, and the measured `DistGeomHelpers` link
map.

RDKit upgrades are owner-only. Routine dependency maintenance must not edit
this lock, widen a version constraint, discover a newer release, or add a
multi-version compatibility branch. An upgrade begins only after an explicit
owner instruction and replaces the source lock, compatibility implementation,
closure proof, standalone tests, 187/182 evidence, and wheel matrix together.

## Why the lock contains more than RDKit and Boost

The canonical RDKit archive does not vendor its complete configure/build
closure. The pinned release otherwise fetches Catch2, better-enums, Eigen, and
RingDecomposerLib. Catch2 is configure-only with tests disabled;
better-enums and Eigen are header dependencies; RingDecomposerLib is linked
through GraphMol. All are therefore exact URL/commit/SHA inputs here. The
probe disables upstream network discovery and fails if configure reports an
unplanned download.

`ForceFieldHelpers` and `ForceField` appear in the closure because the official
ETKDG target requires them. This does not enable RDKit UFF/MMFF as a Hotpot
business optimizer.

## Offline source cache

Populate an external cache once:

```bash
third_party/rdkit/fetch_locked_sources.sh /path/to/rdkit-source-cache
```

Every download uses one literal HTTPS URL and is accepted only when both size
and SHA-256 match. Existing files are verified and are never silently
replaced. For a network-free check or build:

```bash
HOTPOT_RDKIT_OFFLINE=1 \
  third_party/rdkit/fetch_locked_sources.sh /path/to/rdkit-source-cache
```

A fully offline build requires all six archives to be present in that cache.
The archives are intentionally not committed to Git.

## Reproduce the Phase-1 closure probe

The checked-in explicit Conda specification is for this Linux x86_64 probe,
not for release wheels. The chosen prefix is disposable and is not an ABI or
path contract:

```bash
conda create --prefix /tmp/hotpot-rdkit-toolchain \
  --file third_party/rdkit/probe-toolchain-linux-64.explicit.txt

HOTPOT_RDKIT_OFFLINE=1 \
  third_party/rdkit/run_distgeom_probe.sh \
  /path/to/rdkit-source-cache \
  /tmp/hotpot-rdkit-probe-work \
  /tmp/hotpot-rdkit-toolchain
```

The work directory must be absent or empty. The script validates GCC 14.3.0,
CMake 3.31.8, Ninja 1.12.1, Linux x86_64, the generic x86-64 CPU flags,
C++20, PIC, and every archive. It builds only the transitive static closure of
the injected ETKDG executable, runs a deterministic methane embedding, checks
the compiled RDKit/Boost identities, compares the generated target map and
archive manifest with the checked-in proofs, and rejects a shared RDKit or
Boost dependency.

The accepted 2026-10-22 probe built 158 compile/link steps and reported:

```text
rdkit=2026.09.1 boost=1_86 conformers=1
```

The exact target graph is in `probe/expected-target-link-map.txt`; its 18
static archives are in `probe/expected-static-archives.txt`.

## Release-toolchain boundary

The local probe uses a pinned glibc-2.28 Conda sysroot to prove that the source
closure compiles. It is not the future manylinux wheel toolchain. Wheel work
must separately pin each manylinux image by digest, retain the declared CPU
and platform ABI baseline, and rerun this closure proof plus wheel auditing and
Python-RDKit/Open-Babel coexistence tests. No temporary local prefix may enter
the public SDK contract.

The source licenses and license-file hashes are recorded beside each archive
in `rdkit.lock.json`; release packaging must carry the corresponding notices.
