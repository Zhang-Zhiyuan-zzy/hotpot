# Building native rdWrappers

`rdWrappers` is intentionally not configurable against a system, Conda, or
Python-wheel RDKit SDK. The production driver accepts only the six archives
whose sizes and SHA-256 values are frozen in
`third_party/rdkit/rdkit.lock.json`, and it always invokes the source fetcher
in offline mode:

```bash
hotpot/cheminfo/rdWrappers/build_locked.sh \
  /path/to/populated/source-cache \
  /path/to/new-empty-work-directory \
  /path/to/python
```

The source cache and new empty work directory are required. Python is required
unless `HOTPOT_RDWRAPPERS_BUILD_PYTHON=OFF`; it must be CPython 3.10-3.14 with
pybind11 3 installed. The driver never resolves Python for an SDK-only build.
The strict local probe selects its pinned toolchain through
`HOTPOT_RDWRAPPERS_BUILD_PROFILE=probe` and
`HOTPOT_RDWRAPPERS_TOOLCHAIN_PREFIX`; the exact invocation is documented in
`third_party/rdkit/README.md`. Release wheels use the separately pinned
`manylinux_2_28_x86_64` profile and audit described there and by Plan 17.

The Phase-1 Conda compiler uses a newer private `libstdc++` and attempts to
write its temporary prefix into every ELF RPATH. The wrapper build removes
all build RPATH entries after linking and fails if an absolute RPATH remains.
Running that development artifact may therefore require an explicit
`LD_LIBRARY_PATH` pointing at the disposable probe runtime. This is a local
test limitation, not an SDK or wheel runtime contract; those artifacts must be
rebuilt with the pinned manylinux release toolchain.

The build produces two independent artifacts from the same source objects:

- `libhotpot_rdwrappers.so.0`, an installable C ABI plus header-only C++17
  facade; and
- `_rd_native`, a Python extension that statically absorbs the private RDKit
  closure and therefore has no runtime dependency on the SDK shared library.

Install them separately so a wheel is not presented as the C++ SDK:

```bash
cmake --install /path/to/work/build \
  --component rdwrappers_sdk --prefix /path/to/sdk-prefix
cmake --install /path/to/work/build \
  --component rdwrappers_python --prefix /path/to/wheel-staging
```

External C++17 consumers use:

```cmake
find_package(HotpotRDWrappers 0.5.4.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE Hotpot::rdwrappers)
```

Only `hotpot_rd_*` is exported by the SDK library. Only
`PyInit__rd_native` is exported by the extension. Private RDKit headers are
never installed.
