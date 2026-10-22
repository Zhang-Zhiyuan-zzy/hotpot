# Native rdWrappers third-party notices

The native `rdWrappers` binary is built only from the exact source closure in
`third_party/rdkit/rdkit.lock.json`:

- RDKit 2026.09.1 (`BSD-3-Clause`);
- Boost 1.86.0 headers (`BSL-1.0`);
- Eigen 3.4.0 headers (`MPL-2.0` and component licenses);
- better-enums commit `c35576bed0295689540b39873126129adfa0b4c8`
  (`BSD-2-Clause`);
- Catch2 3.4.0, used only while configuring the pinned source graph
  (`BSL-1.0`); and
- RingDecomposerLib 1.1.3_rdkit (`BSD-3-Clause`).

The SDK installer places the corresponding upstream license files beside this
notice. The Python extension statically absorbs the same private RDKit closure;
it does not link a system, Conda, or Python-wheel RDKit SDK.
