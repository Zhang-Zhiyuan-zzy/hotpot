# Stage 17 implementation log

## Recording policy

This log is updated during implementation, not reconstructed afterward. Every
step records the intended result, performed work, verification, unexpected
findings, plan revisions, and Git checkpoint.

The normative constraints are:

- one Hotpot version supports exactly RDKit 2026.09.1
  (`rdkit==2026.9.1`), and the pin changes only on explicit owner instruction;
- `rdWrappers` remains an independently runnable Python and C++ component;
- `obWrappers` remains available even if its builder is no longer selected by
  the composed force-field route;
- standalone conformer generation performs no Hotpot geometry gate,
  perturbation, or force-field optimization;
- the Stage 17 force-field experiment may replace only the initial coordinate
  provider and must not reorder, remove, or weaken any later stage; and
- stereochemistry is separated into exact uninterpreted Level-1 transport and
  Level-2 chemical validation, assignment, and mapping-aware alignment.

When an implementation finding requires a plan change, the complete normative
plan is reread before the change. The revision and its system-wide effect are
recorded here before implementation continues.

## Baseline

| Item | Result |
| --- | --- |
| Authorization date | 2026-10-22 |
| Working branch | `feature/rdWrapper` |
| Worktree | `/tmp/hotpot-rdwrapper` |
| Implementation starting commit | `9640d906a9d13e969152d0f3a4d92a3a16b9e0ce` |
| Planning baseline | `9640d906a9d13e969152d0f3a4d92a3a16b9e0ce` |
| Approved-plan SHA-256 | `1c502f8ed8a48589d7fc67f9d0dd0bf8b882f8ab017a4ef4fdc4617ed2fc4604` |
| Host | Linux 5.15.0-139-generic x86_64, glibc platform |
| Python | CPython 3.11.16, GCC 14.3.0 runtime build |
| Native compiler | GCC 9.4.0 |
| CMake | 3.16.3 |
| NumPy | 2.4.6 |
| NetworkX | 3.6.1 |
| Open Babel | Python/runtime 3.2.1 |
| RDKit runtime | canonical `2026.09.1`; package version `2026.9.1` |
| Existing dependency declaration | `rdkit>=2025.9,<2027` in `pyproject.toml` and `requirements.txt` |
| Existing native build | setuptools/pybind11; four in-tree C++17 extensions; Open Babel wheel headers/libs |
| Frozen benchmark cohorts | 187 input ligands; 182 CBond-positive complexes |
| Published 182 manifest | Missing locally; recorded SHA-256 `4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377` |
| Development-only 182 manifest | Eu smoke artifact, 182 cases; not accepted as the formal cohort |

## Step record

### Step 0 - Freeze authorization and establish the live record

Status: complete

Planned:

- mark the reviewed plan as approved;
- create this live implementation log;
- record repository, environment, dependency, and cohort identities;
- audit the actual native build, molecule/stereo model, and benchmark entry
  points before production edits; and
- execute focused and 187/182 baselines before refactoring shared boundaries.

Performed:

- reread the complete 1,207-line implementation plan after owner approval;
- marked the plan and Stage README as implementation in progress;
- recorded the baseline table above; and
- started independent read-only audits of native/Open Babel construction,
  stereo/data-model transport, and RDKit/benchmark/build feasibility.
- created an isolated temporary setuptools 81/pybind11 3 build dependency
  directory and successfully built all unchanged native extensions in this
  worktree; and
- ran the focused baseline suite after the build: 292 passed, 5 skipped, and 6
  failed after 757.79 s. All six failures were writes to the absent
  `tests/outdir/cheminfo` directory, not product assertions; and
- created that expected ignored output directory and reran exactly the six
  affected tests: 6 passed with 10 pre-existing complex-geometry warnings in
  138.34 s. The effective focused baseline is therefore 298 passed and 5
  skipped, with no product assertion failure.

Verification:

- `git status --short --branch` confirms the dedicated
  `feature/rdWrapper` worktree;
- runtime version probes confirm RDKit `2026.09.1` and Open Babel `3.2.1`;
- `assets/readme/coordination_complex_backend_comparison.json` and the
  maintained benchmark package identify the frozen 187/182 cohorts; and
- the approved plan hash is recorded above;
- the implementation-grounded revision contains 1,348 lines and has SHA-256
  `66e4b38ba793ef58e67816971072819eee1b9ee6c6cf368148e77d115ee785ff`;
  and
- the directory-fixed six-test rerun completed successfully.

Unexpected findings:

- the first test attempt stopped during collection because a fresh worktree
  does not contain native extension artifacts; the unchanged source had to be
  built before a meaningful baseline;
- the repository has no exported CMake target or installed C++ SDK;
- the proposed `std::span` API required C++20 while current Hotpot native code
  and public source consumers are C++17;
- RDKit 2026.09.1 itself requires CMake >=3.18, C++20, Boost >=1.81, and its
  official ETKDG target transitively compiles ForceFieldHelpers;
- this host has CMake 3.16.3/GCC 9.4.0, so a pinned project-local modern native
  toolchain and build probe are required;
- `rdkit==2026.9.1` provides CPython 3.10-3.15 wheels, no CPython 3.9 wheel,
  and no sdist; strict version pinning is incompatible with Hotpot's former
  Python 3.9 claim;
- the local Python RDKit is 2026.09.1 but its separately installed C++
  headers/libs are 2026.03.5 and therefore unusable for this implementation;
- the published 182-case manifest cannot currently be found; only its expected
  digest survives. A different Eu smoke manifest is suitable only for
  development;
- current Hotpot core/rdconvert loses tetrahedral/alkene stereo, isotope,
  radical, and distinct implicit/explicit-H facts;
- the reviewed Level-1 stereo records could not express non-atom carriers,
  enhanced groups, axial/coordination types, or unknown source values;
- ordinary Molecule I/O is Open-Babel-backed and cannot be the universal
  lossless stereo serialization contract;
- `ComplexSessionInput` is a widely consumed flat aggregate, so physical
  nesting would cause broad unrelated source/API churn; and
- with the current third-party wheels, RDKit/Hotpot-first then Open Babel is
  stable, while OpenBabel-first then Python RDKit exits with SIGSEGV 139.

Adjustment:

- completed a full reread of the normative plan immediately before revision;
- kept existing modules C++17 and replaced `std::span` with a C ABI plus
  header-only C++17 facade; only the private RDKit core compiles as C++20;
- specified separate Python-wheel and C++-SDK artifacts around one private
  core/C API implementation;
- narrowed the exact-RDKit Hotpot support matrix to Python 3.10-3.14;
- preserved flat `ComplexSessionInput` and planned zero-copy common views;
- redesigned Level-1 stereo as a versioned, extensible
  anchor/carrier/descriptor/group payload with separate atom/bond transport
  facts;
- made the versioned Hotpot DTO the lossless serialization authority and
  demoted SMILES/SDF/MOL2 to capability-scoped adapters;
- split Level-2 stereo into seven ordered semantic gates;
- recorded the missing formal cohort as a prerequisite rather than silently
  replacing it; and
- changed import-order closure to the supported RDKit/Hotpot-first order while
  retaining the reverse-order crash as an isolated diagnostic.

Git checkpoint: `066e6b2 docs(plan17): begin rdWrapper implementation`.

### Plan revision 1 - Align the reviewed design with executable repository facts

Status: complete

Planned:

- preserve every owner decision while correcting implementation assumptions
  that do not match the actual compiler, ABI, packaging, data model, or
  benchmark assets.

Performed:

- reread all 1,207 original plan lines in five contiguous sections;
- incorporated the native-build, stereo-model, and benchmark audits into the
  normative plan; and
- revised API/build boundaries, support matrix, Level-1 payload, Phase 0-3,
  Phase 7-8, test obligations, packaging, and benchmark provenance together.

Verification:

- the revised design retains native Python/C++ `rdWrappers`, exact RDKit
  2026.09.1, independent `obWrappers`, the unchanged force-field workflow,
  component-level integration, and two-level Hotpot stereo semantics;
- the C ABI removes STL/RDKit/Boost types from the SDK DSO boundary; and
- no production code was changed during the audit/revision step.

Unexpected findings: recorded under Step 0.

Adjustment: the plan is now an implementation-grounded revision rather than a
pure prospective design. The approved scientific and product semantics are
unchanged.

Git checkpoint: pending with Step 0 documentation.

### Step 1 - Pin the Python release contract

Status: complete

Planned:

- replace the moving RDKit range with the exact owner-controlled package;
- update the supported Python range coherently across metadata, CI, wheel,
  Docker, Conda, compatibility scripts, tests, and maintained documentation;
  and
- add an executable packaging contract that prevents either identity from
  drifting.

Performed:

- pinned the Python dependency to `rdkit==2026.9.1`;
- changed the maintained Python range from 3.9-3.14 to 3.10-3.14 across the
  release surfaces listed above;
- removed the obsolete Python-3.9-only Open Babel 3.1 dependency branch so all
  supported interpreters use Open Babel 3.2.x; and
- added metadata tests for the exact interpreter classifiers and RDKit pin.

Verification:

- `tests/test_packaging_metadata.py`: 9 passed;
- both modified shell entry points pass `bash -n`;
- TOML and YAML parsing pass;
- both Docker entry points reject Python 3.9 with exit code 2; and
- `git diff --check` passes for this change set.

Unexpected findings:

- `requirements.txt` omitted the already-declared project dependency
  `cryptography>=44,<47`, causing the existing requirements/metadata parity
  test to fail. The missing declaration was restored in the same metadata
  checkpoint; it does not change the authoritative dependency range; and
- the unqualified `pytest` executable resolves to a user-level system-Python
  installation without RDKit, while `python` resolves to the project Conda
  environment. Verification therefore uses `python -m pytest`, matching the
  interpreter used for the source baseline.

Adjustment:

- historical Stage reports and archived compatibility evidence that mention
  Python 3.9 remain unchanged and are explicitly described as historical;
- no compatibility branch was added for Python 3.9 or for another RDKit
  version.

Git checkpoint: `39978fb build: pin RDKit and Python release matrix`.

### Step 2 - Freeze the standalone Python rdWrappers API

Status: complete

Planned:

- define the reviewed public value contracts and callable signatures before
  native sampler behavior; and
- make the absence of a built native backend explicit during this intermediate
  checkpoint.

Performed:

- added frozen contracts for ETKDG options, coordination constraints,
  conformer records/batches, runtime identity, and generation status;
- added `ETKDGSampler`, `generate_conformers()`, and `runtime_info()` with the
  reviewed signatures; and
- added an internal lazy native dispatcher. It raises a direct runtime error
  while `_rd_native` is not yet built rather than silently using Python RDKit
  or another sampler.

Verification:

- `tests/test_cheminfo/test_rdwrapper_api.py`: 5 passed;
- Python byte-code compilation passes; and
- `git diff --check` passes for the new package and test.

Unexpected findings: none.

Adjustment:

- `ETKDGOptions` is explicitly keyword-only, matching the approved public
  constructor rather than relying on the default positional dataclass form;
- no native behavior, Python-RDKit molecule translation, geometry gate,
  perturbation, or force-field optimization was added in this checkpoint.

Git checkpoint: `c232d1d feat(rdwrappers): freeze Python sampler API`.

### Step 3 - Freeze the two-level stereochemistry API

Status: complete

Planned:

- replace the placeholder module with the reviewed Level-1 transport types
  and Level-2 semantic reports/functions; and
- keep storage/remapping and chemical interpretation unimplemented until their
  separate phases.

Performed:

- added atom/bond anchors, atom/non-atom carriers, enhanced-group identity,
  extensible source descriptor records, and atom/bond transport facts;
- added Level-2 stereo families, issue codes, validation/assignment/alignment
  report contracts, and the reviewed public function signatures; and
- made the intermediate function bodies raise phase-specific
  `NotImplementedError` instead of performing partial or toolkit-defined
  chemistry.

Verification:

- the combined rdWrappers/stereo API suite: 10 passed;
- Python byte-code compilation passes; and
- `git diff --check` passes for the module and contract test.

Unexpected findings: none.

Adjustment:

- Level-1 and Level-2 contracts are frozen together, but their implementations
  remain deliberately separated. No RDKit or Open Babel enum is used as a
  Hotpot semantic identity.

Git checkpoint: pending.
