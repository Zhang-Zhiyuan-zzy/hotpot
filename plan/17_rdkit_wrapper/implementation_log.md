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

Git checkpoint: `18d26ca feat(stereo): freeze transport and semantic APIs`.

### Step 4 - Add the consumer-neutral C++17 molecule view

Status: complete

Planned:

- introduce `ArrayView`, `MoleculeBufferView`, `OwnedMoleculeData`, and
  `ConformerBatchView` without connecting a production consumer; and
- separate generic structural validation from toolkit compatibility policy.

Performed:

- added header-only C++17 pointer/length views and explicit atom/bond counts;
- added a presence mask so a present zero-length field remains distinct from
  an absent field;
- added generic bond-kind codes including zero-order and dative bonds;
- added field-size, index, finiteness, and structural validation; and
- kept topology storage separate from K x N x 3 conformer coordinates.

Verification:

- the standalone C++17 source/API fence compiles with `-Wall -Wextra -Werror`
  and executes successfully: 1 passed;
- the fence proves borrowed array identity, present-empty versus absent field
  semantics, independent conformer batches, and acceptance of a structurally
  valid dative bond by the generic layer.

Unexpected findings: none.

Adjustment:

- this checkpoint has no Open Babel, force-field, pybind, or RDKit consumer;
  Open Babel's dative incompatibility remains in its existing implementation
  until the dedicated adapter migration.

Git checkpoint: `1707bbf feat(native): add generic molecule buffer views`.

### Step 5 - Add field-selective Python molecule buffers

Status: complete

Planned:

- add the typed Python counterpart of the common native view without changing
  an existing consumer; and
- prove that selected fields, absent fields, and empty molecular dimensions
  remain distinct.

Performed:

- added `NativeMoleculeField`, toolkit-neutral bond codes,
  `NativeMoleculeBuffers`, and `pack_molecule()` under the private common
  native package;
- made atom/bond counts unconditional while every data field is independently
  optional; and
- materialized only requested arrays as exact contiguous NumPy dtypes.

Verification:

- combined Python-buffer and C++17-view suite: 5 passed;
- the tests cover an empty field selection, selective packing, contiguous
  dtype/shape contracts, present-empty versus absent bond tables, and generic
  zero/dative/unknown codes;
- Python byte-code compilation and `git diff --check` pass.

Unexpected findings: none.

Adjustment:

- consumer field sets have not yet been moved. Existing Open Babel and
  force-field paths still use their original packer in this checkpoint.

Git checkpoint: `8f269cb feat(native): add field-selective molecule buffers`.

### Benchmark evidence audit 1 - Recover the formal 182-case provenance

Status: blocked on artifact recovery or owner approval for a new identity

Planned:

- recover and hash-check the published Am 182-case canonical manifest with
  SHA-256
  `4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377`;
  and
- never substitute the development Eu smoke cohort for formal evidence.

Performed:

- recovered the original generation chain from the archived Codex thread
  `01a0eb4a-247e-72b0-989a-59df5f4d4da0`;
- identified reference commit `11c0e3d0086957645053023d692723ffd4ef7fae`,
  RDKit generator-fix commit `b8e085d`, and publication-record commit
  `1d87d7436dd84995d2b6b0e33a3418b77fd809f2`;
- recovered the formal input digest, seed, thresholds, optimization settings,
  worker count, and exact generator/reference commands; and
- searched all reachable and unreachable Git blobs by content, Git history,
  the home directory and Trash, and three known repository backup archives.

Verification:

- the formal artifact was generated at
  `movie/benchmarks/extractants_am_187_four_workflows_20261009/rdkit_am/`
  `canonical_cases.json`; the sibling `rdkit` directory was the failed run
  before the Am mapping fix;
- the formal reference produced 182 successes from 187 inputs and excluded
  cases 139, 182, 185, 186, and 187; and
- no searched file, archive member, or Git blob matches the published digest.

Unexpected findings:

- the formal artifact was never committed and its file, reference manifest,
  per-case reports, and trajectories are no longer present;
- the same byte identity cannot be recreated by rerunning the recovered
  command because the artifact includes a microsecond `created_at`, absolute
  reference identity, per-case runtime, validation payloads, convergence, and
  trajectory-derived topology digests.

Adjustment:

- engineering development may use the explicitly labelled Eu smoke cohort;
- no formal benchmark claim will be made from that smoke artifact;
- formal closure requires either an external backup of the old canonical
  artifact (or its complete reference evidence), or explicit owner approval
  to create a deterministic semantic cohort v2 with a new digest and without
  volatile path/time/runtime identity fields.

Git checkpoint: `9400224 docs(plan17): record formal cohort recovery audit`.

### Step 6 - Freeze the standalone C and C++ SDK surface

Status: complete

Planned:

- define a versioned C ABI containing only fixed-width scalars, opaque
  handles, and borrowed pointer/count views; and
- provide the reviewed header-only C++17 facade without exporting STL or
  toolkit types from a shared library.

Performed:

- added C ABI v1 options, molecule/constraint/stereo views, opaque sampler and
  result lifetimes, result/runtime accessors, and explicit status codes;
- reserved complete Level-1 stereo carrier, anchor, group, atom-fact, and
  bond-fact transport in the v1 input contract before native implementation;
- added the C++17 `ETKDGSampler`, free generation function, value contracts,
  RAII lifetime handling, and runtime identity facade; and
- kept all C++ container/string construction inside the consumer translation
  unit.

Verification:

- the C header compiles as strict C11;
- the same header-only C++17 facade compiles and executes with both
  `_GLIBCXX_USE_CXX11_ABI=0` and `=1` against one mocked C ABI;
- the combined common-view and SDK source fences: 4 passed.

Unexpected findings:

- the first mock runtime fence encoded the nine-character string
  `2026.09.1` with length ten, correctly exposing the C API's explicit-length
  behavior. The test fixture now derives the length without the terminating
  NUL; no API signature or runtime rule changed.

Adjustment:

- this checkpoint supplies headers and executable contract tests only. The
  private RDKit core and exported shared-library symbols remain Phase 3 work;
- the Python wheel remains separate from the installed C++ SDK artifact.

Git checkpoint: `356fda8 feat(rdwrappers): freeze C and C++ SDK APIs`.

### Step 7 - Migrate the Python Open Babel packer through the common schema

Status: complete

Planned:

- preserve every existing Open Babel packing name and value while replacing
  its duplicate array construction with the common field-selective packer;
  and
- avoid changing the public `obWrappers` API or any C++ consumer in this step.

Performed:

- forwarded `_MoleculeBuffers` to `NativeMoleculeBuffers`;
- retained the existing `_BOND_KIND_CODES`, schema-version, empty-matrix, and
  `_pack_molecule` compatibility names;
- retained `BondKindCode` instances at the Open Babel compatibility surface;
  and
- made the Open Babel adapter request the complete common field set.

Verification:

- independent common/packing/native-engine rerun: 25 passed;
- worker full `obWrappers` suite: 46 passed, 1 skipped;
- worker native force-field/session regression selection: 60 passed;
- `git diff --check` passes.

Unexpected findings:

- the common DTO adds read-only `atom_count` and `bond_count` fields. No
  repository consumer constructs the former private DTO directly, and all
  former fields retain their names, dtypes, shapes, order, and values.

Adjustment:

- no pybind constructor, C++ molecule type, Open Babel compatibility policy,
  or force-field control flow changed in this checkpoint.

Git checkpoint: `3b3a2b0 refactor(obwrappers): reuse common molecule packing`.

### Plan revision 2 - Close stereo lifecycle and native-source identities

Status: complete

Planned:

- resolve implementation findings that made the first Level-1 DTO ambiguous
  under component assembly and explicit-hydrogen materialization;
- repair the plan as one system rather than patching only the stereo section;
  and
- incorporate the complete offline CMake source graph and recovered formal
  benchmark evidence.

Performed:

- reread all 1,348 lines of plan revision 1 in five contiguous sections before
  making any revision;
- replaced nullable enhanced-group kind/id fields with a single persistent
  `StereoGroupRef(scope, kind, id)` contract;
- split hydrogen transport into immutable source representation and current
  graph state, with only current state admitted to native construction;
- specified one ordered pure remapping policy and every affected molecule/
  working-copy lifecycle operation;
- made hidden-bond ownership, incident-bond cleanup, and component atom-array
  independence explicit Phase-2 prerequisites;
- expanded the C ABI/common transport and test obligations consistently with
  the revised Level-1 DTO;
- updated the source-build design from a guessed RDKit/Boost list to the exact
  six-archive offline closure and generated target map; and
- replaced the no-longer-possible original-manifest recovery step with the two
  owner-visible formal closure paths.

Verification:

- the revised normative plan contains 1,522 lines;
- SHA-256 is
  `8955113b0e2846f1e4934d12866b4c75e7c776af8f9d5013354fb28a91681286`;
- `git diff --check` passes; and
- public API, common/native transport, lifecycle, phase order, test duties,
  packaging, benchmark provenance, and approved decisions were revised
  together.

Unexpected findings:

- equal enhanced-group IDs from independent molecules were indistinguishable
  from a true group split across components;
- copying/assembling hidden bonds and component attribute-array ownership were
  already incorrect before stereo storage;
- one hydrogen field could not serve both source fidelity and current native
  graph construction without double-counting after explicit-H materialization;
- the pinned minimal RDKit configuration also requires Eigen, better-enums,
  Catch2, and RingDecomposerLib source identities; upstream RingDecomposerLib
  used only MD5 and therefore required a Hotpot SHA-256 lock.

Adjustment:

- Level-1 implementation resumes only from this complete revision;
- the already-created unpublished C ABI and Python contract tests will be
  updated before native behavior, so no released ABI is changed retroactively;
- Level-2 semantic gates, force-field workflow order, standalone sampler
  boundaries, and owner-controlled RDKit pin remain unchanged.

Git checkpoint: pending.
