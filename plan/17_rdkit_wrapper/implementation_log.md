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

Git checkpoint: `7f19061 docs(plan17): revise stereo lifecycle and source closure`.

### Step 8 - Move Open Babel policy behind the common C++ view

Status: complete

Planned:

- add a zero-copy common view to the existing owning `MoleculeData` without
  changing its fields, aggregate layout, pybind constructor, or public
  `obWrappers` behavior;
- move dative and required-field policy into an Open Babel adapter validator;
  and
- preserve the former failure timing, text, and mixed-invalid first-error
  order.

Performed:

- forwarded `BondKind` to the toolkit-neutral common enum and added
  `MoleculeData::view()` over the existing vectors;
- made `MoleculeData::validate()` consumer-neutral so a structurally valid
  dative bond is accepted by the common layer;
- added `validate_openbabel_compatibility()` and `make_obmol(view)`, retaining
  the former owning overload as a forwarding adapter;
- called Open Babel compatibility validation at every former native-engine
  early-validation point and at the pybind construction boundary; and
- retained the former per-bond index/self/order/dative validation order rather
  than inheriting the generic validator's column order.

Verification:

- clean rebuild of all four native extensions succeeded after the enum type
  identity changed;
- standalone C++ policy fence proves zero-copy pointers, generic dative
  acceptance, Open Babel dative rejection text/order, and unchanged ordinary
  OBMol construction;
- final common-view plus full `obWrappers` regression: 49 passed, 1 skipped in
  6.83 s; the skip is the existing absent managed extractant dataset;
- `git diff --check` and Python compilation pass.

Unexpected findings:

- naively running the generic validator first changed the first error for a
  payload containing an early dative bond and a later invalid index. The
  adapter now owns the complete legacy order, followed by generic validation,
  so behavior is unchanged;
- an attempted unqualified `pytest` again selected the user-level system
  interpreter without RDKit. No new dependency was installed; verification
  used the project interpreter through `python -m pytest`.

Adjustment:

- common `OwnedMoleculeData` ownership and flat force-field zero-copy views
  remain later, separate checkpoints. This commit changes adapter policy
  placement only and does not change force-field data flow.

Git checkpoint: `2625072 refactor(obwrappers): isolate native compatibility policy`.

### Step 9 - Refreeze Level-1 group and hydrogen transport contracts

Status: complete

Planned:

- apply plan revision 2 to the unpublished Python and C ABI contracts before
  implementing storage, packing, or RDKit graph behavior.

Performed:

- added frozen `StereoGroupRef(scope, kind, id)` and replaced the ambiguous
  nullable group-kind/group-id pair with one optional group reference;
- added frozen `HydrogenTransportState` and split atom transport into current
  hydrogen state, immutable source hydrogen state, and source namespace;
- mirrored group scope and both hydrogen states in fixed-width C ABI v1
  structs and presence masks; and
- retained isotope/radical atom facts and bond display-direction facts as
  independent transport values.

Verification:

- combined Python rdWrappers/stereo API suite: 11 passed;
- strict C11 plus ABI0/ABI1 C++17 SDK facade suite: 3 passed;
- the C++ fence asserts standard-layout stereo/group/hydrogen C structs;
- Python compilation and `git diff --check` pass.

Unexpected findings: none beyond those recorded in plan revision 2.

Adjustment:

- C ABI v1 has not been released or implemented by a shared library, so this
  is a pre-publication contract correction rather than an ABI break;
- Level-1 functions still raise their phase-specific `NotImplementedError`;
  no partial storage or chemistry behavior was introduced.

Git checkpoint: `2801e79 feat(stereo): scope groups and split hydrogen state`.

### Step 10 - Lock and prove the native RDKit source closure

Status: complete

Planned:

- pin the exact source identity for RDKit 2026.09.1 and every source fetched
  by its minimal ETKDG build;
- prove the private C++20/PIC closure without using the system, Conda, or
  Python-wheel RDKit SDK; and
- keep the local proof toolchain distinct from the future release-wheel
  toolchain and public SDK contract.

Performed:

- locked RDKit, Boost, Catch2, better-enums, Eigen, and RingDecomposerLib by
  literal URL, revision, byte size, SHA-256, license identity, and license-file
  digest;
- added verified online/offline cache acquisition and a fully disconnected
  CMake/Ninja closure probe;
- pinned the Linux x86-64 probe compiler/sysroot/CMake/Ninja environment with
  a complete Conda explicit specification;
- disabled unrelated upstream components and network discovery, injected the
  preverified dependencies, and generated the exact transitive target map and
  static-archive manifest; and
- documented the owner-only upgrade rule and the separate manylinux release
  toolchain obligation.

Verification:

- the offline probe completed 158 compile/link steps with GCC 14.3.0, CMake
  3.31.8, Ninja 1.12.1, C++20, PIC, hidden visibility, a glibc-2.28 sysroot,
  generic x86-64 tuning, and POPCNT optimization disabled;
- the deterministic native executable reported
  `rdkit=2026.09.1 boost=1_86 conformers=1`;
- all six cached archives passed size and SHA-256 checks, and configure emitted
  no download action;
- the generated target map and all 18 static archive names exactly match the
  checked-in manifests;
- `ldd` reports no shared RDKit or Boost dependency; and
- JSON parsing, shell syntax, Conda explicit-lock parity/dry-run, and
  `git diff --check` pass.

Unexpected findings:

- the pinned RDKit source does not vendor its complete configuration closure:
  Catch2, better-enums, Eigen, and RingDecomposerLib also require explicit
  source identities;
- upstream RingDecomposerLib acquisition used only MD5, so Hotpot now pins its
  exact RDKit tag/commit and SHA-256 before injection;
- the true ETKDG closure contains 18 RDKit/RingDecomposerLib static archives,
  including the coarse `ForceField` and `ForceFieldHelpers` targets; these are
  compile-time ETKDG dependencies and do not enable RDKit UFF/MMFF as a Hotpot
  business optimizer;
- the first Boost download resumed across mirrors into a CRC-invalid archive;
  it was isolated and replaced by a complete single-source download verified
  against the locked official metadata hash; and
- an early shared-library check matched the word `rdkit` in a toolchain path;
  the check was narrowed to actual `libRDKit`/`libboost` names and the complete
  probe was rerun successfully.

Adjustment:

- the measured target graph replaces the prospective partial library list;
- the local probe prefix is disposable and is not a path or ABI contract; and
- release wheels remain gated on separately pinned manylinux image digests,
  architecture-specific closure reruns, wheel auditing, and import-coexistence
  tests.

Git checkpoint: `ffb2924 build(rdwrappers): lock native RDKit source closure`.

### Step 11 - Reuse common C++ molecule ownership in Open Babel

Status: complete

Planned:

- remove the remaining duplicate owning field declaration from the private
  Open Babel molecule DTO;
- retain the existing 11-argument native construction surface and all
  Open Babel policy/error behavior; and
- keep the common view zero-copy and live if legacy public vectors are
  modified before an operation.

Performed:

- made private `obwrappers::MoleculeData` derive from the toolkit-neutral
  `native::OwnedMoleculeData`;
- added an explicit compatibility constructor with the exact former argument
  order and types, moving each vector once into common ownership;
- captured dimensions before the moves to avoid dependence on C++17 argument
  or move evaluation order; and
- forwarded the common owning view while recomputing live dimensions and the
  Open Babel field mask from the compatibility object's current vectors.

Verification:

- compile-time fences prove common-base reuse, non-aggregate construction,
  the complete former 11-argument constructibility, and unchanged method
  return types;
- a C++ policy test mutates the legacy vectors, then proves updated counts,
  zero-copy pointer identity, generic validation, and unchanged OBMol
  construction;
- a clean forced rebuild of all native extensions succeeds; and
- the Open Babel/common-view/force-field source-fence selection reports
  57 passed and 1 existing managed-dataset skip, with `git diff --check`
  passing.

Unexpected findings:

- inheriting stored `atom_count`/`bond_count` fields while retaining the
  former methods of the same names is legal but deliberately exercised by the
  source fence; no overload ambiguity or pybind construction change occurred.

Adjustment:

- this is an internal ownership refactor only. Open Babel required fields,
  compatibility validation, dative rejection, error order, and engine call
  sites are unchanged.

Git checkpoint: `26f2d41 refactor(obwrappers): reuse common owned molecule data`.

### Step 12 - Implement the pure ordered Level-1 payload core

Status: complete

Planned:

- implement exact immutable Level-1 storage independently of molecule
  lifecycle hooks and toolkit conversion;
- implement one pure stable-row remapper with the ordering/filtering semantics
  fixed in plan revision 2; and
- keep Level-2 chemical interpretation and RDKit conversion out of this
  checkpoint.

Performed:

- added the immutable schema-v1 empty payload singleton and exact molecule
  get/set storage without copying or chemical interpretation;
- added `_remap_stereo_payload()` over an explicit old-row to new-row mapping
  and an optional target-bond set;
- preserved record, carrier, atom-fact, and bond-fact enumeration order;
- dropped a complete record when its anchor or any atom carrier is not mapped,
  while retaining non-atom carriers unchanged;
- checked target bond existence with an undirected endpoint key while
  preserving the stored ordered endpoint tuple; and
- retained group scope/kind/id, descriptors, source namespace/code, isotope,
  radical, display direction, and source/current hydrogen objects exactly.

Verification:

- focused API/storage/remapping suite: 11 passed;
- tests cover record and carrier order, whole-record removal, bond-anchor and
  bond-fact filtering, directed endpoint preservation, enhanced-group lineage,
  opaque future source values, and distinct source/current hydrogen states;
- Python compilation and `git diff --check` pass.

Unexpected findings: none.

Adjustment:

- lifecycle integration intentionally waits for the independent molecule
  hidden-bond/array-ownership repair, avoiding compensation for structural
  defects inside the stereo remapper.

Git checkpoint: `19c7555 feat(stereo): implement ordered Level-1 payload remapping`.

### Step 13 - Repair molecule structural ownership prerequisites

Status: complete

Planned:

- repair existing hidden-bond loss in copy and component assembly;
- make extracted component atom arrays independently owned; and
- remove all active and hidden incident bonds when deleting atoms before
  attaching Level-1 payload lifecycle behavior.

Performed:

- made `Molecule.copy()` clone active, hidden-metal, and hidden-covalent bonds
  in stable per-container order through an explicit source-to-clone atom map;
- made `add_component()` merge all three bond classes from its already
  independent component copy;
- made `components` copy each NumPy atom attribute row and clone each hidden
  bond whose two endpoints both belong to the selected active component;
- deliberately left cross-component hidden bonds out of both child molecules,
  because neither child owns both endpoints, while leaving the source molecule
  unchanged; and
- made single and batch atom removal clear both hidden-bond lists of every
  incident bond and clear removed-atom adjacency caches after successful
  structural removal.

Verification:

- six new lifecycle tests cover independent atom buffers, stable active/hidden
  bond order, endpoint rebinding, component merge, internal/cross-component
  hidden-bond handling, source immutability, and single/batch atom deletion;
- independent focused structural regression: 98 passed;
- worker native/force-field structural selection: 71 passed, 1 existing
  managed-dataset skip, and 1 existing geometry-quality warning;
- worker complex-build/ligand-ring/force-field integration selection:
  82 passed with 2 existing complex-build warnings;
- Python compilation and `git diff --check` pass.

Unexpected findings:

- `Atom.bonds` is rebuilt from active molecule bonds only; a hidden bond's own
  internal `AtomSeq._bonds` field does not register it in either endpoint.
  Removal therefore retains the existing active-bond error sequence and then
  filters hidden containers explicitly;
- the legacy broad `test_core::test_molecule` fixture constructs a 223-atom
  structure and enters `build3d` for several minutes without progress output.
  It was interrupted in the worker diagnostic run and is not counted as a
  pass; focused core tests plus the larger structural/integration selections
  above completed normally. The final suite will report this slow case
  separately if it remains non-terminating.

Adjustment:

- no metadata, property, crystal, or conformer-copy behavior was broadened;
- public exception ordering and `_update_graph()` timing remain unchanged;
- Level-1 payload hooks are still a following isolated checkpoint.

Git checkpoint: `9c0433d fix(molecule): preserve hidden structural lifecycle`.

### Step 14 - Borrow the common ligand view in force-field sessions

Status: complete

Planned:

- retain the public flat aggregate and Python/C++ construction layout of
  `ComplexSessionInput`;
- remove the internal full ligand DTO copy at session construction; and
- preserve generic-versus-Open-Babel policy boundaries, validation timing,
  and legacy first-error order.

Performed:

- replaced the copying `ligand_molecule_data()` helper with the `noexcept`
  zero-copy `ligand_molecule_view()` accessor;
- borrowed all flat atom/bond arrays and optional unit-cell memory directly
  from the owning session input;
- kept `ComplexSessionInput::validate()` consumer-neutral; and
- made the Open Babel session boundary run the compatibility adapter before
  complex-field validation and OBMol construction, retaining the adapter's
  legacy per-bond ordering while the adapter still finishes with generic
  structural validation.

Verification:

- compile-time fences prove `ComplexSessionInput` remains an aggregate and the
  new accessor has the fixed return type and `noexcept` contract;
- source tests prove pointer identity for every array, live element borrowing,
  generic dative acceptance, Open Babel dative rejection, intended
  coordination behavior, and mixed-invalid first-error parity;
- complete native extension rebuild succeeds with only existing Open Babel
  deprecation warnings;
- independent recursive force-field suite, including all nine native C++
  source fences and 87 Python tests: 96 passed in 156.86 s;
- `git diff --check` passes.

Unexpected findings:

- calling generic validation at the session boundary before the Open Babel
  adapter made a later invalid bond index outrank an earlier dative bond. The
  adapter must own the entry ordering because it preserves the historic
  index/self/order/dative scan per bond. A dedicated mixed-invalid assertion
  now fixes this behavior.

Adjustment:

- the input remains one flat owning aggregate; no physical nesting or source
  API migration was introduced;
- intended coordination bonds remain separate session data and retain their
  existing supported dative workflow;
- force-field stage order and optimization behavior are unchanged.

Git checkpoint: `d167b00 refactor(forcefields): borrow common ligand molecule view`.

### Step 15 - Make Level-1 payloads atomic in working-copy commits

Status: complete

Planned:

- carry the immutable Level-1 payload through the existing force-field
  working-copy transaction;
- transfer the exact working reference on success and restore the exact
  original reference on any failed commit; and
- leave public transaction APIs, structural validation, and user properties
  unchanged.

Performed:

- extended the private prepared-commit DTO with the working molecule's exact
  payload reference;
- extended the private rollback snapshot with the original molecule's exact
  payload reference;
- assigned the working payload inside the existing commit try/rollback region;
  and
- restored the original payload together with topology, atom arrays, graph,
  atom pairs, conformers, and metadata on failure.

Verification:

- a successful-commit test proves exact payload-reference transfer and
  unchanged user-property ownership;
- an injected failure after payload assignment proves the transient new value
  was reached and then the exact old payload, atom/bond objects, atom-array
  references, graph object, coordinates, and properties were restored;
- focused working-copy/complex-hydrogen/force-field regression: 104 passed
  with 1 existing geometry-quality warning;
- Python compilation and `git diff --check` pass.

Unexpected findings: none.

Adjustment:

- payloads are immutable and are intentionally neither copied nor chemically
  validated by the transaction layer;
- a decoy `properties["stereo_payload"]` test fixes the contract that Level-1
  state is structural private storage, not mutable user metadata.

Git checkpoint: `1dacca3 feat(stereo): make working-copy payload transactional`.

### Step 16 - Pack Level-1 stereo into field-selective native buffers

Status: complete

Planned:

- add independently selectable stereo-record, atom-transport, and
  bond-transport fields to the common Python/native communication DTO;
- preserve optional-value presence, opaque UTF-8 data, source/current
  hydrogen separation, ordering, and directed bond endpoints; and
- prevent the Open Babel consumer from acquiring fields it does not use.

Performed:

- added fixed-width frozen transport DTOs whose NumPy dtypes match the frozen
  C ABI scalar widths;
- flattened stereo anchors/carriers, enhanced-group lineage, atom facts,
  dual hydrogen states, and bond display facts without sorting or chemical
  interpretation;
- encoded each string column as an owned contiguous UTF-8 byte blob plus
  contiguous offset, length, and presence arrays;
- retained a one-byte blob sentinel so absent, present-empty, and non-empty
  values remain distinct and later C string views never depend on a Python or
  RDKit object lifetime; and
- replaced Open Babel's all-enum field selection with an explicit ten-field
  basic set, leaving all three stereo transports absent for that consumer.

Verification:

- tests cover independent selection, present-empty versus absent, non-ASCII
  UTF-8, exact dtypes/shapes/contiguity, stable record/carrier/fact order,
  enhanced groups, ordered endpoints, explicit zero versus missing isotope/
  radical values, and dual hydrogen presence masks;
- independent common/stereo/Open-Babel regression: 56 passed and 1 existing
  managed-dataset skip;
- a dynamic old-versus-new packer comparison over three representative
  molecules found identical dtype, shape, and raw bytes for every legacy
  Open Babel field (3/3 cases);
- Python compilation and `git diff --check` pass.

Unexpected findings:

- the first external parity harness loaded the historical dataclass module
  without registering it in `sys.modules`, causing a harness-only dataclass
  resolution error. Registering the dynamically loaded module corrected the
  harness; no product-code change was required.

Adjustment:

- the DTO retains its Level-1 schema version. The current C ABI v1 consumer
  supports the current schema-v1 contract; a later schema migration must add
  an explicit compatibility decision rather than silently reinterpret these
  tables;
- existing basic buffer field names, types, and packing algorithms are
  unchanged.

Git checkpoint: `9eb8684 feat(native): pack Level-1 stereo transport fields`.

### Step 17 - Integrate Level-1 payloads with molecule structure lifecycle

Status: complete, with explicit-hydrogen state isolated to a following step

Planned:

- connect the single pure remapper to molecule initialization, copy,
  extraction/assembly, removal, and hidden-bond lifecycle operations;
- retain group lineage, source/current facts, ordering, and targeted-deletion
  semantics without chemical interpretation; and
- avoid duplicating the remapping algorithm in core methods.

Performed:

- initialized every molecule with the exact immutable empty payload;
- wired copy, connected components, and `AtomSeq.to_mol()` through explicit
  stable-row maps and target structural-bond sets;
- made component assembly offset incoming rows, concatenate target records and
  facts first, and preserve equal group IDs under their distinct lineage
  scopes without sorting or renumbering;
- remapped later rows and removed whole dependent records/facts after atom
  deletion;
- removed only the selected endpoint key after active bond deletion or
  permanent hidden-bond clearing, while preserving unrelated historical
  dangling payloads exactly; and
- kept hide and recover operations byte-for-byte payload no-ops.

Verification:

- new lifecycle coverage fixes initialization, exact copy, active/internal-
  hidden component filtering, `AtomSeq` extraction, ordered component merge,
  atom remapping, targeted active/hidden bond deletion, dangling-payload
  retention, lineage scopes, and source/current hydrogen object retention;
- worker combined lifecycle regression: 69 passed with 1 existing
  geometry-quality warning; directed core selection: 2 passed and 1 existing
  managed-data skip;
- independent Level-1/structure/hidden-bond/bond-metadata selection:
  47 passed;
- Python compilation and `git diff --check` pass.

Unexpected findings:

- the legacy RDKit import collapses toolkit implicit and bracket-explicit
  hydrogens into one `Atom.implicit_hydrogens` value, while Hotpot hydrogen
  mutation exposes only net explicit-node additions/removals. Core alone
  therefore cannot prove a reversible per-field update for bracket-explicit
  and `no_implicit` state;
- `add_component()` currently operates in the closed current-release schema-v1
  domain. It retains the target schema when concatenating. Cross-schema
  assembly requires an explicit future migration rule and must not guess by
  choosing a maximum or silently upgrading.

Adjustment:

- explicit-hydrogen current-state transformation is isolated until the RDKit
  Level-1 import mapping provides separate source fields. This checkpoint
  still remaps/removes explicit-H rows and their own facts without modifying
  the immutable source ledger;
- the historical monolithic `test_core::test_molecule` again entered a
  long-running 223-atom `build3d` path and was not counted; focused and broader
  related regressions completed normally.

Git checkpoint: `570d756 feat(stereo): integrate Level-1 molecule lifecycle`.

### Step 18 - Freeze Level-2 descriptor and permutation algebra

Status: complete for Gate 1, with square-pyramidal semantics explicitly open

Planned:

- define a Hotpot-owned, toolkit-independent descriptor grammar;
- freeze reference-relative carrier/permutation transformations before
  topology, geometry, CIP, or toolkit adapter work; and
- back the algebra with a static reviewed golden corpus rather than a runtime
  toolkit oracle.

Performed:

- defined the private versioned grammar
  `hotpot.stereo/2/<TH|DB|AX|SP|TBP|SPY|OH>/<value>`;
- represented tetrahedral and axial identity as even/odd reference parity,
  double-bond identity as together/opposite reference relation, and supported
  coordination families by finite permutation-orbit indices;
- kept CIP labels explicitly outside stored semantic identity;
- implemented strict descriptor parse/render, anchor/carrier identity checks,
  permutation parity, exact Lehmer rank/unrank, bond-endpoint reference rules,
  and descriptor transforms under carrier reordering;
- derived SP, TBP, and OH equivalence orbits from their orientation-preserving
  ideal-geometry rotation groups, yielding 3, 20, and 30 configurations; and
- fixed square-pyramidal parsing/transformation as an explicit `UNSUPPORTED`
  boundary rather than inventing a numbering table without a separately
  audited convention.

Verification:

- the static golden corpus fixes identity/even/odd permutations, non-atom
  carriers, bond-anchor reversal, one/both endpoint reference changes, every
  SP/TBP/OH index, and the complete swap-01 mapping for each finite family;
- focused Gate-1 suite: 57 passed;
- independent complete stereo selection: 82 passed;
- production algebra and golden tests import neither RDKit nor Open Babel;
- `git diff --check` passes.

Unexpected findings:

- the approved plan names square-pyramidal support but does not define a
  reviewable reference geometry, symmetry group, or stable configuration
  numbering. Gate 1 therefore cannot honestly freeze SPY meaning yet. This is
  an explicit remaining Level-2 design obligation, not a silent fallback;
- one diagnostic invocation initially named a nonexistent test file and
  collected no tests. It was corrected to the actual stereo test set; no code
  failure was involved.

Adjustment:

- public `validate_stereochemistry`, `assign_stereochemistry`, and
  `align_stereochemistry` remain unimplemented in this algebra-only gate;
- later gates may consume these helpers but cannot redefine the meaning of
  already supported TH/DB/AX/SP/TBP/OH descriptors;
- SPY must be resolved before the final complete-spatial-semantics claim.

Git checkpoint: `7d2debf feat(stereo): freeze Level-2 permutation algebra`.

### Step 19 - Implement the pinned standalone RDKit core and C ABI

Status: complete for the private core/C ABI checkpoint; the Python binding and
installed SDK remain following checkpoints

Planned:

- construct the RDKit graph directly from the frozen field-selective C view;
- implement one private ETKDGv3 sampler behind the already frozen C ABI;
- preserve stable atom rows, supported Level-1 source facts, deterministic
  fixed-seed behavior, and explicit unsupported outcomes; and
- prove that the shared library boundary exports no RDKit, Boost, STL, or C++
  exception surface.

Performed:

- implemented strict C-ABI option, molecule, stereo-transport, atom/bond
  transport, and coordination-bound validation;
- constructed single, double, triple, aromatic, zero-order, and directed
  dative RDKit bonds without a Python RDKit translation;
- applied current isotope, radical, bracket-explicit-H, no-implicit-H,
  implicit-H verification, raw stereo records, display directions, and
  enhanced stereo groups while retaining source ledgers as transport-only
  provenance;
- fixed the raw stereo namespace to the exact pinned
  `rdkit-2026.09.1`, including the native fixtures;
- defined raw tetrahedral carrier parity relative to RDKit bond-storage order,
  with a true implicit-H sentinel last, and invert `CW`/`CCW` only for an odd
  transported-to-native permutation;
- reject duplicate stereo anchors, inconsistent groups, unsupported raw
  permutation-property tags, invalid carriers, unknown bonds, and
  unrepresentable hydrogen state as explicit unsupported generation results;
- corrected enhanced absolute stereo groups to accept RDKit's valid read ID
  zero rather than imposing the positive-ID rule used by numbered AND/OR
  syntax;
- used selective sanitization that does not recompute transported radicals,
  hydrogen state, or raw chiral tags;
- implemented one-thread ETKDGv3 multi-conformer generation, fixed-seed
  determinism, pruning, random-coordinate mode, optional custom bounds, stable
  coordinate/record ordering, and partial/failure diagnostics; and
- implemented opaque result ownership, runtime identity accessors, hidden
  visibility, and the `hotpot_rd_*` version-script export set. No RDKit force
  field is called by this component.

Verification:

- all four C++20 production units compile with the pinned GCC 14.3 toolchain,
  `-Wall -Wextra -Werror`, hidden visibility, and the exact source-built RDKit
  2026.09.1 headers;
- the direct graph-transfer executable passes aromatic, zero-order, dative,
  raw atom/bond stereo, odd tetrahedral carrier parity, isotope, radical,
  hydrogen, display-direction, enhanced-group, duplicate-anchor, and explicit
  unsupported cases;
- the namespace/ABS-zero corrections were independently recompiled and the
  direct executable reran successfully against the pinned static closure;
- standalone C11, C++17 ABI-0, and C++17 ABI-1 consumers compile, link, and
  run; the C++ consumer also reuses one sampler concurrently in four threads;
- repeated fixed-seed coordinate bytes are identical, optional bounds run,
  and the input arrays remain unchanged;
- the maintained compile-only API selection reports 3 passed;
- `ldd -r` reports no unresolved symbol and no Python/Conda RDKit dependency;
  and the linked export table contains only the approved `hotpot_rd_*`
  surface; and
- `git diff --check` passes.

Unexpected findings:

- the first native implementation accepted the generic namespace `rdkit`
  while the Python adapter correctly used the exact pinned namespace. The
  native boundary was narrowed before publication;
- RDKit absolute enhanced groups legitimately use read ID zero, so a blanket
  positive group-ID validation would have rejected an exact Python-to-native
  transfer;
- raw atom stereo carriers were initially copied but not interpreted by the
  native builder. The core now proves the tetrahedral carrier permutation and
  rejects every unproved carrier/tag combination rather than silently applying
  a source-relative tag to a different neighbor order; and
- bracket-explicit virtual hydrogen and reversed display-direction semantics
  require one further audited adapter/core checkpoint. They are not claimed
  as closed by this step.

Adjustment:

- unsupported molecular representation returns a normal owned result with
  `HOTPOT_RD_GENERATION_UNSUPPORTED`; malformed ABI/data remains a C status;
- the C ABI header is still implementation-local until the binding, common
  Level-1 view, SDK install, and the remaining direction/H conventions close;
- no public Python fallback, force-field seam, or workflow default was added.

Git checkpoint: `1c60a70 feat(rdwrapper): implement pinned native ETKDG core`.

### Step 20 - Implement the exact Python RDKit Level-1 adapter

Status: complete

Planned:

- capture raw RDKit stereochemistry and atom/bond transport facts before any
  Kekulize, sanitization, or assignment operation;
- restore those facts without calling stereochemistry assignment or silently
  changing neighbor-relative raw tags; and
- keep Level-1 transport separate from Level-2 validity and descriptor work.

Performed:

- pinned every RDKit raw record and atom fact to the exact
  `rdkit-2026.09.1` namespace and allocated a distinct opaque enhanced-group
  scope for each independent import;
- captured atom chiral tags, ordered atom carriers, bond stereo/control atoms,
  display directions, enhanced groups, isotope, radical, true implicit H,
  bracket-explicit virtual H, and `noImplicit` from an untouched source clone;
- used a separate property-cache clone only to read derived implicit H, so the
  raw source object and raw tags are not rewritten;
- preserved stable RDKit-index/Hotpot-row mappings, aromatic/dative behavior,
  source/current hydrogen ledgers, and the existing fallback for molecules
  without a Level-1 payload;
- reconstructed tetrahedral tags relative to target RDKit bond-storage order,
  inverting `CW`/`CCW` for odd carrier permutations;
- froze true implicit H as a final `IMPLICIT_H` carrier while representing a
  bracket-explicit virtual H by exactly three atom carriers plus its separate
  hydrogen ledger; both require exactly four total graph/virtual neighbors;
- swapped E/Z carrier sides when a bond anchor is reversed, inverted
  `ENDUPRIGHT`/`ENDDOWNRIGHT`, retained endpoint-independent `NONE` and
  `EITHERDOUBLE`, and explicitly rejected reversed wedge/dash/unknown and
  unfrozen atropisomer direction semantics;
- rejected unknown namespaces/codes, duplicate atom facts, duplicate stereo
  anchors, duplicate bond facts, conflicting enhanced groups, missing or
  duplicate carriers, non-neighbor carriers, unsupported non-atom carriers,
  unrepresentable implicit-H state, and non-tetrahedral permutation tags whose
  raw reordering rule is not yet frozen; and
- did not call `AssignStereochemistry`, CIP assignment, coordinate validation,
  Hotpot geometry gates, or perturbation.

Verification:

- the worker's adapter/conversion/lifecycle selection reports 85 passed;
- an independent Level-1 adapter, bond-metadata, storage, remapping, lifecycle,
  native-packing, and working-copy selection reports 78 passed;
- fixtures cover implicit-H and `[C@@H]` tetrahedra, odd bond/carrier order,
  E/Z and display directions, enhanced ABS/AND/OR groups, isotope/radical/H,
  distinct lineage scopes, unsupported values, source immutability, and the
  absence of stereochemistry assignment;
- Python compilation and `git diff --check` pass.

Unexpected findings:

- `GetNumImplicitHs()` fails on some unsanitized RDKit inputs until the
  property cache is updated. Updating a detached clone provided the value
  without violating raw-source capture;
- RDKit exposes no public `SetNumImplicitHs()`. The adapter therefore derives
  it from the reconstructed graph/current explicit state and rejects a ledger
  value that the pinned toolkit cannot reproduce;
- the initial conservative audit rejected bracket-explicit virtual H. Pinned
  `Atom.h`, `Chirality.cpp`, and `AddHs.cpp` prove that degree-three tetrahedral
  atoms use their one virtual H as a fixed fourth reference, so common
  `[C@@H]` ligands are now supported without mislabelling that H as implicit;
  and
- RDKit has no proved raw reversed-anchor transform for atropisomer
  `CW`/`CCW`; that narrow case remains explicit unsupported pending Level 2.

Adjustment:

- Level 1 preserves raw toolkit state but makes no claim that a transported
  record is chemically valid;
- the exact versioned source namespace is part of the adapter contract, not a
  multi-version compatibility profile;
- public Level-2 validation, assignment, and alignment remain separate gates.

Git checkpoint: `56c0b42 feat(stereo): preserve RDKit Level-1 transport`.

### Step 21 - Carry Level-1 views through the common C++ facade

Status: complete

Planned:

- extend the common C++17 non-owning/owning DTOs to mirror the three already
  frozen Python Level-1 SoA tables;
- pass optional stereo, atom-transport, and bond-transport fields through the
  standalone C++ facade without copying molecule bulk arrays or strings; and
- correct directed dative endpoints before admitting the common payload to
  RDKit.

Performed:

- added C++17 views and owning data for UTF-8 tables, ordered stereo records
  and carriers, dual-hydrogen atom facts, and ordered bond facts;
- retained field-selective absence: an unrequested transport owns/carries no
  data, while a requested empty table remains a present schema-v1 field;
- validated schema, column sizes, slice bounds, string presence, anchor and
  carrier structure, enhanced-group identity, atom ranges, and transport
  dimensions in the common view;
- kept every NumPy/basic array, flattened anchor table, and UTF-8 byte blob as
  a zero-copy borrowed view;
- added a synchronous facade owner that assembles only the small stereo,
  atom-fact, and bond-fact AoS metadata required by the frozen C ABI; it lives
  through the complete C call and never copies molecule bulk arrays, anchor
  arrays, or string bytes;
- validated C++ constraint pair/bounds row-count equality before producing the
  pointer/count C view, closing information that the C ABI itself cannot infer;
  and
- changed common Python bond-index packing so reverse dative bonds are always
  encoded donor-to-acceptor. Ordinary bonds and forward dative bonds retain
  their existing endpoint order.

Verification:

- the worker's common/native/Open-Babel selection reports 48 passed and 1
  existing managed-data skip;
- an independent Python/C++ common-buffer, Level-1 packing, C++ facade, and
  Open-Babel packing selection reports 20 passed;
- the C++ tests prove zero-copy pointer identity, absent versus present-empty
  strings/tables, complete Level-1 C-ABI values, synchronous owner lifetime,
  mismatched constraint rejection, and unchanged ABI-0/ABI-1 facade behavior;
- dative tests cover ordinary, forward, and reverse endpoint cases; and
- `git diff --check` passes.

Unexpected findings:

- the frozen Python/common transport is SoA for contiguous NumPy columns,
  while the already frozen C ABI uses AoS record structs. A small per-call
  metadata assembly is therefore necessary at the header-only facade unless
  one of the public boundaries is changed. The selected bridge preserves both
  contracts and does not duplicate molecular numeric/string storage;
- the C++ constraints facade previously carried pair counts but did not prove
  that the corresponding bounds views had the same row count. The check must
  occur before flattening because that length is intentionally absent from the
  C ABI.

Adjustment:

- `MoleculeBufferView` remains a synchronous non-owning DTO and
  `OwnedMoleculeData` remains the explicit long-lived owner;
- Open Babel continues to request only its original ten basic fields and does
  not pay for Level-1 packing;
- no public dynamic field profile or second communication buffer was added.

Git checkpoint: `32a6064 feat(native): carry Level-1 views into rdWrapper`.

### Step 22 - Close virtual-hydrogen and directed-display semantics in native RDKit

Status: complete

Planned:

- apply the pinned-source evidence from the Python adapter audit to the native
  builder; and
- make every endpoint-order transformation either exact or explicit
  unsupported.

Performed:

- admitted a tetrahedral center with three graph carriers and exactly one
  bracket-explicit virtual H, without encoding that H as an implicit carrier;
- retained a true implicit H as the final explicit `IMPLICIT_H` transport
  token and required graph plus virtual carrier count to equal four;
- applied the same odd-permutation `CW`/`CCW` transform to both conventions;
- transformed reversed `ENDUPRIGHT`/`ENDDOWNRIGHT`, retained reversed `NONE`
  and endpoint-independent `EITHERDOUBLE`, and rejected reversed wedge, dash,
  unknown, or otherwise unproved display directions;
- rejected reversed raw atropisomer `CW`/`CCW` until its convention is frozen;
  and
- required native atom-transport provenance to match the exact pinned RDKit
  namespace, matching the Python adapter rather than silently interpreting a
  future toolkit representation.

Verification:

- both changed C++20 units compile with GCC 14.3, `-Wall -Wextra -Werror`;
- the direct graph-transfer executable links against the exact static source
  closure and passes bracket-H, odd parity, implicit-H distinction,
  EITHERDOUBLE, slash inversion, reversed-wedge rejection, reversed-atrop
  rejection, and unknown atom-namespace rejection;
- `git diff --check` passes.

Unexpected findings:

- RDKit's crossed-double display code is endpoint-independent and can remain
  unchanged under endpoint reversal; it must not be grouped with endpoint-
  anchored wedge/dash codes;
- raw atropisomer code reversal is not specified by the audited pinned APIs,
  so treating it like ordinary E/Z would create a false semantic guarantee.

Adjustment:

- no new carrier kind was introduced: bracket-explicit and implicit H remain
  distinguishable through the hydrogen ledger and the existing sentinel rule;
- unsupported raw direction cases remain Level-1 data in Hotpot but cannot be
  consumed by this RDKit adapter until an exact transform exists.

Git checkpoint: `36069ed fix(rdwrapper): preserve virtual-H and bond direction stereo`.

### Step 23 - Publish the Python binding and standalone native SDK build

Status: complete

Planned:

- bind the frozen C ABI to the public Python contracts without constructing a
  Python RDKit molecule;
- produce an independently installable C/C++17 SDK and a separately staged
  Python extension from the same private RDKit core; and
- prove the exact offline source closure, ownership, GIL, exported-symbol, and
  runtime-dependency boundaries before packaging integration.

Performed:

- added a pybind11 boundary that accepts the field-selective common molecule
  buffers, assembles only the small C-ABI metadata records, retains every
  borrowed NumPy owner through the native call, and releases the GIL only
  while `hotpot_rd_generate()` executes;
- validated exact native-endian dtypes, ranks, shapes, C contiguity, alignment,
  UTF-8 presence/offset tables, molecule dimensions, and paired constraint
  rows before any typed native dereference;
- copied the owned native result into the frozen Python `ConformerBatch`
  contract after reacquiring the GIL and exposed the compiled Hotpot, RDKit,
  C-ABI, and source-commit identities;
- added the locked offline CMake driver, SDK/Python build components, CMake
  package config, public headers, exact third-party notices/licenses, hidden
  visibility, version scripts, and post-link RPATH/dependency audit;
- installed a C++17 SDK whose shared library contains the private C++20 RDKit
  core but exposes only the stable C ABI and header-only facade;
- kept the Python extension independent of the SDK shared library by absorbing
  the same private static core and exporting only `PyInit__rd_native`;
- made SDK-only builds avoid resolving or requiring Python; and
- removed optional `BOND_ORDERS` from the RDKit field request. Hotpot/Open
  Babel legitimately retains aromatic bond kinds with Kekule 1/2 orders, so
  the nonredundant `BOND_KINDS`/aromatic fields are the authoritative graph
  input for this adapter.

Verification:

- a fresh locked-source build completed all 164 Ninja steps from the six
  SHA-verified archives with FetchContent disconnected and no configure-time
  download;
- the compiled Python/API/installed-SDK selection reports 22 passed, including
  deterministic nonmutating generation, the aromatic-Kekule public path,
  malformed and unaligned array rejection, SDK-only Python independence, and
  C++17 consumers built with both `_GLIBCXX_USE_CXX11_ABI=0` and `1`;
- the broader rdWrappers, common-buffer, RDKit Level-1 adapter, storage,
  remapping, lifecycle, native-packing, and working-copy selection reports
  86 passed;
- the extension exports only its ELF version node and `PyInit__rd_native`; the
  SDK exports only its version node and the 15 approved `hotpot_rd_*` symbols;
- both artifacts have no RPATH/RUNPATH and no dynamic RDKit or Boost
  dependency; SDK `ldd -r` is clean, while the extension's Python symbols are
  correctly resolved by the importing interpreter and its import/generation
  tests pass; and
- shell syntax and `git diff --check` pass.

Unexpected findings:

- a C-contiguous NumPy view can still be unaligned. Typed C++ dereference would
  then be undefined behavior, so alignment is now an explicit binding
  precondition with a real offset-buffer regression test;
- the first SDK-only driver revision resolved `python3` before reading the
  Python-build option. That contradicted the standalone C++ component contract
  and was moved inside the Python-only branch;
- the canonical 187-ligand smoke probe initially returned `UNSUPPORTED` for
  aromatic molecules because the optional numeric bond-order column supplied
  Kekule 1/2 values beside an aromatic bond kind. The C ABI does not require
  that redundant column; omitting it preserves the actual graph semantics and
  makes the sampled corpus cases succeed without weakening validation.

Adjustment:

- the current GCC 14.3 Conda proof artifact intentionally contains no
  temporary RPATH and therefore needs its disposable toolchain `libstdc++` via
  an explicit development `LD_LIBRARY_PATH`; it is not a release artifact;
- the root Hotpot wheel build and the pinned manylinux/Python 3.10-3.14 release
  matrix remain separate later checkpoints and are not claimed here;
- no system/Conda/Python-wheel RDKit SDK lookup or public force-field routing
  change was introduced.

Git checkpoint: `42d7c4c feat(rdwrapper): add native Python binding and SDK build`.

### Step 24 - Freeze and execute the detached R0 benchmark

Status: in progress

Planned:

- freeze the exact 187-ligand source and the `1 / 8 / 32` ETKDG matrix before
  measuring the native component;
- separate primary generation time from the deliberately repeated
  determinism and detached observation work; and
- make topology, atom-row, Level-1, nonmutation, finite-coordinate, and
  same-seed checks machine-readable per case.

Performed:

- added `run_r0_benchmark.py` with the canonical Git-blob byte digest,
  one-based physical-line case identity, fixed `20260921 + case_index` seed,
  one outer worker, and one native RDKit thread;
- fixed pruning and random-coordinate policy explicitly and recorded the
  complete runtime identity and execution policy in JSONL/JSON artifacts;
- kept geometry gates, Hotpot perturbation, RDKit/Open-Babel force fields,
  optimization, and conformer selection outside R0;
- reran every successful/partial trial with the same seed and performed all
  structural observations on detached coordinate rows after the primary
  timer; and
- executed the first clean 561-trial matrix at Git checkpoint `b61e98a`.

First-matrix evidence:

- 532 trials returned `SUCCESS`, two returned `PARTIAL`, and 27 returned
  `UNSUPPORTED`;
- all 7,280 returned primary conformers were finite, row-aligned,
  deterministic, topology/atom-order/Level-1 preserving, and nonmutating;
- primary generation took 75.4888 seconds: 96.4381 conformers/s and 7.4316
  trials/s overall;
- counts `1`, `8`, and `32` returned 178, 1,420, and 5,682 conformers at
  61.4018, 93.9200, and 98.8679 conformers/s respectively; and
- case 10 returned 4/8 and 18/32 conformers with RDKit's stable partial-result
  diagnostic. This is valid ETKDG duplicate/pruning behavior, not a failed
  invariant.

Unexpected findings:

- the 27 unsupported trials were exactly nine aromatic `[nH]` ligands at all
  three requested counts. Open Babel supplied the correct logical Hotpot
  implicit-H count, but those payload-free atoms had no adapter transport
  record, so the native graph received an ordinary aromatic nitrogen;
- merely reproducing the established Python `to_rdmol()` explicit-H fallback
  allowed initial graph construction but was insufficient. The deliberately
  restricted native sanitize mask omitted RDKit `SANITIZE_ADJUSTHS`; its first
  Kekule pass removed the bracket H, and ETKDG's later internal Kekule pass
  then rejected the ring;
- adding the pinned RDKit `ADJUSTHS` phase preserves `[nH]` through the later
  embedding path without disabling Kekulization or weakening graph checks.

Adjustment:

- the RDKit adapter now appends an exact `hotpot-legacy/1` hydrogen record only
  for atoms not already covered by the Level-1 ledger. The native side accepts
  only its logical implicit-H field and maps it to the already established
  Hotpot-to-RDKit explicit-H fallback;
- exact `rdkit-2026.09.1` records retain their previous lossless handling and
  unknown namespaces remain unsupported;
- focused compiled/native/Level-1 selections report 63 passed, and the nine
  formerly unsupported canonical cases all return `SUCCESS` at count 1;
- the complete clean 561-trial rerun is still required before this step can be
  closed.

Git checkpoints:

- `7f4c60d test(rdwrapper): add detached R0 benchmark runner`;
- `8e8d2f8 fix(rdwrapper): bridge legacy hydrogen transport`.

### Step 25 - Add the private conformer-provider seam

Status: complete

Planned:

- make only the existing initial ligand-conformer stage replaceable;
- retain Open Babel as the default and leave the public force-field API and
  the remainder of the complex workflow unchanged; and
- make RDKit failure explicit, without a hidden Open-Babel fallback.

Performed:

- added backend-neutral private request/result/protocol contracts and Open
  Babel/RDKit adapter implementations;
- threaded an optional private sampler through the existing worker-process
  spawn boundary;
- replaced only the initial `OBBuilder` conformer call in the ligand stage;
- retained one derived worker seed, one conformer, the stable minimum ordinal,
  and lazy RDKit-wrapper import; and
- kept every downstream force-field, assembly, geometry, optimization, and
  reporting stage on its established route.

Verification:

- the independent conformer seam and force-field/complex selection reports
  177 passed in 166.75 seconds;
- default Open Babel behavior and seed propagation are unchanged;
- the RDKit adapter consumes the native result without mutating the input and
  raises its explicit failure instead of silently changing engines.

Unexpected findings:

- the provider must cross a spawned worker boundary, so its contract must be
  data-oriented and picklable rather than closing over a live toolkit object;
- reseeding each component would have changed the existing workflow. The seam
  therefore transports the single already-derived worker seed unchanged.

Adjustment:

- this is a private replaceable seam, not a new public force-field mode;
- Open Babel remains the default provider and no component has been removed.

Git checkpoint: `097749a feat(forcefields): add private conformer provider seam`.

### Step 26 - Implement Level-2 Gate 2 graph semantics

Status: complete

Planned:

- validate and assign only manually specified tetrahedral and double-bond
  descriptors whose graph meaning is frozen by Gate 1;
- reject unsupported advanced families explicitly; and
- keep all logic toolkit-neutral and independent of coordinates or CIP.

Performed:

- added public graph validation and assignment for strict
  `hotpot.stereo/2/TH/{even,odd}` and
  `hotpot.stereo/2/DB/{together,opposite}` descriptors;
- validated anchors, carrier cardinality, atom references, graph adjacency,
  endpoint grouping, and duplicate identities before assignment;
- made double-bond carrier identities endpoint-scoped so the two endpoint
  implicit-H sentinels and a ring atom shared across both endpoint groups do
  not collide; and
- returned stable issue records without changing the source molecule.

Verification:

- the relevant stereo, Level-1, and Gate-2 selection reports 139 passed;
- fixtures cover valid/invalid tetrahedral and double-bond records, endpoint
  reversal, virtual H, shared-ring carriers, missing edges, duplicates, and
  unsupported families;
- no RDKit assignment, CIP ranking, coordinate inference, or geometry
  alignment is called.

Unexpected findings:

- raw atom identity alone is insufficient for double-bond carriers: one ring
  atom can be a valid carrier at both endpoints, and both endpoints may each
  own an `IMPLICIT_H` sentinel. Endpoint group is therefore part of the
  identity used for validation.

Adjustment:

- enumeration order remains outside the permanent contract, while the
  validation and issue order is deterministic for an identical input;
- Gate 3 explicit mapping and parity alignment remains a separate checkpoint.

Git checkpoint: `b61e98a feat(stereo): implement Level-2 Gate-2 semantics`.
