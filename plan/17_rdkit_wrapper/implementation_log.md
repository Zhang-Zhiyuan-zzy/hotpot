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

Status: complete

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
- the final clean 561-trial rerun at `531130c` reports 559 `SUCCESS`, two
  expected `PARTIAL`, and zero `UNSUPPORTED`/`FAILED` outcomes;
- count 1 returns 187/187 full successes; counts 8 and 32 each return 186 full
  successes plus case 10's stable 4/8 and 18/32 partial batches;
- all 7,649 returned conformers pass finite-coordinate, atom-row, topology,
  Level-1, input-nonmutation, and same-seed checks with zero failures;
- primary generation takes 77.8418 seconds overall at 98.2634 conformers/s;
  counts 1/8/32 run at 62.1682 / 95.6409 / 100.7870 conformers/s; and
- the clean runtime records branch `feature/rdWrapper`, commit
  `531130c71de6fc32cd23946ebc55d4290c164ff4`, Python 3.11.16, native and
  Python RDKit 2026.09.1, one native thread, one outer worker, and a clean Git
  tree.

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

### Step 27 - Implement Level-2 Gate 3 explicit-mapping alignment

Status: complete

Planned:

- align two already validated tetrahedral or double-bond assignments under an
  explicit atom mapping;
- reuse the frozen Gate-1 parity algebra and Gate-2 endpoint-scoped carrier
  identity; and
- keep coordinate assignment, CIP, toolkit perception, and advanced stereo
  families outside this gate.

Performed:

- validated mapping range, injectivity, and complete two-sided coverage of all
  stereo-relevant atoms before comparing assignments;
- transformed tetrahedral descriptors with the Gate-1 permutation parity;
- aligned double-bond endpoint pairs under anchor reversal and applied the
  equivalent endpoint-carrier XOR parity transform;
- retained endpoint scope for double-bond carriers, including two independent
  implicit-H sentinels and one ring atom shared across endpoint groups; and
- returned deterministic missing, extra, duplicate, incomplete, invalid-graph,
  and descriptor-mismatch issue records without modifying either molecule.

Verification:

- the implementation author's complete stereo and RDKit Level-1 selection
  reports 150 passed;
- an independent review selection spanning Gate 1-3, public API, remapping,
  and lifecycle reports 105 passed;
- the reviewer found no blocking mapping, parity, endpoint-scope, signature,
  mutation, or deterministic-order defect; and
- Python compilation and `git diff --check` pass.

Unexpected findings:

- a double-bond carrier cannot be keyed by global atom identity alone. The
  same ring atom may participate once at each endpoint, and each endpoint may
  independently own an implicit-H sentinel. Gate 3 must carry forward Gate
  2's endpoint-scoped identity through mapping and anchor reversal.

Adjustment:

- enumeration order remains outside the permanent search/stereo contract, but
  identical mapping and record inputs produce a fixed issue order;
- exhaustive all-permutation property tests and broader multi-record order
  stress tests remain useful nonblocking hardening, not evidence of an
  observed semantic defect;
- Gate 4 coordinate assignment remains a separate later decision.

Git checkpoint: `b086577 feat(stereo): implement Level-2 Gate-3 alignment`.

### Step 28 - Implement and measure standalone coordination-aware ETKDG

Status: complete for implementation and development evidence; formal cohort
benchmark remains blocked on the recorded external artifact

Planned:

- keep `C1` as stock complete-complex ETKDG and add `C2` only through RDKit's
  supported custom distance-bounds extension;
- require explicit directed donor-to-metal dative graph edges and Angstrom
  bounds; and
- report coordinate generation separately from post hoc bound observation.

Performed:

- required every zero-based coordination row to name an existing directed
  donor-to-metal `DATIVE` edge in the complete native graph;
- overlaid metal-donor and optional unordered donor-pair lower/upper bounds on
  RDKit's topological bounds matrix;
- explicitly triangle-smoothed the overlaid matrix and returned a stable
  failed result when the constraints are inconsistent;
- documented direction and Angstrom units in the Python and C contracts;
- added mono-dentate, bi-dentate, multiple-ligand, direction, symmetry,
  inconsistent-triangle, whole-complex, unsupported-state, and prohibited
  force-field-entry tests; and
- added a detached C1/C2 runner with fixed `1 / 8 / 32` counts, manifest and
  native-structure hashes, fixed seeds, runtime/RDKit commit identity,
  deterministic hashes, topology nonmutation, actual distance observations,
  and a permanent non-formal label for `--smoke` cohorts.

Verification:

- the author reports 35 joint Python/native/API tests and 15 coordination-only
  tests passed; the independent post-format selection reports 33 passed;
- a strict C11 C-API smoke compiles and runs against the current SDK library;
- the one-case development smoke returned full `SUCCESS` counts for both C1
  and C2 at 1, 8, and 32, with fixed-seed equality and preserved input graphs;
- C2's maximum post hoc bound violation was 0.001949 Angstrom, while exact
  zero-violation counts were 0/1, 7/8, and 29/32.

Unexpected findings:

- pinned RDKit copies a supplied custom bounds matrix into the embedding path
  but does not triangle-smooth that caller-provided matrix. The first native
  implementation checked only pairwise lower/upper consistency, so correlated
  impossible bounds could survive until embedding;
- RDKit's multi-fragment path does not use the custom bounds matrix as the
  complete-complex experiment requires. C2 must therefore retain the explicit
  donor-to-metal dative edges and embed one connected native graph;
- an ETKDG `SUCCESS` status does not mean every returned coordinate is exactly
  inside each input interval. Bounds guide distance geometry; strict
  conformance is a separate detached observation and cannot be used to rewrite
  the status.

Adjustment:

- C1/C2 remain coordinates-only research routes: no Open Babel, `hotpot.ff`,
  UFF/MMFF optimization, Hotpot geometry gate, repair, selection, or
  perturbation is called;
- the standalone result is not described as a validated metal-complex force
  field or as production replacement evidence;
- the recorded formal 182-case manifest and matching native-graph sidecars are
  still absent. The Eu `/tmp` smoke cohort was not substituted and no formal
  182-case result is claimed.

Git checkpoint: `edc15be feat(rdwrapper): add coordination-aware ETKDG bounds`.

### Step 29 - Integrate the locked native core into release wheels

Status: implementation complete; full CPython wheel matrix delegated to CI

Planned:

- build the Python extension through the same locked offline source/CMake
  route as the standalone SDK;
- freeze a real manylinux release image and toolchain independently of the
  disposable Conda probe; and
- install only the Python component into wheels while retaining the SDK as a
  separate versioned developer artifact.

Performed:

- added a setuptools CMake extension that verifies/fetches the exact six
  locked archives, disables SDK output, builds `_rd_native`, and installs only
  the `rdwrappers_python` component into the wheel staging tree;
- added the complete native/common-header/CMake/lock input closure to the
  source distribution manifest;
- selected and recorded the exact
  `manylinux_2_28_x86_64@sha256:39df...ff82` image, glibc 2.28, GCC/G++
  14.2.1 RPMs, binutils 2.41 RPM, CMake 4.4.3, and PyPI Ninja 1.13.2/CLI
  identity;
- made the build driver reject any release-profile identity mismatch while
  preserving a separate exact Conda probe profile;
- pinned the wheel workflow to that digest/profile for CPython 3.10-3.14 and
  added installed-wheel Python-RDKit coexistence, runtime identity, and native
  ETKDG smoke checks; and
- kept the public C++ SDK library/headers outside the Python wheel.

Verification:

- packaging/profile tests plus the pre-existing SDK-only Python-resolution
  checks report 8 passed;
- a locally generated source archive contains the wrapper CMake/source/header
  closure, common headers, locked fetcher/manifest, and exact probe CMake
  inputs;
- the pinned container completed the offline 162-target native build;
- the staged extension exports only `PyInit__rd_native`, has no
  RPATH/RUNPATH, and requires at most `GLIBCXX_3.4.22`;
- the staged Python component contains the extension, stub, and licenses but
  no SDK library or public SDK installation tree;
- shell syntax, lock JSON parsing, and `git diff --check` pass.

Unexpected findings:

- the verified image ships gcc-toolset-14 binutils RPM release
  `2.41-4.el8_10.1`, not the initially assumed spelling;
- PyPI `ninja==1.13.2` reports the exact CLI string
  `1.13.2.git.kitware.jobserver-pipe-1`; both real identities are now locked;
- the first profile refactor resolved the optional Python executable before
  reading `HOTPOT_RDWRAPPERS_BUILD_PYTHON`, which would have regressed the
  standalone SDK contract. Resolution was moved back inside the Python-only
  branch and the SDK-only ordering test retained.

Adjustment:

- a complete PEP-517 wheel build reached isolated build-dependency
  installation but produced no output for four minutes in the local network
  environment; the disposable container was stopped rather than presenting a
  hung build as evidence;
- the actual CPython 3.10-3.14 build/install/audit matrix remains an explicit
  release-CI gate. Native container success is not reported as completed
  wheel-matrix success.

Git checkpoint: `531130c build(rdwrapper): integrate locked manylinux wheels`.

### Step 30 - Implement Level-2 Gate 4 coordinate semantics

Status: complete

Planned:

- assign the already declared tetrahedral and double-bond records from an
  explicit coordinate array without discovering new stereoelements;
- use Hotpot-owned, reference-relative geometry rather than toolkit stereo
  assignment; and
- report scale-aware degeneracy and stored-descriptor contradictions without
  mutating the molecule, source payload, or caller coordinate array.

Performed:

- added strict finite `(atom_count, 3)` coordinate handling and record-scoped
  geometry diagnostics;
- assigned tetrahedral even/odd identity from the ordered-carrier signed
  volume, using the center as the projective proxy for an implicit hydrogen;
- assigned double-bond together/opposite identity from endpoint-scoped
  transverse carrier directions, including one implicit hydrogen at either
  endpoint;
- made both assignments invariant to translation, proper rotation, positive
  scale, anchor reversal, and the parity rules frozen by Gates 1-3;
- returned newly assigned payload values while leaving the stored Level-1
  payload untouched; and
- changed the stale Gate-2 coordinate placeholder assertion from deferred
  `UNSUPPORTED` to the now implemented `DEGENERATE_GEOMETRY` result.

Verification:

- the final complete `test_*stereo*.py` selection reports 171 passed;
- fixtures cover all tetrahedral carrier permutations, every double-bond
  endpoint-swap/anchor-order combination, explicit and implicit hydrogen,
  rigid transforms, reflection, scale, invalid coordinate arrays,
  degeneracy, contradiction ordering, and non-mutation;
- an independent scientific review reproduced the permutation and tolerance
  counterexamples described below, then confirmed all 24 permutations and
  all local near-zero cases after correction;
- Python compilation and `git diff --check` pass; and
- neither RDKit nor Open Babel is used as the runtime semantic oracle.

Unexpected findings:

- the first implementation normalized tetrahedral volume by the three edges
  from whichever carrier occupied the fourth reference slot. Although the
  signed volume transformed correctly, the degeneracy threshold therefore
  depended on carrier order: one fixed near-planar geometry could be
  unresolved in one order and assigned in another;
- exact-zero checks for center-carrier distance, double-bond axis length, and
  endpoint transverse direction admitted nonzero values far below the
  declared relative tolerance.

Adjustment:

- tetrahedral degeneracy now uses the maximum of all carrier-pair distances as
  a permutation-invariant characteristic length and evaluates normalized
  signed volume;
- center-carrier, double-bond-axis, explicit endpoint-carrier, and transverse
  magnitudes now use deterministic local relative scales before assignment;
- practical molecular coordinate ranges are the supported numerical domain;
  special policy for finite values near floating-point overflow or underflow
  remains outside this gate.

Git checkpoint: `31fe1a4 feat(stereo): implement Level-2 Gate-4 geometry`.

### Step 31 - Revise the cross-gate stereo model before Gate 5

Status: complete for the normative revision; dependent implementation has not
started

Planned:

- reread the complete implementation plan before changing it, as required by
  the owner;
- resolve representation questions exposed by the Gate 5/6 and Gate 7 design
  reviews across Level 1, the common/native boundary, public reports, and all
  later semantic gates; and
- remove conflicting or underspecified claims instead of compensating inside
  one local algorithm.

Performed:

- reread the complete 1,522-line normative plan and checked every affected
  Level-1 lifecycle, API-tree, native-boundary, Level-2 gate, test-obligation,
  and approved-decision section before editing;
- appended `StereoAnchorKind.AXIS` as stable native code 3 for a complete
  ordered allene/cumulene path while retaining real `BOND` anchors for formal
  atropisomerism;
- fixed constituent-edge deletion and native explicit-unsupported behavior as
  cross-layer requirements so an axis can neither survive a broken path nor
  fall through an `else == BOND` conversion;
- added the backward-compatible `StereoDerivedDescriptor` contract and
  `StereoAssignmentReport.derived=()` field, keeping CIP names outside stored
  descriptor/source identity;
- bounded Stage-17 CIP output to priorities fully decided by direct atomic
  number or isotope mass and explicitly deferred recursive duplicate-node,
  ring, pseudoasymmetric, descendant-stereo, and axial naming;
- froze square-pyramidal stereo as 30 proper-C4 configurations with ideal-site
  order `+z,+x,+y,-x,-y`, without changing the existing SP/TBP/OH indices; and
- specified enhanced-group inversion, exact represented-graph equivalence,
  proper-rotation coordinate fitting, TBP/SPY ambiguity, chelate partition,
  and explicit unsupported coordination boundaries.

Verification:

- independent organic/cheminformatics review identified the missing complete
  cumulene-axis identity and distinguished graph canonical rank from CIP;
- independent architecture review traced the additive enum through Python
  lifecycle, flat common buffers, C++ view validation, public C ABI, RDKit
  construction, adapters, and required tests;
- independent coordination review derived SPY's order-4 proper rotation group
  and 30 orbits and confirmed that chelation constrains equivalence/alignment,
  not the coordinate-fit score;
- the public validation/assignment/alignment function signatures remain
  unchanged, existing native enum values and ABI layouts remain unchanged,
  and the two-argument assignment-report construction remains valid.

Unexpected findings:

- a two-endpoint `BOND` cannot represent a long cumulene axis: deleting an
  internal double bond would leave a stale record, and endpoint rows alone do
  not recover an unambiguous ordered path;
- the approved plan required CIP as derived output but neither specified a
  return location nor froze the recursive IUPAC Sequence Rules. A canonical
  graph rank is not a chemically valid replacement;
- Gate 1 correctly left SPY unsupported because its ideal-site order and
  proper symmetry group had not yet been frozen.

Adjustment:

- Gate 5 now starts with the additive AXIS Level-1/common/C-ABI checkpoint
  before consuming that representation chemically;
- native RDKit materialization explicitly rejects interpreted Hotpot AXIS
  records until a separate pinned `CHI_ALLENE` mapping fixture is reviewed;
- Gate 6 delivers exact graph stereogenicity plus the strictly decidable CIP
  subset and reports all wider cases as unresolved instead of overclaiming;
- Gate 7 first freezes SPY algebra/goldens, then adds declared-record graph,
  coordinate, mapping, and chelate semantics without automatic center
  discovery.

Git checkpoint: `1079b40 docs(plan17): revise advanced stereo semantics`.

### Step 32 - Add ordered AXIS transport across Level 1 and native boundaries

Status: complete

Planned:

- add the approved ordered axis-path value without changing existing enum
  numbers, C ABI layout/version, or public function signatures;
- prove exact Level-1 lifecycle behavior before assigning any AX chemistry;
  and
- make the RDKit adapter reject interpreted Hotpot AXIS input explicitly until
  its pinned toolkit mapping is separately frozen.

Performed:

- appended Python `StereoAnchorKind.AXIS`, common-native
  `StereoAnchorKind::AXIS=3`, and public C
  `HOTPOT_RD_STEREO_ANCHOR_AXIS=3`, retaining ATOM=1 and BOND=2;
- preserved every ordered axis row during copy/remap/component assembly and
  removed a record as a whole when an axis atom or consecutive constituent
  edge is absent;
- made permanent removal/clear of any constituent active or hidden bond drop
  the record, while hide/recover and unrelated chord deletion remain exact
  non-events;
- reused the flat anchor offset/count/index columns for variable-length paths
  and required `AXIS` count at least three in the common C++ view and public C
  ABI validators without adding chemical checks there;
- converted the RDKit graph builder from an implicit ATOM/else-BOND branch to
  an exhaustive ATOM/BOND/AXIS switch and returned stable `UNSUPPORTED` for
  AXIS before materialization; and
- retained Python Level-1's structural-storage behavior for malformed opaque
  paths, while the native structural boundary rejects AXIS counts 0, 1, or 2.

Verification:

- the final stereo/common-native/C-API selection reports 181 passed;
- rebuilt-extension native binding, C++ facade, and packing selection reports
  23 passed with the locked RDKit library path;
- independent audit reported 30 focused AXIS passes, 140 broader
  stereo/native passes, and 30 native-binding/coordination passes;
- the locked-toolchain C11 compile/link/run smoke passes and asserts the exact
  AXIS unsupported diagnostic, invalid lengths 0-2, unknown code, and null
  result ownership on invalid input;
- tests cover ordered path remap, missing atom/edge, copy, component split,
  target-first assembly order, atom deletion, hide/recover, both permanent
  hidden-clear APIs, chord deletion, flat buffer offsets, range checks, and
  additive numeric codes; and
- Python compilation and `git diff --check` pass.

Unexpected findings:

- the RDKit builder's previous binary branch treated every non-ATOM value as
  BOND. Adding only an enum would therefore have silently truncated AXIS to
  its first two rows; the switch had to be made exhaustive in the same
  checkpoint;
- a development attempt to add the native assertion to the old standalone
  `core_graph_transfer.cpp` executable did not reach AXIS because its existing
  tetrahedral-carrier fixture now fails an earlier established validation.
  AXIS coverage was retained in the passing public C-API smoke instead of
  weakening that unrelated validation;
- a system-compiler diagnostic link used the host libstdc++ and was
  incompatible with the locked library. The release-identity GCC/G++ link and
  run passed; only that pinned result is counted.

Adjustment:

- Level 1 remains lossless transport, not a path-chemistry validator;
- Gate 5 may now consume AXIS only after checking simple/maximal odd-length
  cumulative-double-bond paths and endpoint carriers;
- native ETKDG continues to report AXIS unsupported rather than claiming
  stereochemical preservation through an unreviewed `CHI_ALLENE` conversion.

Git checkpoint: `7817fab feat(stereo): add ordered AXIS transport`.

### Step 33 - Implement Level-2 Gate 5 axial semantics

Status: complete for AXIS and formal-atrop records; enhanced groups remain the
next separate Gate-5 checkpoint

Planned:

- consume the ordered Level-1 `AXIS` path only after validating its chemical
  graph meaning, while retaining a real ordered `BOND` for formal
  atropisomerism;
- freeze endpoint-scoped carrier, reversal, parity, coordinate-assignment,
  mapping, and diagnostic behavior without claiming a rotational barrier; and
- keep enhanced-group inversion semantics out of this checkpoint so their
  group-wide alternatives can be reviewed independently.

Performed:

- accepted simple, maximal odd atom paths of length 3, 5, ... only when every
  constituent edge has numeric order 2 and semantic kind `double`, and rejected
  internal off-axis neighbours;
- accepted formal atrop records only on a real numeric-order-1 semantic
  `single` bond, without inferring experimental isolability;
- required exactly two endpoint-local carriers at each end, limited public
  Gate-5 graph semantics to atoms and at most one implicit hydrogen per end,
  and retained the wider frozen Gate-1 algebra carrier contract unchanged;
- implemented deterministic parity under either endpoint swap, full ordered
  path/bond reversal, identity and non-identity atom mappings, and AXIS/BOND
  anchor-kind isolation;
- assigned `AX/even|odd` from the frozen axis triple product after projecting
  both ordered endpoint directions into the plane normal to the axis; and
- returned realized assignments and coordinate contradictions without
  mutating the molecule, its payload, or the caller coordinate array.

Verification:

- the complete `test_*stereo*.py` selection reports 208 passed;
- the focused Gate-1-through-Gate-5 selection reports 140 passed;
- fixtures cover all eight endpoint-swap/path-reversal combinations, allene,
  longer even-bond cumulene, formal atrop, explicit and implicit hydrogen,
  invalid paths and semantic bond kinds, rigid transforms, reflection,
  positive scale, degeneracy, contradiction, non-identity mapping, and input
  immutability;
- an independent review ran 500 randomized projection, terminal-swap, and
  path-reversal property cases and reported no failure after correction;
- Python compilation and `git diff --check` pass; `ruff` was not installed in
  the active development environment and is not claimed as executed; and
- neither RDKit nor Open Babel is used as the runtime AX semantic oracle.

Unexpected findings:

- the first coordinate implementation used the unprojected endpoint-direction
  norms. Its triple-product sign was correct, but longitudinal components
  could incorrectly change a resolved geometry into `DEGENERATE_GEOMETRY`;
- the first graph check required numeric order 2/1 and excluded only aromatic
  and dative kinds. Hotpot stores numeric order and semantic `BondKind`
  separately, so inconsistent `2 + single/unknown` and `1 + double/unknown`
  records were incorrectly admitted; and
- the frozen Gate-1 golden corpus deliberately contains an AX vacancy. The
  generic permutation algebra therefore cannot be narrowed to Gate-5's
  chemically supported atom/implicit-H subset.

Adjustment:

- endpoint directions are now projected first, checked against endpoint-local
  scales, normalized in the transverse plane, and only then used in the
  normalized triple product;
- cumulative path edges now require both order 2 and kind `double`, while the
  formal atrop axis requires both order 1 and kind `single`; and
- regression fixtures preserve the distinction between generic Gate-1
  algebra and public Gate-5 graph validation.

Git checkpoint: `b471052 feat(stereo): implement Level-2 Gate-5 axial semantics`.

### Step 34 - Complete Gate 5 enhanced-stereo group semantics

Status: complete

Planned:

- support `ABS`, `AND`, and `OR` only for the implemented tetrahedral and
  axial parity families while keeping grouped double-bond records explicit
  unsupported outcomes;
- compare relative groups by mapped member partition and group kind rather
  than transport provenance scope/ID; and
- evaluate coordinate consistency and alignment with one group-wide inversion
  bit instead of treating inverted members as unrelated contradictions.

Performed:

- retained ungrouped and `ABS` records as individually fixed facts and allowed
  them to align with one another only when each transformed descriptor agrees;
- aggregated local `AND`/`OR` membership by its complete `StereoGroupRef`, but
  compared different molecules only by the mapped member partition and kind;
- accepted either no inversion or a uniform inversion across every member of
  an `AND` or `OR` group, while keeping the two kinds chemically distinct;
- separated raw coordinate assignment from consistency checking so assignment
  always returns every realized TH/AX descriptor and group validation emits at
  most one deterministic issue for a partial inversion;
- skipped group consistency when any member lacks a coordinate assignment,
  preserving the member's graph/geometry issue without inferring a partial
  group state from incomplete evidence;
- split alignment transformation from group comparison, retained the
  transformed source descriptor, and adopted the matched target's group
  lineage in the aligned payload; and
- aggregated crossed split/merge partitions by connected component so each
  independent mismatch has one stable issue at its lowest reference record
  with the sorted target-space atom union.

Verification:

- the final dedicated enhanced-group file reports 30 passed and the complete
  `test_*stereo*.py` selection reports 238 passed;
- fixtures cover TH and AX groups, mixed TH+AX membership, `ABS`/ungrouped
  equivalence, AND/OR distinction, scope/ID independence, uniform and partial
  inversion, crossed partitions, target-record reordering, non-identity atom
  mappings, target group-ref adoption, coordinate non-mutation, and grouped DB
  rejection for all three group kinds;
- the incomplete-evidence regressions cover both a three-member coordinate
  group with two different assigned deltas plus one degenerate member and an
  incomplete alignment component beside an independent fixed mismatch;
- independent review reports 112 focused passes, 237 pre-final-fixture stereo
  passes, and 1,000 randomized partition/kind/parity/non-identity-mapping oracle
  passes with no blocker or correctness nonblocker;
- Python compilation and `git diff --check` pass; `ruff` remains unavailable
  in the active environment and is not claimed; and
- a wider `test_cheminfo` diagnostic reached 56 percent with no failure before
  it was deliberately interrupted after two minutes without output; it is not
  counted as a completed regression run.

Unexpected findings:

- comparing raw `StereoGroupRef` values across molecules would incorrectly
  reject chemically equivalent groups imported with different lineage scope
  or source ID;
- reporting each descriptor difference immediately would incorrectly reject a
  legal uniform group inversion and could produce one contradiction per
  member;
- an initial raw-assignment aggregation considered only members that assigned
  successfully. In a group of three, one fixed and one inverted assignment
  beside a degenerate member could therefore be mislabeled as a partial group
  inversion despite incomplete evidence; and
- globally suppressing group comparison after any failed record match would
  hide valid diagnostics from unrelated complete groups or fixed records.

Adjustment:

- group identity is local to one payload; cross-payload equivalence uses only
  mapped membership and `AND`/`OR` kind;
- coordinate consistency now runs only when all local group members have raw
  assignments;
- alignment skips only the incomplete connected component and continues to
  evaluate every independent complete component and fixed record; and
- `aligned_payload` uses target group refs but deliberately preserves the
  transformed source descriptor, making an accepted uniform inversion visible
  rather than cloning the target record.

Git checkpoint: `631ff90 feat(stereo): implement Gate-5 enhanced groups`.

### Step 35 - Add the reusable native-spec RI target for Gate 6

Status: complete

Planned:

- reuse the single native RI implementation for exact represented-graph
  automorphisms without adding VF2, NetworkX matching, or another C++ path;
- keep the adapter private and chemistry-neutral so Gate 6 owns graph labels,
  pin constraints, and semantic postfilters; and
- cache native target preparation while retaining independent lazy query
  cursors.

Performed:

- added private `_NativeSpecTarget(target_spec: dict)` in
  `hotpot.cheminfo.search.native`;
- constructed and retained one `NativeTarget` snapshot per adapter instance;
- created a new `NativeProblem` for every `iter_mappings(query_spec: dict)`
  call and returned lazy `query_key -> target_key` mappings aligned to native
  `problem.query_keys`; and
- preserved all native target, query, and cursor exceptions without wrapping
  them or adding a duplicate schema validator.

Verification:

- 49 native-spec, prepared-query, compiler, kernel, and production-kernel tests
  pass;
- a fake native module proves one target construction, one independent problem
  per query, shared target identity, and the mapping direction;
- the real extension returns the two deterministic automorphisms of a
  symmetric three-node path, and a private row-field pin selects only the path
  reversal on repeated calls;
- malformed target input preserves the direct native exception type and
  message; and
- independent review found no blocker and confirmed laziness, sparse external
  key handling, cursor/problem/target lifetime, cache scope, and absence of
  public or C++ API changes.

Unexpected findings:

- the existing `_NativeMappingSource` intentionally reverses mapping direction
  to satisfy a legacy NetworkX-style contract, so it cannot be reused directly
  for Gate-6 `query -> target` automorphisms; and
- an empty native-set constraint is a wildcard, not an impossible or unpinned
  sentinel. Point pins must therefore be omitted when absent and encoded with
  a concrete stable row value when present.

Adjustment:

- the new adapter reads cursor values directly instead of routing them through
  `_NativeMappingSource`;
- pin construction remains in `stereo.py`, while the adapter accepts only an
  already packed native query spec; and
- each pin uses a private native scalar row field, leaving unpinned node labels
  independent of graph-row identity.

Git checkpoint: `0a82896 feat(search): add reusable native spec target`.

### Step 36 - Project the complete represented graph for Gate 6

Status: complete

Planned:

- build one exact augmented graph over active, hidden-metal, and
  hidden-covalent bonds without mutating the molecule;
- represent each non-graph implicit protium as an ordinary hydrogen node so
  graph automorphisms can exchange explicit and implicit protium; and
- encode exact chemical labels and optional stable row pins for the reusable
  native RI target while retaining dative direction for a later semantic
  postfilter.

Performed:

- added the private `_Gate6Graph`, `_gate6_projection()`, and
  `_gate6_query_spec()` implementation in `hotpot.cheminfo.stereo`;
- projected atomic number, normalized isotope, formal charge, radical count,
  aromatic atom state, exact normalized bond order/kind/aromatic state, and
  all three represented bond stores into a deterministic native target;
- added chemically ordinary virtual protium nodes for
  `Atom.implicit_hydrogens`, with a private nonchemical row field used only
  when a query node is explicitly pinned;
- normalized explicit aromatic bonds and cyclic Kekule single/double bonds
  between aromatic atoms while retaining the single biaryl bridge identity;
- retained directed dative edges outside the undirected native labels for
  exact mapping postfiltering in the next step; and
- rejected duplicate/out-of-range atom facts, parallel edges, self-loops,
  non-finite bond orders, negative implicit-H counts, and unknown dative
  directions as deterministic unsupported projection states.

Verification:

- the complete previously tracked stereo selection reports 238 passed;
- projection, reusable-spec-target, and native production-kernel coverage
  reports 33 passed;
- real-reader regressions prove `[H]C` projects one explicit plus three
  virtual hydrogens and fully explicit methane projects no virtual hydrogen;
- a real native RI query proves that an explicit protium can be pinned to and
  exchanged with a virtual protium;
- independent review exercised 500 randomized valid graphs through
  projection, `NativeTarget`, and RI self-automorphism without failure;
- both independent chemistry and implementation audits found no remaining
  blocker; and
- Python compilation and `git diff --check` pass.

Unexpected findings:

- an initial review inferred from `Molecule.add_hydrogens()` that
  `implicit_hydrogens` was a total target count and proposed subtracting graph
  hydrogen neighbours. Real OB reader inputs and the search semantics proved
  that the field is normally the remaining non-graph count: subtraction turns
  `[H]C` into CH3 and rejects fully explicit methane;
- `Molecule.add_hydrogens()` itself leaves the old implicit count after
  materializing graph hydrogens, so that operation can create a stale core
  state inconsistent with normal reader semantics; and
- native graph data requires edge/record pairs in lexicographic edge order,
  including when a low-row virtual-H edge follows higher-row real edges.

Adjustment:

- Gate 6 consumes `Atom.implicit_hydrogens` directly and freezes that contract
  with real-reader tests; it does not use an ambiguous subtraction heuristic
  to hide the independent `add_hydrogens()` lifecycle defect;
- virtual hydrogens carry no chemical virtual-node flag, but every represented
  node carries a private target row value so source carriers can be pinned to
  either real or virtual target nodes; and
- edge records are sorted together after all real and virtual edges are
  assembled, preserving native graph alignment.

Git checkpoint: `3328f53 feat(stereo): add Gate-6 graph projection`.

### Step 37 - Prove Gate 6 carrier equivalence with native RI

Status: complete

Planned:

- classify TH carrier pairs and each DB/AX endpoint pair by exact labeled
  represented-graph automorphism rather than graph hashes or canonical ranks;
- preserve dative direction, every other supported stereo record, and
  enhanced-group semantics while completely excluding the record whose
  stereogenicity is under test; and
- retain unsupported Hotpot semantic records as a local tri-state boundary:
  fixed records do not block a result, while an otherwise viable mapping that
  moves one remains explicitly unresolved.

Performed:

- reused one `_NativeSpecTarget` per projected molecule and issued pinned
  whole-graph RI queries for all six TH pairs or the two endpoint-local DB/AX
  pairs;
- fixed every anchor/path row pointwise, converted supported explicit and
  implicit H carriers into the same augmented atom-token domain, and admitted
  a carrier equivalence only when at least one complete automorphism survives
  every semantic postfilter;
- compared the complete mapped directed-dative edge set and reused the frozen
  TH/DB/AX descriptor transformations plus enhanced-group partition/uniform-
  inversion logic for all descendant records;
- returned `NON_STEREOGENIC` for a proved equivalent pair, `UNSUPPORTED` only
  when all otherwise viable mappings move uninterpreted semantic stereo, and
  no issue when every topology mapping is disproved by exact supported facts;
- integrated Gate 6 into both public validation and assignment without
  changing public signatures or exports, and reused the same generic
  family-transform helper in public cross-molecule alignment;
- pointwise-pinned disconnected components containing no stereo semantic
  reference and added parent-local, exact atom/bond/dative H slots to remove
  irrelevant factorial hydrogen permutations; and
- exempted the current compared pair, every supported descendant H carrier,
  and the H references of opaque records from those nonchemical slots so
  symmetry breaking cannot alter stereo parity semantics.

Verification:

- the complete stereo selection reports 294 passed; adding the reusable-spec
  and production-kernel selections reports 308 passed;
- the dedicated Gate-6 equivalence file reports 37 passed and covers every TH
  pair, both DB/AX endpoints, exact atom labels, ordinary explicit/implicit
  protium equivalence, D/T distinction, hidden bonds, aromatic normalization,
  dative direction, descendant TH/DB/AX, AND/OR uniform and partial inversion,
  unsupported tri-state behavior, deterministic non-mutation, and bounded
  symmetry enumeration;
- a rejected descendant-stereo branch swap plus an unrelated implicit-H4
  component yields at most two mappings, while two equivalent methyl carriers
  yield exactly one mapping instead of the unlabelled H factorial;
- an independently supplied Si-C/C counterexample with explicit plus implicit
  H at both descendant centers reports the parent and both descendants as
  non-stereogenic, proving H-slot symmetry breaking cannot hide a required
  parity-compensating H exchange;
- an independent randomized oracle compared 500 labeled-tree TH cases with
  NetworkX automorphism orbits without disagreement, and independent enhanced
  TH/AX and dative/hidden/H/unsupported probes found no blocker;
- both chemistry and code/performance re-reviews found no remaining blocker;
  Python compilation and `git diff --check` pass; and
- `ruff` is unavailable in the active environment and is not claimed.

Unexpected findings:

- the first exact implementation was chemically correct but could enumerate
  every irrelevant automorphism after a topology swap was rejected by
  descendant stereo; one disconnected implicit-H4 component alone multiplied
  that path by 4!;
- the first H-slot optimization was too strong: applying an arbitrary slot to
  descendant explicit/implicit H carriers could prevent the odd H exchange
  needed to preserve the descendant descriptor, so a symmetric parent was
  falsely retained as stereogenic; and
- the earlier Gate-2 through Gate-5 fixtures intentionally used identical
  carbon leaves and sometimes stored an `IMPLICIT_H` carrier while the
  molecule declared zero implicit H. Those abstractions were valid before
  exact graph equivalence but correctly became non-stereogenic or unresolved
  under Gate 6.

Adjustment:

- record-free disconnected components use the pointwise identity because
  their automorphisms compose independently; components containing any
  current, supported, opaque, or cross-component enhanced-group record remain
  free to map;
- H slots are scoped by parent plus exact H atom identity, incident bond
  identity, and relative dative direction, and every H participating in
  interpreted or uninterpreted stereo algebra is exempt;
- the Si-C/C parity counterexample and both bounded-enumeration cases are
  permanent regressions; and
- legacy Gate-2 through Gate-5 tests now use chemically distinct carriers and
  exactly one declared implicit H wherever their original valid-case intent
  requires one. Their graph, coordinate, alignment, and enhanced-group
  assertions are otherwise unchanged.

Git checkpoint: `ca401d5 feat(stereo): prove Gate-6 carrier equivalence`.

### Step 38 - Derive the direct-priority Gate 6 stereo names

Status: complete

Planned:

- add the frozen derived-output contract without changing stored stereo
  identity, existing public function signatures, or two-argument assignment
  report construction;
- emit `R/S` and `E/Z` only when direct atomic number and explicitly available
  isotope mass completely order every required carrier group; and
- keep recursive, duplicate-node, ring, pseudoasymmetric, descendant-stereo,
  and axial CIP cases explicit unresolved outcomes rather than substituting a
  toolkit rank or graph order.

Performed:

- added the public frozen `StereoDerivedDescriptor` value and the backward-
  compatible `StereoAssignmentReport.derived=()` field;
- implemented a direct carrier-priority pass that treats implicit H,
  unlabelled explicit H, and explicit 1H as protium, orders D and T by mass,
  resolves repeated non-H elements only when every competing mass is explicit
  and unique, and never uses natural mass, atom row, canonical rank, or RDKit
  `_CIPRank` as a tiebreak;
- reused the frozen Gate-2 algebra after high-to-low carrier ordering, yielding
  TH `even -> S`, TH `odd -> R`, DB `together -> Z`, and DB `opposite -> E`
  independently at the two double-bond endpoints;
- derived names from the final coordinate-assigned return payload without
  mutating stored payload or input coordinates, marked ungrouped/ABS results
  absolute and AND/OR results non-absolute, and retained stable source-record
  indices and order;
- reused one complete Gate-6 validation result for graph, coordinate, and
  priority processing: a record-local graph issue suppresses only that record,
  while a global projection issue suppresses all derived names and additional
  priority diagnostics;
- returned one record-scoped `UNSUPPORTED` priority issue for an unresolved TH
  or DB direct order and for every otherwise valid AX record, whose axial CIP
  name remains outside the Stage-17 subset; and
- migrated the six pre-existing valid-AX assignment assertions to retain their
  coordinate/parity/group checks while also requiring the newly explicit AX
  priority boundary.

Verification:

- the dedicated derived-output file reports 32 passed, including all 24 TH
  carrier permutations, both DB endpoint and anchor directions, explicit and
  implicit H, T/D/H and 13C/12C ordering, conservative unlabelled-carbon ties,
  recursive/ring/descendant boundaries, enhanced groups, stable multi-record
  order, final-coordinate payload behavior, and negative-H-isotope rejection;
- the API plus dedicated derived selection reports 38 passed, the complete
  shell-expanded stereo selection reports 326 passed, and the complete native
  search selection reports 189 passed;
- the combined stereo plus reusable-spec and production-kernel selection
  reports 340 passed, of which 14 are the reusable native-target and production
  kernel regressions used by Gate 6;
- an explicit call-count regression proves assignment invokes complete graph
  validation once, and global versus record-local issue regressions prove the
  intended suppression boundary;
- independent chemistry review confirmed the direct-priority subset and the
  TH/DB mappings, including a pinned RDKit 2026.09.1 geometry oracle for the
  Hotpot `even -> S` convention; independent architecture, redundancy, and
  test reviews found no checkpoint blocker; and
- Python compilation, tracked and new-file whitespace checks, and line-length
  checks pass. The active environment still has no `ruff`, so no `ruff` result
  is claimed.

Unexpected findings:

- the first draft allowed a negative explicit H isotope to enter the sortable
  mass tuple; it could distinguish the Gate-6 graph but is not a chemical mass
  priority and must make a competing same-Z direct order unresolved;
- valid AX records previously expected an empty assignment issue list because
  Stage 17 had not yet attempted derived names; implementing the approved
  derived-output contract correctly exposes the frozen axial-CIP boundary;
  and
- a quoted `tests/test_cheminfo/test_*stereo*.py` argument is not expanded by
  this pytest installation and runs no tests. The initially reported wildcard
  aggregate was discarded and every recorded aggregate above was reproduced
  with shell-expanded paths.

Adjustment:

- normalize only valid H mass values for the direct-priority domain: ordinary
  protium is mass 1, positive D/T values are retained, and a non-positive
  explicit value cannot resolve a same-Z comparison;
- keep AX coordinate assignment intact but require the exact Stage-17
  `UNSUPPORTED` priority diagnostic and no derived AX descriptor; and
- use explicit paths or shell-expanded globs in subsequent verification logs,
  while preserving full recursive and axial CIP as intentionally deferred
  scientific work.

Git checkpoint: `9c8f4b8 feat(stereo): derive direct Gate-6 stereo names`.
