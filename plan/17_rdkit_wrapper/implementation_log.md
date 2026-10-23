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

### Step 39 - Freeze the Gate 7 square-pyramidal algebra

Status: complete for algebra; public coordination semantics remain unsupported

Planned:

- replace the temporary SPY sentinel with the approved numeric `SPY/1..30`
  grammar and proper-C4 permutation orbit;
- preserve every existing SP/TBP/OH index and transform while adding no new
  public class, function, or signature; and
- prove the finite algebra independently before graph, coordinate, alignment,
  or chelate semantics consume it.

Performed:

- fixed square-pyramidal ideal-site order as `+z,+x,+y,-x,-y` and added the
  four determinant-positive rotations that hold the apex fixed and cycle the
  square base;
- added `SQUARE_PYRAMIDAL: 30` to the finite coordination cardinalities and
  reused the existing single orbit-representative and carrier-reordering
  implementation rather than creating a SPY-specific transform algorithm;
- removed the two temporary SPY-unsupported branches from descriptor parsing
  and transformation, so only canonical decimal indices `1..30` are accepted;
- retained the original Gate-1 golden file byte-for-byte and created a
  separately versioned Gate-7 golden that freezes the site order, four proper
  rotations, all 30 lexicographic orbit representatives, all 30 results for a
  carrier 0/1 swap, and all 15 determinant-negative mirror pairs; and
- retained an explicit public no-cutover fence: valid `SPY/1` algebra still
  produces record-scoped `UNSUPPORTED` from validation, assignment, and both
  sides of alignment, with exact payload preservation and no derived output.

Verification:

- a test-first run produced the expected six failures for the missing SPY
  parser, transform, range, and rotation table before the production change;
- the independent oracle partitions all 120 permutations into 30 disjoint
  size-four C4 orbits, checks all 30 descriptors, and compares all 3,600
  descriptor/reordering combinations plus their inverse transformations;
- that oracle resolves reordered positions through deliberately non-contiguous
  carrier atom identities rather than reproducing the production inverse-
  permutation implementation;
- integer proper-rotation matrices independently induce the frozen C4 orbit,
  while a determinant-negative yz reflection independently induces the frozen
  mirror permutation and proves that no mirror pair is collapsed;
- the focused Gate-7/algebra/Gate-2 selection reports 89 passed, the dedicated
  SPY file reports 13 passed, and the complete stereo selection reports
  338 passed;
- the original Gate-1 golden equals its committed bytes and retains SHA-256
  `8d7856e0b593ff7af510e885b6f3fe2ef0ca047cd34a3563bde086af9afbcf4b`;
  and
- independent chemistry, architecture, and test reviews found no remaining
  checkpoint blocker; Python compilation, whitespace, and line-length checks
  pass.

Unexpected findings:

- appending SPY data to the existing Gate-1 `/1` golden would have weakened
  the evidence that the already frozen SP/TBP/OH numbering remained exactly
  unchanged, so the first local test draft was replaced rather than extended;
- the first 3,600-case oracle explicitly built the same inverse permutation as
  production and therefore could have repeated a direction bug; identity-
  semantic lookup was required for an independent proof;
- active rotation and site pullback permutations use inverse conventions, but
  the complete C4 group is inverse-closed; the tests now state and prove this
  instead of relying on an accidental tuple ordering; and
- pinned RDKit 2026.09.1 and OpenSMILES expose SP/TB/OH but no native SPY
  family. Hotpot SPY numbering therefore cannot be equated to RDKit's
  missing-ligand octahedral convention or delegated to a toolkit tag.

Adjustment:

- keep the Gate-1 corpus immutable and place all SPY facts behind the distinct
  `hotpot.stereo.level2-gate7-spy-algebra-golden/1` schema;
- keep graph validation, coordinate assignment, alignment, Gate-6 semantic
  context, and chelate comparison explicitly closed at this checkpoint; and
- stop before those semantics because the current plan does not yet freeze
  SP/TBP/OH ideal-site order, direction normalization and residual scoring,
  geometry/tie/ambiguity thresholds, chelate graph scope, or duplicate-family
  center policy. The full plan must be reread before that normative revision.

Git checkpoint: `0eb3536 feat(stereo): freeze Gate-7 SPY algebra`.

### Step 40 - Freeze the complete Gate 7 coordination contract

Status: complete for the normative revision; dependent implementation has not
started

Planned:

- stop after the algebra checkpoint because coordinate and chelate behavior
  could change descriptor identity, validity, and alignment semantics;
- reread the complete implementation plan before revising it, as required by
  the owner; and
- replace underspecified coordination statements coherently across governing
  decisions, Level 2, Phase 7, test obligations, non-goals, and approved
  implementation refinements.

Performed:

- reread all 1,625 pre-revision lines and traced every Gate-7 consequence
  through the public API, graph/stereo gates, benchmarks, phases, tests,
  packaging boundary, non-goals, and approved decisions;
- froze ideal site order as SP `+x,+y,-x,-y`, TBP
  `+z,-z,+x,(-1/2,+sqrt(3)/2,0),(-1/2,-sqrt(3)/2,0)`, SPY
  `+z,+x,+y,-x,-y`, and OH `+x,-x,+y,-y,+z,-z`, with
  `representative[site] == carrier_position`;
- limited coordination public graph semantics to one ungrouped atom-point
  SP/TBP/SPY/OH record per center, exact active-plus-hidden first-shell
  coverage, no element inference, and no requirement that coordination ligands
  be graph-distinct;
- made TH/SP/TBP/SPY/OH declarations mutually exclusive at one center without
  removing the already supported TH enhanced-group contract at other centers;
- froze all-record same-center rejection, stable group/non-atom/shell/coupled
  diagnostics, base-valid-only coupling analysis, opaque haptic/centroid
  preservation, and full represented-graph projection parity for a
  coordination-only payload;
- defined chelation as the carrier-pair same-component relation after removing
  only the current record's center-carrier edges, with coupled multicenter
  participation explicitly unsupported and independent centers retained;
- froze center-rooted unit directions, no donor-centroid subtraction, equal
  radial weight, determinant-positive orthogonal Procrustes/Kabsch fitting,
  and numerically stable `atan2(norm(cross), clipped_dot)` angular residuals;
- set the accepted angular-RMS score cutoff to 12 degrees with no separate
  per-carrier cutoff, the raw same-family uniqueness margin to 2 degrees, and
  the TBP/SPY cross-family margin to 3 degrees;
- required a strong alternate TBP/SPY fit to report
  `COORDINATE_CONTRADICTION` without changing family, while a boundary tie is
  `DEGENERATE_GEOMETRY` and only a unique declared-family fit can change its
  numeric index;
- retained coordination outside derived CIP, pairwise Gate-6 stereogenicity,
  automatic center discovery, enhanced groups, hapticity, and coupled-
  multicenter interpretation; and
- required private graph/chelate and coordinate helpers to be proved behind
  the existing unsupported fence, followed by one atomic public cutover across
  validation, assignment, alignment, record transforms, and Gate-6 context.

Verification:

- independent proper-rotation calculations reproduced the frozen nearest-
  wrong-descriptor angular RMS values: SP 63.639610 degrees, TBP 26.832816,
  SPY 32.970886, and OH 48.189685;
- the closest ideal TBP/SPY cross-family fit is 18.973666 degrees, and the
  12-degree boundary remains below half of the smallest same-family
  separation, TBP 26.832816 degrees;
- computational-chemistry review confirmed that 12 degrees is an RMS cutoff,
  `atan2` is equivalent to `acos` for normalized directions but numerically
  stronger, and an unfiltered raw second-best score must participate in the
  2-degree margin even when it lies just above 12 degrees;
- cheminformatics review confirmed the active-plus-both-hidden represented
  graph, base-valid coupling set, pairwise chelate partition, duplicate-ligand
  allowance, opaque haptic boundary, Gate-6 descendant semantics, and no-CIP
  scope;
- architecture review found the governing decision, detailed Gate-7 contract,
  Phase-7 sequence, tests, non-goals, and approved decision mutually
  consistent; and
- the documentation diff passes whitespace checks. No production or public
  behavior changed in this step, so the Step-39 unsupported no-cutover fence
  remains authoritative.

Unexpected findings:

- conventional point-cloud centering is chemically wrong for SPY: its ideal
  carrier-vector centroid is nonzero, so subtracting the donor centroid would
  change apex/equatorial weighting and descriptor meaning;
- the closest distinct TBP descriptor is only 26.832816 degrees away and the
  closest TBP/SPY shapes are 18.973666 degrees apart, so an unqualified
  `Kabsch + tolerance` statement could not define a deterministic assignment;
- the existing enhanced-group inversion machinery is binary and cannot be
  applied to 3/20/30-state coordination families without new chemistry;
- a coordination-only payload did not necessarily trigger Gate-6 projection,
  which could otherwise create two validity definitions for the same active
  plus hidden represented graph; and
- haptic and centroid carriers do not exist in the Level-1 enum, so claiming a
  dedicated Gate-7 haptic validator would require fabricating a representation
  rather than preserving the opaque adapter fact.

Adjustment:

- treat 12/2/3 degrees as transparent, versioned Hotpot engineering policy,
  not universal experimental chemistry constants;
- compare chelation through carrier-pair relations rather than component IDs,
  traversal order, metal heuristics, or fit weights;
- allow graph-equivalent coordination ligands while using exact descriptor and
  chelate semantics to constrain mappings; and
- implement Gate 7 through private golden/helper checkpoints before removing
  the public fence atomically, with no validate-only or coordinate-only
  intermediate public state.

Git checkpoint: `d2655b9 docs(plan17): freeze Gate-7 coordination semantics`.

### Step 41 - Clarify current implicit hydrogen in the Gate 7 shell

Status: complete for the normative clarification; private graph helpers remain
behind the public unsupported fence

Planned:

- pause the private adjacency test before it hard-codes whether Gate-6 virtual
  hydrogens participate in a coordination center's represented first shell;
- reread the complete implementation plan before revising it, as required by
  the owner; and
- resolve the ambiguity consistently in the governing decision, detailed
  Level-2 contract, test obligations, and approved implementation refinement.

Performed:

- reread all 1,803 pre-revision plan lines, including the current/source
  hydrogen lifecycle, Gate-6 represented graph, Gate-7 shell, implementation
  phases, tests, and non-goals;
- distinguished explicit atom-point carrier rows from Gate-6 virtual nodes
  created from the current implicit-hydrogen state;
- defined the Gate-7 first shell from the complete Gate-6 current-state
  projection: active bonds, both hidden-bond collections, and current
  implicit-H virtual edges all participate;
- kept the immutable source-hydrogen ledger outside shell construction, in
  agreement with the already frozen Level-1 lifecycle; and
- specified that an unmaterialized current implicit H cannot be silently
  ignored or coerced into an atom carrier: it produces the existing stable
  shell-set mismatch until represented as an explicit atom row.

Verification:

- the clarification preserves the all-ATOM-only Gate-7 domain and introduces
  no new public carrier kind, message, API, or fallback;
- explicit-hydrogen materialization remains the only mechanical route from a
  virtual current-state neighbour to a valid atom-point carrier;
- source-ledger-only hydrogen facts still cannot affect current native graph
  construction; and
- documentation whitespace checks pass. No production or public behavior
  changed, so the Gate-7 unsupported no-cutover fence remains authoritative.

Unexpected finding:

- excluding virtual hydrogens from an otherwise complete Gate-6 shell would
  allow a center with four explicit neighbours plus one current implicit H to
  pass an SP declaration as four-coordinate, even though the represented
  current graph contains five neighbours.

Adjustment:

- the private adjacency helper will retain Gate-6 virtual nodes for shell
  equality and reachability, while range-valid Gate-7 carriers remain real
  atom rows only.

Git checkpoint: `019acc7 docs(plan17): include current H in Gate-7 shell`.

### Step 42 - Add the private Gate 7 coordination graph primitives

Status: complete for the private graph checkpoint; every public coordination
family remains behind the existing unsupported fence

Planned:

- implement the represented-shell, all-record same-center, base-valid site,
  chelate-pair, and coupled-multicenter primitives specified by Gate 7;
- prove those private helpers directly before connecting any public validation,
  assignment, alignment, transform, or Gate-6 semantic path; and
- preserve the existing SP/TBP/OH and Gate-7 SPY algebra goldens byte-for-byte.

Performed:

- added explicit private coordination and atom-geometry family sets plus the
  minimal immutable `_Gate7CoordinationSite` record-index/descriptor/center/
  carrier-row DTO;
- derived one complete undirected adjacency from the existing Gate-6
  current-state projection, retaining active bonds, both hidden-bond classes,
  and current implicit-H virtual edges rather than creating a second graph;
- implemented deterministic all-record TH/SP/TBP/SPY/OH same-center rejection
  for successfully parsed, in-range atom anchors, so record order never picks
  a survivor;
- implemented the private SP/TBP/SPY/OH base validator for exact anchor,
  cardinality, atom-point, distinctness, range, center exclusion, ungrouped,
  and complete represented-shell requirements with the frozen diagnostics;
- kept virtual graph keys in shell equality while filtering them out of
  `StereoValidationIssue.atom_indices`, which remains a real molecule-row
  field;
- defined chelation only as the unordered carrier-pair same-component relation
  after removing the current site's center-carrier spokes;
- implemented coupled-multicenter detection through that single component-
  reachability implementation, covering shared carriers, centers used as
  carriers, and components reaching another declared center without redundant
  special-case scans; and
- left graph-equivalent ligand branches valid and deferred descriptor symmetry
  to the later RI-automorphism/descriptor-stabilizer semantic cutover.

Verification:

- the dedicated private Gate-7 graph file reports 31 passed across every
  coordination family, active/hidden/current-H shell coverage, source-ledger-
  only H exclusion, carrier/group/range failures, all-record same-center
  conflicts, chelate order independence, independent and coupled centers,
  a three-center coupling chain, graph-equivalent ligands, and the public
  no-cutover fence;
- the complete shell-expanded stereo selection reports 369 passed;
- stereo plus reusable native-spec and production-kernel regressions report
  383 passed;
- the complete native RI search selection reports 195 passed plus 49 subtests;
- the original Gate-1 golden remains SHA-256
  `8d7856e0b593ff7af510e885b6f3fe2ef0ca047cd34a3563bde086af9afbcf4b`,
  and the Gate-7 SPY golden remains SHA-256
  `fbda20439136c68ac50a2d508acc9970003e3721d73524c1c721ffff022a473d`;
- Python compilation, whitespace checks, and line-length checks pass; and
- independent architecture, cheminformatics, redundancy, and test audits
  found no remaining checkpoint blocker.

Unexpected findings:

- the first current-implicit-H diagnostic included the synthetic Gate-6 node
  key in `atom_indices`; shell semantics were correct, but that public
  diagnostic field must remain indexable against the real molecule table;
- direct shared-carrier and center-as-carrier scans duplicated the already
  sufficient component-reachability rule under exact-shell validity;
- a carrier path back to its own center is unreachable for a normal base-valid
  exact-shell site because all center incident spokes have been removed; the
  private branch is retained and tested with a synthetic helper input rather
  than misrepresented as a normal chemical fixture; and
- malformed/opaque exclusion can only be proved at the private parsed-site
  boundary in this checkpoint; automatic orchestration exclusion remains an
  integration obligation for the atomic public cutover.

Adjustment:

- compare the augmented shell exactly but expose only real atom rows in
  diagnostics;
- keep one component-reachability implementation for all multicenter coupling
  forms; and
- retain explicit-mapping, RI automorphism, and coordinate-assignment semantics
  as separate later concerns instead of prematurely quotienting duplicate
  ligands in the graph helper.

No performance benchmark is claimed for this private semantic checkpoint; no
public path invokes the new helpers yet.

Git checkpoint: `e099a3d feat(stereo): add Gate-7 coordination graph primitives`.

### Step 43 - Add the private Gate 7 coordinate classifier

Status: complete for the private coordinate checkpoint; public Gate-7
validation, assignment, alignment, transforms, and Gate-6 semantics remain
behind the existing unsupported fence

Planned:

- freeze an independent coordination-geometry golden before public use;
- implement one center-rooted, no-centroid, determinant-positive coordinate
  fitter for SP/TBP/SPY/OH;
- separate geometric scoring from the exact raw-radian 12/2/3-degree policy so
  threshold equality can be proved without fitting rounded boundary fixtures;
- test all frozen descriptors, mirrors, invariances, degeneracy, five-
  coordinate family discrimination, and the public no-cutover fence; and
- leave every public coordinate dispatcher untouched in this checkpoint.

Performed:

- added immutable ideal site directions for SP, TBP, SPY, and OH in their
  frozen order, then mapped each algebra representative using
  `representative[site] == carrier_position` into stored carrier order;
- added a single row-vector proper-Procrustes fitter using
  `H = ideal.T @ observed` and `R = U @ D @ Vt`, with `det(R)=+1`, no
  translation, no donor-centroid subtraction, no reflection, and equal unit-
  direction weights;
- documented the important objective boundary: Kabsch minimizes squared
  Euclidean chord lengths, while Hotpot computes the frozen angular RMS only
  after that rotation; no angular re-optimization occurs;
- computed each residual with `atan2(norm(cross), clipped_dot)`, ranked every
  candidate by raw `(score_radians, descriptor_index)`, and kept the complete
  unfiltered runner-up list;
- implemented the exact policy order for five-coordinate records: both-family
  12-degree failure, 3-degree TBP/SPY ambiguity, strong alternate-family
  contradiction without migration, then declared-family 2-degree numeric-
  index ambiguity;
- normalized center-to-carrier directions through two stable scaling stages:
  one site-wide coordinate scale prevents finite subtraction overflow, and
  one per-vector component scale prevents norm underflow while preserving the
  relative `1e-8` coincidence test;
- added the independent
  `level2_gate7_coordination_geometry_golden.json` with prior-golden SHA
  fences, raw-radian policy values, ideal templates, complete proper rotation
  groups, literal-y reflection maps, nearest-wrong scores/tie counts, the
  closest TBP/SPY score, and a no-centroid SPY probe; and
- kept the new helpers private and unreferenced by the existing public
  coordinate assignment dispatcher.

Verification:

- the dedicated coordinate file reports 46 passed, including all 83 ideal
  descriptors, every family rotation and mirror map, all global nearest-wrong
  ties, the 60 closest TBP/SPY ties, arbitrary translation/proper rotation/
  scale/independent-radius invariance, no-centroid SPY, exact inside/equal/
  outside 12/2/3-degree policies, real cross-family contradictions, and the
  public no-cutover fence;
- the combined Gate-7 algebra, graph, and coordinate selection reports
  90 passed;
- the complete shell-expanded stereo selection reports 415 passed;
- stereo plus reusable native-spec and production-kernel regressions report
  429 passed;
- the complete native RI search selection reports 195 passed plus 49 subtests;
- the original Gate-1 and SPY algebra goldens retain SHA-256
  `8d7856e0b593ff7af510e885b6f3fe2ef0ca047cd34a3563bde086af9afbcf4b`
  and `fbda20439136c68ac50a2d508acc9970003e3721d73524c1c721ffff022a473d`;
- the new coordination-geometry golden has SHA-256
  `a565dafee783d2fe9ffb0575b757ebb6869cb2634bcdfa5c1f9c2ca3055e8dc5`;
- Python compilation, whitespace, and line-length checks pass; and
- independent architecture, computational-chemistry, and test audits found no
  remaining checkpoint blocker.

Unexpected findings:

- constrained Kabsch and angular-RMS minimization are different objectives;
  describing the former as directly optimizing the latter would permit a
  future algorithm change that silently changes frozen descriptor scores;
- the 2-degree same-family boundary is not cleanly reachable inside the
  accepted region along the nearest ideal interpolation paths, so exact policy
  equality belongs in a pure raw-score test rather than a numerically forced
  coordinate fixture;
- square-planar reflection can preserve the descriptor because an in-plane
  reflection of its two-dimensional site set is realizable by a proper
  three-dimensional rotation; a universal "mirror must change index" test
  would therefore be chemically wrong;
- reconstructing large vector norms after per-vector scaling could overflow
  for finite coordinates, while a single site-wide pre-scale could make small
  local vectors underflow under a huge common translation; and
- the closest TBP/SPY distance has 60 symmetry-related ordered witnesses, so
  one descriptor pair alone is not a unique chemical boundary.

Adjustment:

- keep the chordal Kabsch rotation and angular score as two explicit stages;
- test policy boundaries from synthetic raw radians and test the fitter from
  physical coordinate fixtures;
- use the two-stage scale-free direction normalization before SVD; and
- preserve all descriptor/family decisions as private results until the graph,
  alignment, transformation, Gate-6 context, and coordinate paths can cut over
  atomically.

No production performance claim is made: no public or force-field path calls
the private coordinate classifier yet.

Git checkpoint: `035e2c2 feat(stereo): add Gate-7 coordinate classifier`.

### Step 44 - Cut over Gate 7 coordination semantics atomically

Status: complete for the public Gate-7 semantic checkpoint

Planned:

- remove the public SP/TBP/SPY/OH unsupported fence only after graph
  validation, coordinate assignment, generic transforms, explicit alignment,
  and Gate-6 descendant semantics are connected together;
- preserve exact atom identities and the frozen carrier-permutation algebra
  without introducing a chemical-symmetry quotient for equivalent ligands;
- compare mapped source chelate relations with target relations under explicit
  mappings and retain the same relation as a Gate-6 automorphism postfilter;
- keep coordination records out of pairwise stereogenicity and direct-priority
  CIP assignment; and
- reuse one represented-graph projection snapshot per public operation.

Performed:

- parsed every Level-2 payload once for graph validation, applied symmetric
  all-record TH/SP/TBP/SPY/OH same-center rejection first, retained the prior
  DB/AX duplicate-anchor rule, and skipped all further family semantics for
  blocked atom-centered records;
- connected the private Gate-7 base validator and coupled-multicenter detector
  to public validation over the complete Gate-6 current-state projection;
- made a syntactically valid `hotpot.stereo/2` head with a recognized
  SP/TBP/SPY/OH family trigger that projection even when its numeric value is
  invalid, while preserving the record-scoped `INVALID_DESCRIPTOR` result;
- routed accepted coordination records through the private no-centroid
  coordinate classifier, replacing only the numeric value in the declared
  family and leaving strong TBP/SPY family contradictions unresolved rather
  than migrating families;
- routed TH/SP/TBP/SPY/OH through the single generic carrier-permutation
  transform and gave every coordination record the explicit `(family,
  center-row)` alignment identity;
- computed source and target chelate relations on their own graph snapshots,
  mapped source atom-pair identities through the caller's explicit mapping,
  and rejected split or merged partitions with the frozen diagnostic;
- admitted valid coordination records into Gate-6 descendant semantic
  contexts, precomputed their chelate pairs in the reduced-payload index
  domain, and required both transformed descriptor and mapped chelate
  equality for every RI automorphism;
- kept coordination records out of Gate-6 carrier-pair candidates, so
  graph-equivalent ligands remain allowed and no coordination
  `NON_STEREOGENIC` or derived CIP result is produced; and
- reused each reference/target projection in validation and alignment, with a
  regression that proves validate and assign project once and alignment
  projects each side once.

Verification:

- the dedicated public Gate-7 file reports 108 passed, covering all 83 frozen
  descriptors, all four families, assignment agreement/contradiction,
  no-family-migration policy, generic transforms, explicit alignment,
  nonidentity chelate preservation, split/merge rejection, symmetric
  same-center failure, current/source-H behavior, coordination-only projection
  failures, independent/coupled centers, unsupported group/non-atom
  boundaries, and the atomic public-route sentinel;
- the complete stereo selection reports 487 passed;
- stereo plus reusable native-spec and production-kernel regressions report
  501 passed;
- the complete native RI search selection reports 195 passed plus 49
  subtests;
- the original Gate-1, SPY algebra, and coordination-geometry goldens retain
  SHA-256 values
  `8d7856e0b593ff7af510e885b6f3fe2ef0ca047cd34a3563bde086af9afbcf4b`,
  `fbda20439136c68ac50a2d508acc9970003e3721d73524c1c721ffff022a473d`,
  and `a565dafee783d2fe9ffb0575b757ebb6869cb2634bcdfa5c1f9c2ca3055e8dc5`;
- Python compilation, whitespace, and line-length checks pass; and
- independent architecture, cheminformatics, redundancy, and test audits
  found no remaining checkpoint blocker.

Unexpected findings:

- the existing generic transform dispatcher sent every non-TH/non-DB record
  through AX, so coordination support required an explicit atom-geometry
  branch rather than only removing validation's unsupported fence;
- the original Gate-6 carrier resolver grouped every non-TH family as two
  endpoint pairs and would silently truncate five- and six-coordinate carrier
  lists; the coordination branch now requires all-ATOM carriers and returns
  their complete ordered rows;
- computing chelation from mapped carriers on the target graph would compare
  the target partition with itself and miss a genuine source split/merge;
- independently validating and then aligning initially projected each
  coordination molecule twice; the projection result is now reused and its
  exact call count is tested; and
- a family-recognizable descriptor such as `SP/999` is still a declared
  coordination record for global represented-graph validity. Restricting the
  trigger to numerically valid descriptors would let malformed values bypass
  projection and create a weaker coordination-only graph contract.

Adjustment:

- keep family-head recognition separate from numeric-value validation so
  declared coordination cannot bypass global projection;
- retain separate Gate-6 and cross-molecule chelate caches because their
  reduced-payload and original-payload index domains differ, while both reuse
  the single `_gate7_chelate_pairs` and `_gate7_mapped_chelate_pairs`
  algorithms; and
- preserve exact explicit-mapping semantics for equivalent ligands without
  adding a symmetry quotient or a second matcher.

No production R0/F1 or 187/182 benchmark claim is made at this checkpoint;
the formal cohort artifact remains unavailable and the public semantic
cutover changes no force-field provider.

Git checkpoint: `a948279 feat(stereo): cut over Gate-7 coordination semantics`.

### Step 45 - Separate and rerun R0 Level-2 evidence

Status: complete for the canonical 187-ligand R0 rerun

Planned:

- retain the byte-frozen 187-ligand corpus, `1/8/32` matrix, physical-line
  case identities, seeds, native ETKDG options, determinism rerun, and primary
  generation-throughput denominator unchanged;
- report stereo-payload coverage before interpreting any stereo parity result;
- run public Level-2 validation, assignment, and identity alignment only as a
  detached post-generation observation, with its own timing and without
  changing native status, coordinates, or throughput; and
- classify inputs without a declared `hotpot.stereo/2` record as
  `NOT_APPLICABLE`, never as a valid or equivalent stereo result.

Performed:

- extended the R0 input snapshot with record, atom-transport,
  bond-transport, nonempty-payload, and declared-Level-2 counts;
- added primitive-only serializers for public validation issues and derived
  descriptors, then recorded validation, assignment, and identity-alignment
  evidence independently for each applicable returned conformer;
- assigned onto a molecule clone before alignment so the source payload
  remains immutable and alignment is not a vacuous source-to-source call;
- gave the Level-2 observer a fresh C-contiguous `float64` coordinate copy,
  included that copy cost only in `level2_observation_seconds`, and added a
  mutation witness proving that the observer cannot alter the native result;
- treated both a complete `hotpot.stereo/2/...` descriptor and the malformed
  bare `hotpot.stereo/2` namespace declaration as applicable, allowing the
  public validator to report the latter as `INVALID_DESCRIPTOR`; and
- bumped the benchmark schema to `hotpot.plan17.r0/2`, added separate
  applicability and issue aggregates, and exposed the returned-conformer
  count covered by `NOT_APPLICABLE` without counting it as a pass.

Verification:

- the focused R0 runner selection reports 12 passed;
- the combined stereo and `rdWrappers` selection reports 534 passed and 2
  skipped;
- Python compilation, whitespace, line-length, JSON `allow_nan=False`, and an
  independent read-only R0 audit report no blocker;
- the clean-tree formal rerun used commit
  `90dd0923199c57b083cb8f8d1f952a86ba090b85`, RDKit `2026.09.1`, Python
  `3.11.16`, one outer worker, and one ETKDG internal thread;
- all 561 matrix trials were emitted: 559 `SUCCESS` and 2 deterministic
  `PARTIAL` results, both for physical case 10, returning 4 of 8 and 18 of 32
  requested conformers respectively;
- all 7,649 returned conformers were finite and atom-row exact; atom-order,
  topology, Level-1 transport, input-nonmutation, and same-seed mismatch
  failure counts were all zero;
- aggregate primary-generation time was 78.004250637 seconds, giving
  7.191915766 ligands/second and 98.058758818 conformers/second; detached
  Level-2 observation added 0.017287096 seconds outside that denominator;
- all 561 input trials had zero stereo records, atom transport, and bond
  transport, so all 7,649 returned conformers are explicitly
  `NOT_APPLICABLE`; assessed, valid, invalid, equivalent, and non-equivalent
  Level-2 conformer counts are all zero; and
- run-specific artifacts are
  `/tmp/hotpot-plan17-r0-gate7-90dd092/r0_cases.jsonl` with SHA-256
  `2b630f6d492955a31467daaa8525c7c811d9f3954391da0be17ce14b428dc39a`
  and `r0_summary.json` with SHA-256
  `63ac2945abb8c5b82236f6303951873ec31bfdcedae1f90611077514953640f7`.

Unexpected findings:

- the canonical SMILES corpus contains no Level-2 sidecar, and the current
  public Open Babel import path produces an empty stereo payload for every
  case, including the source row containing SMILES `@` markers;
- therefore the prior zero `level1_parity_failure_count` proves only that an
  empty transported payload remains empty, not that source chirality was
  imported and preserved; and
- the two partial results and 7,649 total returned conformers exactly match
  the prior R0 run, while the new primary conformer throughput differs by only
  about -0.21%, consistent with ordinary run-to-run timing variation because
  Level-2 observation is excluded from the denominator.

Adjustment:

- keep the current canonical R0 report honest by using
  `NOT_APPLICABLE` rather than deriving a stereo pass from absent facts;
- require any future Level-2 sidecar to be an independently hash-governed
  artifact attached only to a post-generation observation clone, never to the
  native ETKDG input; and
- retain Level-1 nonmutation as transport evidence only until an importer or
  explicit sidecar supplies nonempty source stereo facts.

This rerun closes the standalone R0 generation matrix, but it makes no formal
F1/C1/C2 or 182-complex claim; the frozen 182-case manifest and graph sidecars
remain unavailable.

Git checkpoint: `90dd092 test(rdwrapper): separate R0 Level-2 evidence`.

### Step 46 - Add bounded external-toolkit stereo observations

Status: complete for the Gate-7 toolkit-observation checkpoint

Planned:

- use pinned RDKit only where it independently perceives SP, TBP, and OH
  configurations from three-dimensional coordinates;
- keep Hotpot descriptor numbers distinct from RDKit's private permutation
  numbers through an explicit version-frozen lookup;
- use Open Babel only within the narrower stereo families it can parse and
  inspect, in an isolated process; and
- state every non-covered semantic boundary explicitly rather than extending
  either toolkit beyond its real model.

Performed:

- added a frozen Hotpot-to-RDKit `2026.09.1` lookup for all 3 SP, 20 TBP, and
  30 OH descriptors and rebuilt independent RDKit graphs from each Hotpot
  ideal-coordinate template;
- required RDKit's perceived chiral tag and `_chiralPermutation` to match all
  53 lookup entries, with one proper-rotation, x-reflection, and first-carrier
  exchange witness for every family;
- saved, enabled, and restored RDKit's process-global non-tetrahedral
  chirality switch in a module fixture so results do not depend on prior test
  state and the suite leaves no global-state leak;
- added an Open Babel 3.2.x subprocess capability witness showing that
  selected TH, DB, and SP strings are specified and distinguishable while
  retaining the same carrier-reference multiset; and
- explicitly excluded SPY, AXIS, chelate partitions, enhanced groups, and
  Hotpot's 12/2/3-degree acceptance policy from external-toolkit claims.

Verification:

- the dedicated toolkit-observation selection reports 65 passed;
- the combined stereo and `rdWrappers` selection, including the two supported
  import-order subprocesses, reports 601 passed and 2 skipped;
- Python compilation, whitespace, and line-length checks pass; and
- an independent cheminformatics review found no remaining blocker in the
  RDKit mapping, transform witnesses, Open Babel scope, or stated exclusions.

Unexpected findings:

- RDKit's non-tetrahedral perception is controlled by mutable process-global
  state, so relying on its ambient default would make the test order-sensitive;
- Open Babel stereo configuration semantics include more than raw carrier
  references; comparing only `refs` cannot establish Hotpot descriptor
  equivalence; and
- exact Open Babel `3.2.1` would be narrower than the declared project range
  `>=3.2.1,<3.3`, whereas the capability witness is valid for 3.2.x.

Adjustment:

- describe RDKit as an independent toolkit observer of Hotpot's templates,
  not as an independent source of those coordinates;
- describe Open Babel strictly as a parsing-and-distinction capability
  witness, not as a Hotpot semantic oracle; and
- keep `_chiralPermutation` evidence pinned to the exact RDKit release while
  accepting the declared Open Babel 3.2.x compatibility range.

Git checkpoint:
`c79d02c test(stereo): cross-check Gate-7 toolkit observations`.

### Step 47 - Execute the local Phase-8 release regression

Status: complete for local source/native validation; external release gates
remain open

Planned:

- rerun force-field, native geometry, trajectory, working-copy, search,
  inference, pipeline, stereo, `rdWrappers`, packaging, and public-API tests;
- prove both supported Hotpot/Python-RDKit import orders in clean processes;
- rebuild the Python extension and standalone SDK from the locked offline
  closure in new directories, then rerun C11/C++17 ABI consumers; and
- audit current-build symbols, dynamic dependencies, RPATH, installed headers,
  and licenses without treating a probe artifact as a release wheel.

Performed:

- added subprocess regressions for Hotpot-then-RDKit and
  RDKit-then-Hotpot, each exercising real Hotpot and RDKit molecule operations;
- intentionally did not run OpenBabel-first followed by Python RDKit because
  that unmodified third-party-wheel order is the known isolated segfault
  diagnostic, not a supported order;
- ran the repository's curated functional-regression target set directly
  after the coverage wrapper reported that the local environment lacks the
  optional `coverage` module;
- redirected model and Hugging Face caches to writable `/tmp` roots and reran
  every model-backed failure caused by the sandbox's read-only default cache;
- verified all six source archives offline and rebuilt both artifacts in 164
  Ninja steps with the pinned GCC 14.3/CMake 3.31.8/Ninja 1.12.1 probe
  toolchain;
- installed the fresh SDK and its CMake package, public headers, notices, and
  licenses under `/tmp/hotpot-plan17-phase8-sdk-02`; and
- exercised the new extension and SDK rather than reusing the earlier
  development artifacts.

Verification:

- the curated functional selection initially reported 2,435 passed, 12
  skipped, 49 subtests passed, 2 failed, and 9 setup errors; all 11 nonpassing
  cases had the same read-only default model-cache cause rather than an
  assertion or implementation mismatch;
- the redirected model-cache rerun reported 55 passed before one additional
  MCA stdout assertion exposed a Hugging Face Xet logging fallback to the same
  read-only cache; redirecting `HF_HOME` and `XDG_CACHE_HOME` made that final
  case pass;
- the complete stereo and `rdWrappers` selection reports 601 passed and 2
  skipped;
- the fresh-extension/fresh-SDK `rdWrappers`, C/C++ consumer, public API, and
  packaging selection reports 71 passed;
- C11 and C++17 consumers compiled, linked, and ran, including both
  `_GLIBCXX_USE_CXX11_ABI=0` and `=1` through the installed CMake package;
- the SDK exports exactly 15 `hotpot_rd_*` functions and its version node; the
  extension exports only `PyInit__rd_native` and its version node;
- neither artifact has RPATH/RUNPATH or a dynamic RDKit/Boost dependency, and
  SDK `ldd -r` is clean with the recorded probe-toolchain runtime;
- installation contains four public headers and eight license/notice files;
- fresh runtime identity is Hotpot `0.5.4.0`, native RDKit `2026.09.1`; and
- the run-specific extension and SDK SHA-256 values are
  `0763fe8fad5464e858e745097950179d9062a4ab26c107a2d44749eddf7f2248`
  and `dc8d1a7a53cc6dfa4f47995eef29a48ad8af557ba458654aebadf859152d86a4`.

Unexpected findings:

- `tests/run_coverage.sh` cannot produce coverage artifacts in the current
  environment because `coverage` is absent; the identical functional targets
  were executed directly with pytest instead;
- model-backed tests first failed because their default cache is outside the
  writable sandbox, and Hugging Face Xet subsequently wrote its cache error to
  captured stdout; explicit writable model, Hugging Face, and XDG cache roots
  closed both environmental failures;
- the first fresh native build verified and extracted every source archive
  but stopped before CMake because the active Python lacks `pybind11`; using
  the already isolated build-dependency directory on temporary `PYTHONPATH`
  allowed the second new-directory build to complete without installing or
  changing the environment; and
- the probe SDK requires GLIBCXX/CXXABI versions newer than the host system
  GCC 9 runtime. It loads and passes with the recorded GCC 14.3 toolchain
  runtime, but ABI-0/ABI-1 facade compatibility does not mean arbitrary old
  libstdc++ runtime compatibility.

Adjustment:

- report the curated regression as staged closure with explicit environmental
  reruns, not as one initially green coverage invocation;
- keep the OpenBabel-first crash as a documented isolated diagnostic and test
  only supported clean-process import orders in normal CI;
- scope SDK compatibility claims to the tested ABI modes and toolchain runtime;
  and
- leave the CPython 3.10-3.14 manylinux wheel matrix, auditwheel output, and
  formal 182-case O0/F1/C1/C2 matrices open rather than replacing them with a
  CPython 3.11 probe build or development smoke cohort.

The local Phase-8 checkpoint is closed. Stage-17 release validation remains
blocked by the unavailable frozen 182-case manifest/native sidecars, the
missing dedicated O0/F1 benchmark runner, and the unexecuted five-interpreter
manylinux wheel workflow.

Git checkpoint: `10bc417 test(rdwrapper): cover supported import orders`.

### Step 48 - Reconfirm C1/C2 smoke without widening its claim

Status: complete as current-build development evidence; formal 182 evidence
remains unavailable

Planned:

- rerun the existing one-case C1/C2 smoke with the freshly rebuilt native
  extension at conformer counts `1/8/32`;
- preserve its permanent `development-smoke` and
  `formal_claim_allowed=false` labels; and
- search the authorized ai-14 study/test roots for the exact frozen formal
  manifest before retaining the external blocker.

Performed:

- preloaded the fresh source-tree development extension and ran the unchanged
  detached C1/C2 benchmark runner against the one-case native-graph fixture;
- kept force fields, Open Babel, `hotpot.ff`, geometry gates, perturbation,
  repair, and selection outside both routes; and
- searched `ai-14:~/mca_test` and `ai-14:~/mca_study` by relevant file names,
  recorded digest text, and the exact formal-manifest SHA-256.

Verification:

- all six route/count cells returned `SUCCESS` with the full requested count;
- every coordinate batch was finite and exact-shape, every fixed-seed rerun
  was byte-deterministic, and every input graph was preserved;
- C2 maximum detached bound violation was `0.001949029236` Angstrom at all
  three counts, with strict satisfaction counts `0/1`, `7/8`, and `29/32`;
- C1 maximum detached bound violations were `0.359520734434`,
  `0.438428833325`, and `0.642839307960` Angstrom, with strict satisfaction
  counts `0/1`, `2/8`, and `7/32`; and
- `/tmp/plan17-c1c2-smoke-phase8-c79d02c/report.json` has SHA-256
  `3c05f6b3862a1c19baddd1e3ad1776ea94992d5c6c00e96e7dbfae01d439070e`.

Unexpected findings:

- unlike the R0 runner, the C1/C2 development CLI has no explicit path for
  injecting an out-of-tree extension. Its first source-tree invocation stopped
  before generation because `_rd_native` was not installed in the package;
  preloading the exact fresh extension into its package-qualified module name
  allowed the unchanged runner to execute; and
- no file under either authorized ai-14 root matched formal manifest SHA-256
  `4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377`.
  An unrelated 256-case MCA manifest was not treated as a substitute.

Adjustment:

- retain the one-case result only as a route/runtime smoke and do not infer
  whole-cohort chemical performance from it;
- do not add a production loader or public option solely for development
  extension injection; and
- keep formal C1/C2, O0/F1, and provider-selection decisions blocked until the
  exact 182 manifest and native graph sidecars are restored or the owner
  explicitly approves a newly frozen cohort.

No production-code checkpoint is associated with this evidence-only rerun.

### Step 49 - Publish the implementation and validation report

Status: complete for the implementation handoff; release gates remain open

Planned:

- consolidate the final architecture, public APIs, implementation route,
  validation evidence, benchmarks, artifact identities, and remaining gates;
- distinguish component/local completion from formal workflow and wheel
  release completion; and
- update the Stage-17 index without hiding any unavailable evidence.

Performed:

- added `implementation_report.md` with the Python, C ABI, C++17, common-view,
  force-field seam, R0, C1/C2, Level-1, Level-2, build, packaging, and release
  outcomes;
- included exact R0 and development-smoke benchmark tables, test selections,
  source/golden/binary/report hashes, checkpoint chronology, supported claims,
  and explicit non-claims; and
- updated `README.md` to link the report and state that component
  implementation/local validation are complete while release closure is not.

Verification:

- an independent final audit checked every R0/C1/C2 number, accessible artifact
  hash, stereo golden, source lock, API summary, native export count, installed
  header/license count, test statistic, chronology entry, and non-claim;
- the audit found no remaining blocker after the README status was aligned with
  the report; and
- Markdown whitespace, line-length review, link targets, Python compilation of
  both benchmark runners, and `git diff --check` pass.

Unexpected finding:

- the first README status listed the missing 182 artifacts and wheel matrix but
  omitted the separately missing formal O0/F1 runner and provider-decision
  evidence. The final status now names all three release-blocker classes.

Adjustment:

- call the delivered code and local validation complete, but do not call Stage
  17 release-complete;
- keep Open Babel as the default force-field coordinate provider; and
- require the formal 182 artifact/runner evidence and five-version wheel audit
  before a release sign-off or default-provider change.

No production-code checkpoint is associated with this reporting-only step.

### Step 50 - Isolate and harden the release-wheel payload

Status: packaging fix complete; final CPython 3.10-3.14 matrix pending

Planned:

- execute a CPython 3.11 diagnostic build from a clean committed source tree
  before spending the full five-interpreter matrix;
- audit installed behavior, import order, native linkage, exports, licenses,
  and archive membership rather than accepting a repaired filename alone; and
- preserve the sdist native-build closure while keeping the Python wheel a
  runtime artifact rather than an undeclared C++ SDK.

Performed:

- pinned the local diagnostic to the release workflow's exact
  `manylinux_2_28_x86_64` image digest, cibuildwheel 4.3.0, the six verified
  locked source archives, and the `manylinux_2_28_x86_64` build profile;
- disabled cibuildwheel's host-source mount and moved the build to a clean
  detached worktree so ignored host-built extensions could not enter the
  container source tree;
- added a fresh-process `RDKit -> Hotpot` installed-wheel smoke alongside the
  existing `Hotpot -> RDKit` route and asserted both the Python distribution
  identity `2026.9.1` and native runtime identity `2026.09.1`;
- separated setuptools package data from sdist build inputs with exact
  namespace-package exclusions, retaining only the rdWrapper stub and runtime
  notice before CMake installs the extension and seven dependency licenses;
  and
- added a stdlib-only pre-upload wheel-content gate with focused tests. The
  gate strictly admits the runtime rdWrapper payload, requires one ABI
  extension and the project license, and rejects wrapper sources, headers,
  CMake inputs, common native headers, locked RDKit sources, and SDK artifacts.

Verification:

- the diagnostic CPython 3.11 wheel built, repaired, installed, and passed both
  import-order smokes at source commit `0719e2a`; its size was `5,678,830`
  bytes and SHA-256 was
  `523a5c5d70eef22152c14f6fee87ab07a732a4a1d2dfb98e3bcf018fe27ca36a`;
- `_rd_native` had no RPATH/RUNPATH, no dynamic RDKit, Boost, or Conda
  dependency, only system runtime dependencies, and only
  `PyInit__rd_native` as a function export;
- a fresh setuptools `build_py` staging tree contains only
  `__init__.py`, `contracts.py`, `native.py`, `sampler.py`, `_rd_native.pyi`,
  and `_native/THIRD_PARTY_NOTICES.md` under rdWrappers, while common `_native`
  contains only its two Python runtime modules;
- a fresh sdist still contains the wrapper CMake files, build driver, native
  source and headers, common view headers, source lock/fetcher, and probe
  closure; and
- the focused packaging suite reports `19 passed`; Ruff, workflow YAML
  parsing, formatting, and `git diff --check` pass. The new auditor rejects the
  diagnostic wheel and enumerates all leaked build/SDK members.

Unexpected findings:

- the first attempted audit used a source tree containing ignored host-built
  extensions and therefore was not valid release evidence. A clean detached
  worktree eliminated that contamination;
- the otherwise functional diagnostic wheel contained 16 rdWrapper
  source/header/CMake/build members plus two common native headers because
  `include-package-data = true` allowed `MANIFEST.in`'s sdist closure into the
  wheel through separately discovered namespace packages; and
- `auditwheel show` classifies the whole Hotpot wheel as `linux_x86_64`, not as
  strict standalone manylinux, because `_ob_native` intentionally retains the
  declared external `libopenbabel.so.8` dependency and its package-relative
  RUNPATH. This does not affect `_rd_native`'s self-contained native closure,
  but repaired dual tags must not be described as strict auditwheel closure.

Adjustment:

- use exact setuptools namespace keys instead of parent-package globs;
- keep `MANIFEST.in` unchanged so source builds remain reproducible;
- retain the declared Open Babel packaging exception and report it explicitly;
  and
- treat the CPython 3.11 artifact only as diagnostic evidence. Rebuild and
  audit all five interpreters from checkpoint `37a08b3` before changing the
  wheel-matrix release status.

Git checkpoint: `37a08b3 build(rdwrapper): harden release wheel payload`.

### Step 51 - Execute the five-interpreter release-wheel matrix

Status: complete for the pinned local workflow-equivalent matrix; a
GitHub-hosted Actions run is not claimed

Planned:

- build CPython 3.10-3.14 wheels from the clean packaging checkpoint rather
  than reuse the contaminated CPython 3.11 diagnostic artifact;
- reproduce the release workflow's pinned manylinux image, cibuildwheel,
  locked native sources, repair policy, installed-wheel tests, and upload
  content gate locally; and
- inspect the final archives and `_rd_native` ELF closure independently while
  preserving the declared cross-wheel Open Babel dependency exception.

Performed:

- created a clean detached source tree at exact commit
  `37a08b30f785bf54149436ae2ce5d852ac7cbd22` and verified that it remained
  clean throughout the builds;
- used cibuildwheel 4.3.0 with the exact
  `quay.io/pypa/manylinux_2_28_x86_64` image digest
  `sha256:39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82`,
  the six hash-verified locked native source archives, and offline native
  source mode;
- built and repaired one wheel for each of CPython 3.10, 3.11, 3.12, 3.13,
  and 3.14, then installed each wheel into cibuildwheel's isolated test
  environment;
- ran the installed-wheel Hotpot-to-RDKit and fresh-process RDKit-to-Hotpot
  import-order smokes, search and relevant-cycle checks, `_ob_native` and
  `_rd_native` imports, and an ETKDG sample shape/status check for every ABI;
- asserted Python distribution identity `rdkit==2026.9.1` and native runtime
  identity `2026.09.1` in every installed-wheel test; and
- ran the strict archive-content auditor, ZIP-integrity and duplicate-member
  checks, embedded-tag inspection, and a direct ELF dynamic/export audit on
  every final artifact.

Verification:

| ABI | Wheel size (bytes) | SHA-256 |
| --- | ---: | --- |
| cp310 | 5,632,596 | `c381bfdc10746b60791ab30a699ac57240a748ad5e515a96ae993008fc75d34f` |
| cp311 | 5,637,406 | `fc1cecff6ac43b050cd6f57c1241e355c746eb5874a090b12cedeb9946c78eba` |
| cp312 | 5,659,599 | `4fa3f7141e8990bf38c54fdb426952f293f98f7e95c7e41e08842e5cc7839527` |
| cp313 | 5,659,946 | `717ce94bc6a8c36b519da474ae240dfb8737eab8fa12c5b381dcd1eacccdb446` |
| cp314 | 5,662,757 | `9c0fa5382727d8f7d9fc3ad51a9fe897679efaf6f4ef8de2264b0297355e7003` |

- every artifact is named
  `hotpot_zzy-0.5.4.0-<abi>-<abi>-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
  and embeds exactly the matching `manylinux_2_27_x86_64` and
  `manylinux_2_28_x86_64` tags;
- all five strict content audits pass. Each archive has 869 ZIP entries, 761
  files, and 14 rdWrapper runtime files, with no duplicate member and no
  wrapper source, header, CMake, common-native-header, locked-source, or SDK
  contamination;
- every `_rd_native` extension is a stripped 64-bit x86-64 ELF with no RPATH
  or RUNPATH and no dynamic RDKit, Boost, or Conda reference. Its `NEEDED`
  set is exactly `libpthread.so.0`, `libstdc++.so.6`, `libm.so.6`,
  `libgcc_s.so.1`, `libc.so.6`, and `ld-linux-x86-64.so.2`;
- every `_rd_native` extension exposes exactly one globally defined function,
  `PyInit__rd_native@@HOTPOT_RDWRAPPERS_PYTHON_1.0`; and
- the respective `_rd_native` build IDs are
  `855d8edef80a3d34c7b8689bc18c945578794d61`,
  `102e41a9157ec500702abf70a37a3d7ddbe590da`,
  `c4948122fc1f7545d84ae22ab21a776b2a564f68`,
  `36d867ec2b7aab9258f2dd82cd6801e484d76ac8`, and
  `6dff41cd32fcfd9659953526b883657421cacf96` for cp310 through cp314.

Whole-wheel `auditwheel show` reports all five artifacts as compatible with
`linux_x86_64`, with versioned system symbols constraining them to
`manylinux_2_27_x86_64`. It does not grant strict standalone manylinux policy
compliance because the pre-existing `_ob_native` packaging model deliberately
excludes `libopenbabel.so.7` and `libopenbabel.so.8` from repair. The pinned
Python `openbabel` distribution supplies `libopenbabel.so.8`, and all five
isolated coexistence tests pass. This is a controlled cross-wheel dependency,
not a strict standalone-auditwheel claim.

Unexpected findings:

- the first temporary launcher omitted cibuildwheel's private `PYTHONPATH` and
  stopped before Docker or any build; the corrected launcher used the pinned
  tool installation;
- the private download cache initially contained only the CPython 3.11 RDKit
  wheel. `PIP_FIND_LINKS` alone did not force the local candidate, so exact
  ABI-specific direct-reference constraints were added for the PyPI-hash-
  verified RDKit 2026.9.1 wheels;
- the first CPython 3.10 container was intentionally stopped after source and
  dependency identity verification while its unconstrained test-dependency
  installation remained slow. It copied no artifact; the constrained rerun
  completed and passed;
- the first CPython 3.14 container was likewise intentionally stopped after
  verification while PEP 517 attempted to resolve an Open Babel build
  dependency. Adding the locally verified OpenBabel 3.2.1 wheel to the exact
  constraint set made the clean rerun complete and pass; and
- only task-owned containers were stopped. No unrelated process or container
  was modified, and every reported artifact comes from a complete successful
  run.

Adjustment:

- retain ABI-specific, hash-verified dependency constraints as local
  reproducibility support without changing the release workflow's dependency
  identities;
- close the local five-interpreter wheel-matrix gate under the declared Open
  Babel packaging model;
- do not represent this local pinned reproduction as GitHub-hosted CI
  provenance or as strict standalone auditwheel manylinux compliance; and
- keep Stage-17 release completion open for the unavailable frozen 182-case
  O0/F1/C1/C2 evidence and the unresolved default-provider decision.

No new production-code checkpoint is associated with this evidence step. The
matrix source checkpoint is
`37a08b30f785bf54149436ae2ce5d852ac7cbd22`.

### Step 52 - Freeze the corrected force-field validation contract

Status: complete; implementation authorized and active

Planned:

- replace the obsolete byte-identical historical-artifact prerequisite with a
  current semantic 187/182 cohort;
- freeze the owner-corrected force-field acceptance semantics, route matrix,
  experiments, and ancillary data deliverables before changing code; and
- keep the default-provider cutover separate from authorization to execute the
  validation.

Performed:

- reread the complete validation plan after the final owner corrections;
- froze O0/F1/F2/C1/C2 route semantics, the A01-A10 acceptance contract, the
  full-182 F1/default threshold, mandatory C1/C2 continuation, and F1 repeat;
- recorded that an intermediate non-finite gradient fails only when it prevents
  continued optimization and the established standard workflow cannot recover;
- added the owner-authorized restoration of the tracked 187-SMILES source and
  publication of selected route structures through the governed `hotpot.ds`
  dataset workflow; and
- changed the plan and Stage-17 index from review-pending to implementation in
  progress without changing the current Open Babel default.

Verification:

- the canonical source is fixed to Git commit `11c0e3d`, 187 physical lines,
  and SHA-256
  `e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`;
- the complete expected route-result matrix is 182 cases by five routes, while
  failed route-cases remain explicit failures rather than receiving substituted
  structures; and
- Markdown table consistency and `git diff --check` pass.

Unexpected findings: none in the contract-freeze step.

Adjustment:

- execute the remaining work in small checkpoints and revise the plan only
  after another complete reread if implementation evidence exposes a genuine
  cross-document conflict.

Git checkpoint: `abd1023 docs(plan17): freeze force-field validation contract`.

### Step 53 - Restore the tracked canonical extractant source

Status: complete

Planned:

- restore the exact 187-line proposal source without deriving it from a local
  dataset payload; and
- prevent silent canonicalization, reordering, or future content drift.

Performed:

- restored `molecules/extractant/extractants.smi` byte-for-byte from Git commit
  `11c0e3d`;
- retained all physical lines in their historical order without rewriting any
  SMILES; and
- added a source regression that reads the tracked bytes directly.

Verification:

- the file has exactly 187 physical lines;
- its SHA-256 is
  `e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`;
- the focused source regression passes; and
- `git diff --check` passes.

Unexpected findings: none.

Adjustment:

- use this tracked file as the semantic cohort builder's canonical default;
  the ignored historical dataset copy is no longer required for validation.

Git checkpoint: `c6d884d data(validation): restore canonical extractant source`.

### Step 54 - Make the tracked corpus the benchmark source of truth

Status: complete

Planned:

- remove the remaining benchmark dependency on an ignored local copy of the
  extractant SMILES while preserving the exact scientific bytes.

Performed:

- pointed both built-in Eu and Am extractant suites at the tracked
  `molecules/extractant/extractants.smi` asset;
- updated the benchmark contract test and maintained command examples; and
- updated the Open Babel extractant bridge fixture to use the same source.

Verification:

- the built-in suite metadata, 187-record loader, Am source reuse, and
  counterion-free case-134 checks report `4 passed`;
- the canonical source digest remains unchanged; and
- `git diff --check` passes.

Unexpected findings:

- legacy benchmark consumers still named the ignored governed-data copy even
  though the newly restored tracked asset has identical canonical content.

Adjustment:

- retain generated coordinate payloads under ignored `hotpot/dataset/data`,
  but use the tracked proposal source for all new cohort construction.

Git checkpoint: `4e6782d test(validation): use tracked extractant corpus`.

### Step 55 - Add the frozen-coordinate F2 replay provider

Status: complete

Planned:

- generate ETKDG coordinates before the complete force-field workflow;
- address the exact hydrogenated worker components without a graph-only
  collision between repeated ligands; and
- prove that F2 supplies unchanged rows rather than invoking RDKit inside the
  ligand-build worker.

Performed:

- added a validation-only, pickle-safe `FrozenCoordinateReplayAdapter`;
- froze proposals on the same hydrogenated, positional-ID worker graph used by
  the existing private provider seam;
- keyed each proposal by stable atom IDs, atom/bond semantics, dative direction,
  and the represented stereo payload;
- stored and rechecked an exact little-endian float64 coordinate SHA-256 before
  replay; and
- exposed a coordinate-free proposal manifest for per-case evidence.

Verification:

- four focused tests prove exact row replay, no input mutation, spawn-safe
  pickling, component mismatch rejection, and tampered-byte rejection;
- the focused test selection reports `4 passed`; and
- Python whitespace and `git diff --check` pass. Ruff is not installed in the
  active local environment, so the later configured regression environment
  must run the maintained lint selection.

Unexpected findings:

- repeated isomorphic ligand components cannot safely use a graph-only lookup;
  the worker's refreshed global atom IDs are needed to make proposal addressing
  unambiguous.

Adjustment:

- keep this provider in benchmark infrastructure. It does not add a public
  force-field backend/profile switch or a second ETKDG implementation.

Git checkpoint: `21c0035 test(validation): add frozen F2 replay provider`.

### Step 56 - Implement the corrected A01-A10 acceptance evaluator

Status: complete for the pure evidence contract

Planned:

- judge the selected structure independently from exploratory frames;
- fail unavailable gradients only when established recovery and continued
  optimization cannot be proved; and
- preserve non-finite observations in strict JSON without emitting NaN or
  infinity.

Performed:

- added immutable proposal, numerical-frame, recovery, quality, semantic, and
  stage-trace evidence records;
- added a pure evaluator returning one explicit decision for every A01-A10
  gate;
- required each unavailable-gradient event to identify an established named
  recovery, a later finite-gradient frame, and a later frame proving resumed
  optimization; and
- normalized non-finite numerical values to `null` plus finite/diagnostic facts.

Verification:

- 15 synthetic tests cover all nine required plan scenarios, masked terminal
  labels, unlisted recovery, hidden provider fallback, graph mutation, gate
  completeness, and strict JSON serialization; and
- the focused selection reports `15 passed`.

Unexpected finding:

- a terminal reason alone cannot establish or refute recovery; the evaluator
  must consume ordered frame and recovery evidence.

Adjustment:

- keep this evaluator pure and translate mutable Hotpot reports only in the
  route evidence writer.

Git checkpoint:
`6194214 test(validation): enforce forcefield acceptance contract`.

### Step 57 - Prepare governed benchmark-dataset publication

Status: implementation complete; formal publication awaits route evidence and
manual review

Planned:

- model all 182 by five route results without substituting structures into
  failed cells; and
- publish through the existing Bundle, assessment, certificate, and ingestion
  lifecycle rather than a parallel benchmark store.

Performed:

- added a strict 910-result mapper with stable case-major/route order;
- copied only each passing route's own selected structure and retained failed
  results without a coordinate asset;
- added owner-`User` authorization, result/source/schema/summary control assets,
  non-finite JSON normalization, and an inventory-candidate generator;
- added separate prepare and confirmed publish operations; and
- documented the dataset and added its tracked result schema.

Verification:

- the dataset/documentation selection reports `15 passed` in the configured
  environment;
- temporary-store integration proves synchronize, certificate, and
  `ingest(include_incomplete=True)` behavior; and
- no formal store, final inventory manifest, or scientific payload was written.

Unexpected finding:

- failed route-cases would be silently omitted by default ingestion; preserving
  the complete matrix requires explicit incomplete-case ingestion and reason.

Adjustment:

- defer certification and ingestion until the complete manual review is part of
  the certificate-bound evidence.

Git checkpoint: `115ef2c feat(dataset): prepare Plan 17 benchmark publication`.

### Step 58 - Implement the current semantic 187/182 cohort

Status: complete for implementation and focused fixtures; formal CBond
materialization pending

Planned:

- execute current CBond exactly once per tracked source line;
- serialize a deterministic, coordinate-free graph and represented stereo
  identity; and
- reconstruct every route from that identity without re-running CBond.

Performed:

- added a canonical-source/hash check and fixed Am/standard-settings cohort
  builder with an injectable CBond seam;
- retained atom rows/IDs/elements/charges/aromaticity, complete bond semantics,
  dative direction, Level-1 stereo/transport, metal/donor rows, seeds, and
  settings;
- enforced the exact 182-positive/five-negative partition and semantic graph
  digests;
- added deterministic JSON read/write validation and direct Hotpot molecule
  reconstruction; and
- derived a C1/C2 generation projection that explicitly converts only
  coordination edges to donor-to-metal dative form while retaining the source
  bonds and canonical business-graph digest.

Verification:

- four focused tests prove Git-tracked source identity, exactly one CBond call
  per case, deterministic path-free bytes, JSON round-trip, stereo transport,
  Hotpot graph reconstruction, and the generation-only dative transform;
- the combined cohort/provider/route/evaluator selection reports `29 passed`;
  and
- Python compilation and `git diff --check` pass.

Unexpected finding:

- current CBond emits ordinary metal-ligand bonds, whereas native whole-complex
  ETKDG requires explicit directed dative bonds. Passing the unmodified native
  projection would therefore reject valid cohort graphs.

Adjustment:

- keep the dative conversion confined to the C1/C2 generation projection and
  copy only generated coordinate rows onto the untouched canonical Hotpot graph
  before force-field optimization.

Git checkpoint: `ecaff12 test(validation): build semantic 187 cohort`.

### Step 59 - Wire the five validation routes to shared force-field policy

Status: complete for route execution primitives

Planned:

- run O0/F1/F2 through one private complete-complex workflow with only the
  coordinate provider changed;
- pass C1/C2 whole-complex ETKDG coordinates directly into the existing
  complex-entry optimization policy; and
- retain proposal and stage timing/identity evidence.

Performed:

- added O0/F1/F2 dispatch using `OpenBabelConformerAdapter`,
  `RDKitConformerAdapter`, or the frozen replay adapter at the existing sampler
  seam;
- made all provider routes share the same force field, budgets, hydrogen policy,
  trajectory detail, seed policy, and stage sequence;
- added C1 stock and C2 placement-policy-bounded whole-complex generation;
- derived C2 metal-donor bounds from the current native covalent-radius target
  and fixed 0.70/1.50 placement ratios;
- copied generated rows onto the untouched Hotpot graph and invoked
  `optimize_complex(add_hydrogens=False)` from `COMPLEX_UNTANGLING`; and
- recorded proposal, force-field, and total route durations separately.

Verification:

- ten focused provider/route tests pass;
- O0/F1/F2 tests prove one shared workflow/options set and the F2 replay type;
- C1/C2 tests prove dative generation input, exact rows at force-field entry,
  no second hydrogen/ligand build, correct bounds, and no source topology
  mutation; and
- Python compilation and `git diff --check` pass.

Unexpected finding:

- the C1 stock embedding graph also needs the explicit dative representation to
  keep pinned RDKit's multi-fragment path from bypassing the complete-complex
  experiment, even though C1 applies no custom distance bounds.

Adjustment:

- treat F2 and C1/C2 as validation controls only; no public backend/profile or
  alternate production workflow is introduced.

Git checkpoint: `0d170d8 test(validation): wire five forcefield routes`.

### Step 60 - Bind benchmark sample identity to semantic graph identity

Status: complete

Planned:

- prevent two route records with the same source-line/route label but different
  reconstructed chemical graphs from being treated as the same scientific
  sample.

Performed:

- added the canonical semantic graph SHA-256 to Bundle `core_info` and the
  persisted identity schema; and
- documented explicitly that coordinates and energies remain outside sample
  identity.

Verification:

- the dataset publisher and documentation selection reports `15 passed`;
- the temporary Bundle synchronize/certify/ingest lifecycle remains green; and
- Python compilation and `git diff --check` pass.

Unexpected findings: none.

Adjustment:

- retain source line, route, and semantic graph digest as the complete sample
  identity tuple; use coordinate hashes only for immutable asset identity.

Git checkpoint: `75049ca fix(dataset): bind benchmark semantic identity`.

### Step 61 - Translate production route facts into A01-A10 evidence

Status: complete for implementation and focused fixtures; formal 182-case
evidence pending

Planned:

- evaluate the selected force-field structure rather than treating every
  exploratory numerical excursion as a failure;
- require explicit proof that any unavailable-gradient event was recovered and
  optimization continued; and
- retain every ETKDG coordinate proposal before force-field work can mutate or
  fail.

Performed:

- added a report/trajectory translator that binds the real selected main
  trajectory frame to the committed molecule and final energy/gradient report;
- made finite-energy and finite-gradient flags authoritative and rejected
  mismatches between the selected archive frame and final report;
- derived unavailable-gradient recovery only from a same-branch, same-stage
  production perturbation/rollback event followed by finite gradients and
  resumed optimization, rather than accepting caller assertions;
- derived A10 stage outcomes from observed proposal, build, trajectory,
  optimization, and acceptance facts instead of copying the expected stage
  list into successful records;
- matched O0/F1/F2 proposal rows to the complete expected ligand-component set,
  and bound F2 rows to each component SHA-256 rather than an unordered digest
  collection;
- normalized graph comparison so ordinary bond row/order/orientation is not a
  false semantic change while atom order, dative direction, formal charge, and
  represented stereo remain protected; and
- added failure-atomic, full-coordinate F2 and C1/C2 proposal artifacts written
  before downstream force-field execution, including canonical and file hashes.

Verification:

- 66 focused cohort, provider, route, evidence, runner, gallery, dataset, and
  pure-acceptance tests pass;
- negative fixtures prove that a missing/mismatched selected frame, missing
  provider component, absent coordination stage, unproved gradient recovery,
  semantic mutation, and changed F2 coordinate all fail their intended gates;
- failure fixtures prove F2 and C2 proposal bytes remain available when the
  force-field call raises; and
- a real spawned-worker O0 smoke route passes all A01-A10 gates, including the
  worker atom-ID renumbering boundary; and
- Python compilation, Ruff checks/format, and `git diff --check` pass for the
  affected files.

Unexpected findings:

- the first evidence draft could synthesize a selected frame, mark expected
  stages complete without observations, and accept an asserted recovery with
  no archived action; and
- retaining only proposal hashes was insufficient when a force-field failure
  occurred before a successful route result existed.

Adjustment:

- make trajectory/report consistency and pre-forcefield proposal persistence
  hard evidence boundaries before starting the formal route matrix.

Git checkpoint: `537784b test(validation): bind forcefield evidence to artifacts`.

### Step 62 - Bind the formal native and Python runtimes

Status: complete for the formal-start gate; ai-14 execution pending

Planned:

- prevent a formal matrix from combining the pinned native wrapper with a
  different Python RDKit distribution;
- bind the native wrapper to the approved RDKit source commit, C API, and lock;
  and
- preserve RDKit, Open Babel, extension, and lock identities in every formal
  run manifest.

Performed:

- added both the imported RDKit core version and installed distribution version
  to the collected runtime identity;
- bound the native wrapper to the approved RDKit commit, C API version, and
  exact dependency-lock contents;
- recorded SHA-256 identities for both native extension binaries; and
- required the compiled, loaded-native, and Python-reported Open Babel versions
  to agree before formal execution.

Verification:

- nine focused validation-runner tests pass, including independent version,
  source-commit, C-API, lock, and Open Babel mismatch failures;
- focused Ruff import, syntax, and formatting checks pass; and
- `git diff --check` passes.

Unexpected finding:

- ai-14's existing `hp` environment contains Python RDKit `2026.03.6`, while
  the verified Hotpot native wheel embeds RDKit `2026.09.1`; the former runtime
  would therefore have mixed two RDKit releases despite satisfying the old
  native-only gate.

Adjustment:

- assemble the validation runtime under the permitted `~/mca_test` tree with
  the pinned Python RDKit wheel and keep the existing Conda environment
  unmodified.

Git checkpoint: `fa828b5 test(validation): bind formal runtime identity`.

### Step 63 - Normalize invalid source atom identifiers at cohort freeze

Status: implementation complete; the schema-v1 formal artifact was later
superseded by Step 66

Planned:

- freeze one deterministic, reconstructable graph identity for each CBond
  result without changing its chemical graph.

Performed:

- retained existing atom IDs when they are unique and non-negative; and
- mapped invalid/default identifiers to contiguous atom-table rows before
  serializing both the graph and its coordination metadata, matching Hotpot's
  established chemical-boundary identity policy; and
- kept the general graph serializer literal, so a later force-field identity
  collision remains visible to A09 instead of being silently normalized.

Verification:

- 40 focused cohort, evidence, runner, gallery, publication, and dataset tests
  pass;
- a new negative fixture proves duplicate default IDs are normalized
  consistently in graph atoms, the metal record, and reconstruction;
- the clean ai-14 reconstruction froze all 187 source records into the required
  182 CBond-positive and five negative cases (139, 182, 185, 186, and 187),
  with canonical SHA-256
  `c35638bbe2b4c71a9fee536c529522ad3e5f2d92b5e336e09ced87a7ce1f8642`;
- focused Ruff import, syntax, and formatting checks pass; and
- `git diff --check` passes.

Unexpected finding:

- the first formal ai-14 cohort build reached all 187 CBond inferences, then
  correctly refused to write the artifact because current `read_mol -> CBond`
  objects carried the Atom constructor's default ID `0` on every atom; the
  failed attempt produced no cohort file.

Adjustment:

- normalize only the technical atom-identity field at the serialization
  boundary. Atom order, elements, charges, bonds, donor rows, stereo, and CBond
  partition remain unchanged.

Git checkpoint: `682e801 fix(validation): normalize cohort atom identities`.

### Step 64 - Project fully materialized force-field hydrogens at the RDKit seam

Status: implementation, native integration, and diversified ai-14 pilot
complete; formal matrix pending

Planned:

- remove the common RDKit input failure without changing standalone rdWrapper
  semantics or the O0 workflow;
- preserve exact atom/bond row mapping and the Level-1 source ledger; and
- prove that ordinary partially explicit molecules retain their remaining
  implicit hydrogens outside the force-field seam.

Performed:

- added an explicit opt-in mode to the single `RDKitConformerAdapter`
  implementation for working copies whose hydrogens have already been fully
  materialized as graph atoms;
- projected only a private copy, set current logical/transport hydrogens to
  `(0, 0, true)`, and retained source hydrogen provenance unchanged;
- mechanically replaced transported `IMPLICIT_H` stereo carriers with their
  unrepresented graph-hydrogen rows and rejected missing or ambiguous mappings;
- enabled that mode only for the F1/F2 force-field providers and C1/C2
  hydrogenated whole-complex inputs; and
- left O0, the standalone native wrapper, the default adapter mode, and
  `add_hydrogens=False` inputs unchanged.

Verification:

- 43 focused tests pass, including the real compiled native extension;
- fully materialized methanol now embeds successfully with its exact six graph
  rows, while `[H][CH2]` in default mode retains two remaining implicit
  hydrogens;
- the ai-14 case-1 standard-parameter run passes A01-A10 for O0, F1, F2, C1,
  and C2;
- a diversified 12-case, five-route ai-14 pilot completed all 60 route-cases:
  O0/F1/F2/C1 each passed 12/12 and C2 passed 11/12; and
- C2 case 81 failed explicitly before force-field entry because its fixed
  coordination bounds could not pass RDKit triangle smoothing. This is a
  route-hypothesis outcome, not a fallback or a materialized-hydrogen failure.

Unexpected findings:

- the initial tempting rule, “if a graph contains H then clear implicit H,” is
  chemically invalid: partially explicit inputs such as `[H][CH2]` legitimately
  contain both a graph H and remaining non-graph hydrogens; and
- C2's fixed pairwise coordination bounds are not feasible for every complex,
  even when stock C1 generation and all ligand-provider routes succeed.

Adjustment:

- encode full hydrogen materialization as an explicit force-field-seam
  precondition rather than inferring it from graph contents; and
- retain the C2 failure as required negative evidence in the formal 182-case
  experiment instead of introducing an unapproved fallback.

Git checkpoint: `9b31908 fix(validation): project materialized hydrogens at RDKit seam`.

### Step 65 - Seal the validation-to-dataset evidence chain

Status: complete

Planned:

- prevent a logically different run or selected structure from entering the
  governed dataset under a valid primary-route label; and
- make the publication Bundle self-contained and immutable with respect to all
  source validation evidence.

Performed:

- required five distinct primary run IDs plus one distinct F1-repeat ID;
- compared every F1 primary/repeat pass and A01-A10 gate decision and encoded
  any disagreement in the conversion manifest;
- bound every selected structure to the exact `selected.sdf` beside its own
  case `result.json` and rejected path escape or cross-case substitution;
- revalidated F2 and C1/C2 proposal contents, hashes, canonical identities,
  routes, and fixed sibling paths;
- revalidated trajectory tree hashes, file counts, byte counts, SDF policy,
  containment, and absence of symlinks;
- froze the complete run, cohort, source, gallery, manual-review, conversion,
  proposal, trajectory, and selected-structure evidence into the Bundle;
- emitted one aggregate evidence index as a certified Bundle control asset;
  and
- revalidated the copied evidence and prepared dataset structures before the
  atomic Bundle rename, closing the validation-to-copy race boundary.

Verification:

- 13 focused publication/dataset tests pass, including negative proposal,
  trajectory, cross-case selection, repeat-drift, and copy-time mutation cases;
- the real prepare -> synchronize -> certify -> ingest lifecycle passes on the
  complete fixture matrix;
- the combined Plan-17 validation, force-field seam, and compiled native test
  selection reports 173 passed; and
- focused Ruff E/F/I and `git diff --check` pass.

Unexpected findings:

- merely indexing the bytes copied at publication time could certify evidence
  that no longer matched the already validated runner declarations if source
  files changed between validation and copying; and
- proposal and trajectory artifacts were previously carried as evidence but
  not independently rehashed at the publication boundary.

Adjustment:

- treat Bundle preparation as a second complete evidence-verification boundary,
  not as a trusted file-copy step. Dataset ingest remains explicitly separate
  and still requires owner-authorized confirmation.

Git checkpoint: `8986654 fix(validation): seal publication evidence chain`.

### Step 66 - Preserve current implicit-hydrogen semantics in the cohort

Status: local correction and affected regression complete; clean ai-14 cohort
reconstruction and full formal rerun pending

Planned:

- investigate the seven common F1/F2 native generation failures labelled as
  aromatic kekulization errors in the first formal matrix;
- determine whether they are route-science failures or an evidence-transport
  defect before interpreting route coverage; and
- keep any superseded formal evidence rather than overwrite or selectively
  splice it into a corrected result matrix.

Performed:

- traced a representative failing `[nH]` source through fresh current CBond,
  schema-v1 serialization, reconstruction, force-field hydrogen materialization,
  and the compiled native RDKit boundary;
- proved that current CBond retained `implicit_hydrogens=1` on the two
  pyrrolic nitrogens after its polar-hydrogen removal policy, while the
  schema-v1 cohort omitted that current atom state and reconstructed both as
  zero;
- audited the source corpus and found 29/187 inputs with logical polar
  O-H/aromatic-N-H state before CBond, including no-CBond case 139. Therefore
  as many as 28/182 positive cases are potentially affected, although actual
  loss depends on donor working-copy recomputation; the seven kekulization
  failures are only the visible subset, not a safe rerun boundary;
- added `implicit_hydrogens` to every serialized atom and restored it when
  reconstructing the Hotpot graph;
- incremented both the semantic cohort and derived C1/C2 native-graph schemas
  from 1 to 2, so a schema-v1 artifact cannot be accepted accidentally;
- made the current implicit-hydrogen count part of both the per-case semantic
  graph digest and the aggregate cohort digest; and
- added compiled-native regression coverage proving that a cohort round trip
  retains two pyrrolic N-H sites, force-field materialization creates their
  graph H atoms, and native ETKDG succeeds.

Verification:

- 29 focused cohort and compiled-native tests pass;
- the full affected selection reports 181 passed;
- invalid missing, boolean, negative, floating-point, and string hydrogen
  counts are rejected;
- schema v1 is rejected, and changing only an implicit-hydrogen count changes
  both the semantic graph and cohort identities;
- focused Ruff E/F/I checks pass; and
- `git diff --check` passes.

Unexpected finding:

- the first complete formal matrix at `6071464` used cohort schema v1. Its
  routes all consumed the same serialized graph, but that graph was not a
  faithful current-CBond chemical state. Consequently the matrix is retained
  only as diagnostic execution evidence and cannot support final scientific
  route conclusions.

Adjustment:

- rebuild the entire 187/182 cohort under schema v2 in a new ai-14 study
  directory, then rerun all 182 O0/F1/F2/C1/C2 cases and all 182 F1-repeat
  cases. Do not overwrite the schema-v1 matrix, rerun only the seven visible
  failures, or combine schema-v1 and schema-v2 outcomes.

Git checkpoint: `b0aa6c5 fix(validation): preserve cohort implicit hydrogens`.

### Step 67 - Execute the authoritative schema-v2 formal matrix and gallery

Status: automated validation and exhaustive gallery generation complete;
attributable human review and governed dataset publication pending

Planned:

- build a clean exact-revision schema-v2 cohort rather than repairing the
  schema-v1 artifact in place;
- require one complete O0/F1/F2/C1/C2 matrix plus an independent complete F1
  repeat under the formal runtime gate; and
- render every primary route-case without substituting a rejected frame or a
  different route's structure for a failed result.

Performed:

- installed the exact clean `b0aa6c5` checkout at
  `ai-14:~/mca_test/hotpot-plan17-b0aa6c5` and passed the formal identity gate
  against Hotpot 0.5.4.0, native/Python RDKit 2026.09.1, Open Babel 3.2.1,
  native RDKit commit `fece8caa...`, C API v1, and the locked extension hashes;
- rebuilt all 187 current-CBond cases into
  `~/mca_study/plan17-b0aa6c5/cohort.json`, retaining the exact 182-positive
  and five-negative partition and producing canonical scientific digest
  `8f68e8b74d51eac07f28846cb901e92b94e5697aa0495733dbc44e03ae198102`;
- round-tripped every positive graph through Hotpot reconstruction and checked
  every derived C1/C2 atom document with zero mismatches;
- ran case 26 through O0/F1/F2/C1/C2 first; all five routes passed A01-A10,
  proving that the repaired pyrrolic N-H state reached the real native path;
- executed the full 1,092-result formal matrix with 16 spawned workers in
  506.107 wall seconds; and
- created five exhaustive PyMOL contact sheets plus all 910 route-case cells:
  869 selected structures, 41 scientific-failure placeholders, and zero render
  errors. The generated review template remains correctly marked
  `pending_human_review` with 0/910 rows reviewed.

Automated result matrix:

| Route | Passed | Automated failed | Execution failed | Principal failures |
| --- | ---: | ---: | ---: | --- |
| O0 | 180/182 | 2 | 0 | A08: 24, 96 |
| F1 | 180/182 | 1 | 1 | no conformer: 10; A08: 29 |
| F2 | 180/182 | 1 | 1 | no conformer: 10; A08: 29 |
| C1 | 172/182 | 2 | 8 | eight no-conformer cases; A08: 21; unrecovered case 70 |
| C2 | 157/182 | 4 | 21 | five no-conformer; 16 inconsistent-bound; A08: 21, 54, 63; unrecovered case 70 |
| F1 repeat | 180/182 | 1 | 1 | exactly the same case/gate classifications as F1 |

Verification:

- the formal summary is complete with 1,092 observed of 1,092 expected,
  no missing, duplicate, or unexpected results;
- the F1 primary/repeat acceptance and complete A01-A10 signatures agree for
  all 182 cases;
- all 1,049 admitted structures across the six runs have finite selected
  coordinates, energy, RMS gradient, and maximum gradient, a matching selected
  archive frame, a passing standard gate, and a selected-structure asset;
- the only unavailable-gradient evidence is C1/C2 case 70. No established
  recovery restored a finite direction, so both routes correctly fail it;
- the seven former schema-v1 `[nH]` failures (26, 35, 36, 98, 153, 154, and
  180) pass all five primary routes, and no matrix result reports a
  kekulization, aromaticity, or hydrogen-representation error;
- gallery manifest SHA-256 is
  `1eebd77a35849da2bb65502d85f30c94a1eb668290857e4ec1cafd032451055a`;
  every declared image/contact-sheet digest validates; and
- a technical contact-sheet pre-screen found no immediately obvious failed
  rendering or grossly disconnected accepted structure, but it is not the
  attributable human review required by the publication contract.

Unexpected findings:

- every one of the 182 fresh positive CBond graphs contains at least one
  nonzero current implicit-hydrogen count. Nonpolar graph hydrogens often
  coexist with those source/current counts and are not necessarily missing
  downstream, but schema v1 failed to preserve the literal atom state for the
  entire cohort, reinforcing the full-rerun decision;
- after fixing that evidence defect, F1 and F2 converge on the same two real
  failures and F1 repeat is classification-stable; and
- current C2 fixed bounds are triangle-inconsistent for 16 cases. This is a
  negative result for that versioned hypothesis, not authority to add a
  fallback or loosen bounds inside the frozen experiment.

Adjustment:

- retain Open Babel as the default complex provider because F1 achieved
  180/182 rather than the owner-required 182/182; retain `rdWrappers`, the
  provider seam, and the F1/F2 experimental components;
- retain C1/C2 as measured research components without promoting either
  whole-complex hypothesis; and
- do not complete the review artifact or publish the owner-authorized 910
  dataset records until a human reviewer has examined the five galleries and
  supplied attributable completion evidence. On completion, the current
  automated matrix can contribute at most 869 selected coordinate assets;
  all 41 failed route-cases remain explicit records without substituted
  coordinates.
