# Plan 17 native RDKit wrapper implementation and validation report

Report date: 2026-10-23

Current branch: perf/mca-resource-efficient-workflow

Original implementation baseline:
9640d906a9d13e969152d0f3a4d92a3a16b9e0ce

Authoritative automated-validation source checkpoint:
b0aa6c5224a97469359c53448ba48d24d8a46112

Normative sources:

- rdkit_wrapper_implementation_plan.md;
- forcefield_component_validation_plan.md; and
- implementation_log.md.

## 1. Executive status

Plan 17 has delivered the standalone native RDKit conformer component, its
Python and C/C++ interfaces, the consumer-neutral molecule transport, the
private force-field conformer-provider seam, the seven-gate Hotpot
stereochemistry implementation, and the validation-only O0/F1/F2/C1/C2 route
matrix.

The corrected schema-v2 formal validation is complete:

- the canonical source contains 187 physical ligand records;
- current CBond produces 182 positive complexes and the five expected
  no-CBond cases;
- O0, F1, F2, C1, C2, and the complete F1 repeat produced 1092/1092 result
  records;
- every admitted selected structure has finite coordinates, energy, RMS
  gradient, and maximum gradient and passes the standard terminal geometry
  gate; and
- the F1 primary and repeat classifications agree for all 182 cases.

F1 passed 180/182 cases. The fixed owner-approved threshold for replacing O0
is 182/182. Native rdWrappers, the injectable RDKit adapter, and the private
provider seam are retained, but Open Babel remains the composed complex
default. No public backend/profile selector was added.

The five primary-route acceptance galleries are also complete: 869 admitted
selected structures were rendered, 41 failed route-cases received explicit
scientific placeholders, and there were no renderer failures. A supplementary
failure-diagnostic gallery now covers all 41 failures without changing their
status: 10 have nondegenerate rejected complex frames, F1 case 10 has only a
174-atom all-zero collapsed attempt, and 30 have no route-generated coordinate
frame and therefore show a labeled input graph. The generated manual-review
artifact remains pending. It must not be converted into a completed,
attributable human review by automation. Consequently the owner-authorized
910-record dataset conversion and Bundle publication have not yet run.

This checkpoint is not the final release checkpoint because the exhaustive
attributable human gallery review, followed by formal-data dataset conversion
and publication, remains outstanding. The conditional CPython 3.10-3.14 wheel
rerun required by the later adapter change has now passed at the authoritative
source checkpoint.

## 2. Final architecture and implementation route

The standalone component is:

    Hotpot Molecule
      -> field-selective NativeMoleculeBuffers
      -> Python _rd_native binding
      -> versioned C ABI v1
      -> one private C++20 RDKit 2026.09.1 ETKDG core
      -> owned status and (K, N, 3) coordinates

    External C/C++ caller
      -> C ABI v1 or header-only C++17 facade
      -> the same C ABI and private ETKDG core

The ligand-provider validation routes are:

    one schema-v2 semantic Hotpot graph
      -> O0: OpenBabelConformerAdapter
      -> F1: RDKitConformerAdapter
      -> F2: frozen ETKDG coordinate replay
      -> identical ligand gate, topology/recovery, placement,
         coordination restoration, complex optimization, and acceptance

The whole-complex research routes are:

    one schema-v2 semantic Hotpot graph
      -> C1: stock whole-complex ETKDG
      -> C2: fixed coordination-bounded whole-complex ETKDG
      -> generated rows copied onto the untouched canonical Hotpot graph
      -> existing complete-complex entry gate, topology/recovery,
         Open Babel optimization, and terminal acceptance

C1 and C2 originally stopped after standalone coordinate generation in the
owner-approved implementation plan. The later owner-approved force-field
validation plan explicitly required both routes to continue into force-field
optimization on all 182 cases. That continuation is benchmark infrastructure,
not a production workflow cutover.

There remains one ETKDG sampler implementation. The Python binding and installed
C++ SDK reuse the same private core and C API. Native sampling and force-field
use do not construct or translate through a Python rdkit.Chem.Mol.

The force-field hydrogen projection is deliberately narrow:

- default standalone RDKitConformerAdapter behavior remains unchanged;
- F1/F2 and C1/C2 explicitly opt into a fully-materialized-H working-copy
  projection;
- only a private copy is normalized for native construction;
- current implicit/bracket hydrogen counts are cleared only after graph
  hydrogens have been materialized;
- the source hydrogen ledger remains unchanged; and
- transported implicit-H stereo carriers are mapped to their exact graph-H
  rows or rejected as ambiguous.

This prevents native double counting without treating the mere presence of one
explicit H atom as proof that the whole molecule is fully materialized.

## 3. Public and private API result

### 3.1 Public Python rdWrappers

The public component contract is:

```text
GenerationStatus
ETKDGOptions(
    *, conformer_count=1, seed=None,
    prune_rms_threshold=None, use_random_coordinates=False
)
CoordinationEmbeddingConstraints(
    coordination_pairs, metal_donor_bounds,
    donor_pair_indices=None, donor_pair_bounds=None
)
ConformerRecord
  - ordinal: int
  - engine_conformer_id: int
ConformerBatch
  - status: GenerationStatus
  - coordinates: float64[K, N, 3]
  - records: tuple[ConformerRecord, ...]
  - engine: str
  - engine_version: str
  - diagnostic: str | None
RuntimeInfo
  - hotpot_version: str
  - rdkit_version: str
  - native: bool
ETKDGSampler(options: ETKDGOptions | None = None)
  - generate(mol, *, constraints=None) -> ConformerBatch
generate_conformers(mol, *, options=None, constraints=None) -> ConformerBatch
runtime_info() -> RuntimeInfo
```

Generation is coordinate-only and nonmutating. It does not change topology,
input coordinates, or mol.conformers and does not run a Hotpot geometry gate,
perturbation, Open Babel minimization, or RDKit UFF/MMFF optimization.

### 3.2 Stable C ABI v1 and C++17 facade

The C ABI exposes fixed-width scalar and pointer/length DTOs, opaque sampler
and result ownership, generation/result accessors, and exact Hotpot/RDKit
runtime identity. No STL, Boost, RDKit type, C++ exception, or Python object
crosses the shared-library boundary.

```text
hotpot_rd_sampler_create(...)
hotpot_rd_sampler_destroy(...)
hotpot_rd_generate(...)
hotpot_rd_result_status(...)
hotpot_rd_result_coordinates(...)
hotpot_rd_result_records(...)
hotpot_rd_result_engine(...)
hotpot_rd_result_engine_version(...)
hotpot_rd_result_diagnostic(...)
hotpot_rd_result_destroy(...)
hotpot_rd_status_message(...)
hotpot_rd_runtime_abi_version()
hotpot_rd_runtime_hotpot_version()
hotpot_rd_runtime_rdkit_version()
hotpot_rd_runtime_rdkit_commit()
```

The principal versioned DTOs are `hotpot_rd_options_v1`,
`hotpot_rd_molecule_view_v1`, `hotpot_rd_constraints_view_v1`,
`hotpot_rd_stereo_record_v1`, `hotpot_rd_atom_transport_v1`,
`hotpot_rd_bond_transport_v1`, and `hotpot_rd_conformer_record_v1`. Result
pointers are borrowed from an opaque `hotpot_rd_result` and remain valid until
`hotpot_rd_result_destroy()`.

The installed header-only C++17 facade exposes ETKDGOptions,
CoordinationEmbeddingConstraintsView, ConformerGenerationResult, RuntimeInfo,
ETKDGSampler, generate_conformers(), and runtime_info(). ABI-0 and ABI-1
consumer builds passed against the same SDK library.

```text
hotpot::rdwrappers
  - enum class GenerationStatus
  - struct ETKDGOptions
  - struct CoordinationEmbeddingConstraintsView
  - struct ConformerRecord
  - struct ConformerGenerationResult
  - struct RuntimeInfo
  - class ETKDGSampler
      - ETKDGSampler(ETKDGOptions options = {})
      - generate(MoleculeBufferView, optional<constraints>) const
  - generate_conformers(MoleculeBufferView, options, optional<constraints>)
  - runtime_info()
```

The facade copies result-owned values into consumer-side STL objects. The dual
ABI result establishes facade/DSO boundary compatibility, not compatibility
with arbitrary old `libstdc++` runtimes.

### 3.3 Common molecule transport

NativeMoleculeBuffers and MoleculeBufferView are consumer-neutral and
field-selective. Static graph data is transported once and conformer
coordinates separately. Generic validation admits toolkit-neutral bond
semantics; Open Babel and RDKit compatibility policy remains local to each
adapter.

Python exposes `pack_molecule(mol, fields)` with independently selectable
topology, coordinate, Level-1 stereo, atom-transport, and bond-transport
fields. C++ exposes `ArrayView<T>`, `MoleculeBufferView`, `OwnedMoleculeData`,
the three stereo/atom/bond transport view-owner pairs, and
`ConformerBatchView`. Missing fields remain distinguishable from present but
empty fields.

This is the single Hotpot contiguous transport implementation. No second
RDKit-specific buffer system was introduced.

### 3.4 Private force-field seam

The force-field workflow depends on one private ConformerSampler contract with
Open Babel and RDKit adapters. The seam is injectable in tests and worker-safe,
but it is not a public registry or complete-workflow selector.

Open Babel remains the composition-root default. RDKit failure is explicit;
there is no hidden Open Babel fallback inside F1.

## 4. Stereochemistry result

Level 1 preserves ordered anchors/carriers, opaque source values, enhanced
group lineage, isotope/radical facts, separate current and source hydrogen
states, and ordered bond display direction through molecule lifecycle,
working-copy transactions, generic packing, and native generation.

```text
StereoAnchorKind, StereoCarrierKind, StereoGroupKind
StereoGroupRef, StereoCarrier, StereoAnchor, StereoRecord
HydrogenTransportState, AtomTransportRecord, BondTransportRecord
StereoPayload
get_stereo_payload(mol) -> StereoPayload
set_stereo_payload(mol, payload) -> None
```

Level 2 implements the seven planned gates:

```text
StereoType, StereoIssueCode
StereoValidationIssue, StereoValidationReport
StereoDerivedDescriptor, StereoAssignmentReport, StereoAlignmentReport
validate_stereochemistry(mol, *, coordinates=None)
assign_stereochemistry(mol, *, coordinates=None)
align_stereochemistry(reference, target, atom_mapping)
```

| Gate | Implemented result |
| --- | --- |
| 1 | Frozen TH/DB/AX/SP/TBP/SPY/OH descriptor and permutation algebra |
| 2 | Deterministic graph validation for declared TH/DB semantics |
| 3 | Explicit-mapping alignment and parity transformation |
| 4 | Coordinate assignment, degeneracy, and contradiction reporting |
| 5 | Ordered axial/atropisomeric and enhanced ABS/AND/OR semantics |
| 6 | Complete represented-graph projection, native RI equivalence, directly decidable derived names |
| 7 | Atomic SP/TBP/SPY/OH public cutover, represented shells, multicenter boundaries, and chelates |

The canonical 187 source importer still produces an empty represented stereo
payload for this corpus. Therefore dedicated fixture/golden suites validate
Level 1 and Level 2, while R0 does not establish source-SMILES chirality
preservation.

## 5. Pinned dependency and packaging result

One Hotpot release supports exactly native RDKit 2026.09.1 and Python package
rdkit 2026.9.1. Upgrades are owner-controlled and deliberately lazy. The
native offline closure locks RDKit, Boost, Eigen, better-enums, Catch2, and
RingDecomposerLib by exact source identity and SHA-256.

The private native implementation uses C++20. The installed public C/C++ API
and existing native modules remain C++17-consumable.

Two packaging products remain distinct:

- Python wheels absorb the private core into _rd_native and do not publish the
  C++ SDK; and
- the developer SDK installs the C ABI library, public headers, CMake package,
  notices, and licenses without private RDKit headers.

A prior clean local CPython 3.10-3.14 matrix from source checkpoint 37a08b3
passed build, repair, isolated installation, coexistence, payload, and ELF
audits. Because the packaged conformer adapter later changed, the complete
workflow-equivalent matrix was rerun from a clean detached b0aa6c5 checkout.

All five current-checkpoint wheels passed build, repair, isolated installation,
both import orders, native search/relevant-cycle imports, Open Babel and RDKit
coexistence, exact Python/native RDKit identities, and an ETKDG generation
smoke. The archive-content auditor, CRC scan, duplicate-member check, embedded
tag audit, and direct `_rd_native` ELF audit also passed.

| ABI | Wheel size (bytes) | SHA-256 |
| --- | ---: | --- |
| cp310 | 5,638,695 | 4f35c10ed1b9134cfc013398942afab645d0abf669c962b55d62df37f894859d |
| cp311 | 5,643,505 | f87d03e47e7059195ff06d3a94cece7a45179548a58e6a52a77bba0d55c3e4c2 |
| cp312 | 5,665,701 | 50099f8d7e6b3f56f7fdaa4a9e31b484e0579b4ac21395303a5d75b7d915af1e |
| cp313 | 5,666,048 | 18f1eb3ea95f87f3062a340420018d5d8e3d8c7bd67483fade4d2d946c61d5d4 |
| cp314 | 5,668,859 | 1d6d21f913a1d07087ed0dbd3ceeabc828e01aca131143d0756ddfa5864f7d80 |

Every archive contains 871 entries and 763 files with zero duplicate members,
exactly one `_rd_native` extension, and no wrapper source, header, CMake, SDK,
or locked-source contamination. Each extension is stripped x86-64 ELF, has no
RPATH/RUNPATH or dynamic RDKit/Boost/Conda dependency, exposes only the
`PyInit__rd_native` function, and needs only the same six system runtimes as
the earlier matrix. Its per-ABI build ID is unchanged from the earlier native
build, as expected because the conditional rerun was triggered by packaged
Python adapter code rather than an rdWrapper-native change.

The project continues to make no strict standalone auditwheel claim while the
pinned Open Babel wheel supplies libopenbabel.so.8.

## 6. Standalone R0 result

The canonical 187-ligand R0 result remains valid component evidence because
the schema-v2 correction affects the later CBond semantic-cohort
serialization, not standalone native generation.

| Requested conformers | Trials | Success | Partial | Returned conformers |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 187 | 187 | 0 | 187 |
| 8 | 187 | 186 | 1 | 1,492 |
| 32 | 187 | 186 | 1 | 5,970 |
| Total | 561 | 559 | 2 | 7,649 |

Both partial results are deterministic case-10 outcomes: 4/8 and 18/32
conformers. There were no unsupported or failed trials.

| R0 metric | Result |
| --- | ---: |
| Primary generation time | 78.004250637 s |
| Matrix trials per second | 7.191915766 |
| Returned conformers per second | 98.058758818 |
| Finite and atom-row-exact conformers | 7,649 / 7,649 |
| Atom-order failures | 0 |
| Topology failures | 0 |
| Level-1 transport failures | 0 |
| Input-mutation failures | 0 |
| Same-seed mismatch failures | 0 |

The Level-1 payload was empty for every R0 trial. The zero transport-failure
count is therefore an empty-payload nonmutation result, not evidence that the
source-SMILES stereo notation was imported.

## 7. Formal cohort correction

### 7.1 Why schema v1 was rejected

The first formal cohort and matrix used schema v1. That schema serialized
elements, charges, aromaticity, bonds, stereo payload, and coordination data
but omitted each atom's current implicit_hydrogens value.

Fresh CBond investigation proved the omission was chemically material:

- CBond can remove polar graph H while preserving the parent atom's current
  implicit-H count;
- schema-v1 reconstruction defaulted that count to zero;
- force-field hydrogen materialization therefore could not restore some
  current chemical states; and
- the most visible symptom was seven aromatic [nH] failures before ETKDG.

All 182 positive schema-v2 graphs contain at least one nonzero current
implicit-H value. The direct downstream effect is case-dependent because some
hydrogens already exist as graph atoms and donor working copies may recompute
state, but schema v1 did not faithfully preserve the current CBond atom state
for any of the 182 positive cases. Selectively rerunning seven visible failures
would therefore have been invalid.

The schema-v1 cohort canonical SHA-256 was
c35638bbe2b4c71a9fee536c529522ad3e5f2d92b5e336e09ced87a7ce1f8642.
The complete matrix at 6071464 is retained as diagnostic execution evidence
only. It is not combined with schema-v2 outcomes and supports no final
scientific route conclusion.

### 7.2 Schema-v2 contract and verification

Schema v2:

- serializes exact nonnegative integer implicit_hydrogens on every atom;
- restores it during Hotpot graph reconstruction;
- includes it in the semantic graph and aggregate cohort identities;
- increments both semantic-cohort and derived C1/C2 native-graph versions;
- rejects schema v1 rather than filling or inferring the missing state; and
- rejects missing, Boolean, negative, floating-point, and string counts.

The corrected cohort identities are:

| Artifact | Identity |
| --- | --- |
| Canonical 187 source SHA-256 | e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7 |
| Cohort schema | 2 |
| Cohort canonical SHA-256 | 8f68e8b74d51eac07f28846cb901e92b94e5697aa0495733dbc44e03ae198102 |
| cohort.json file SHA-256 | 2e6702e59e57b2b8a56653cf5442f99df07f58c97a18b59afbc01cfba1ca58be |
| CBond-positive count | 182 |
| no-CBond indices | 139, 182, 185, 186, 187 |

Every one of the 182 positive graphs round-tripped through
semantic_case_molecule() with no graph-field difference. Every C1/C2 native
atom projection also matched its source semantic atom table.

Case 26 then passed O0, F1, F2, C1, and C2 with all A01-A10 gates true. The
previously visible schema-v1 [nH] failures at cases 26, 35, 36, 98, 153, 154,
and 180 all passed every primary schema-v2 route. The 1092-result matrix
contains no kekulization, aromatic-representation, or hydrogen-transport
error.

## 8. Formal runtime and execution identity

The ai-14 formal start gate recorded:

| Identity | Value |
| --- | --- |
| Hotpot commit | b0aa6c5224a97469359c53448ba48d24d8a46112 |
| Git worktree | clean |
| Hotpot version | 0.5.4.0 |
| Python | CPython 3.13.15 |
| Native RDKit | 2026.09.1 |
| Python RDKit | 2026.09.1 |
| Python distribution | 2026.9.1 |
| RDKit source commit | fece8caa860bdf6c9c82bdb9253f22a0cf87150c |
| rdWrapper C API | 1 |
| Open Babel | 3.2.1 |
| Workers | 16, spawn |

The formal run produced:

- 1092 expected results;
- 1092 observed results;
- zero missing, duplicate, or unexpected results;
- manifest SHA-256
  51931b8928d5d27a40035bc8ad17ea3383b3a8e8692f401554c4363ecb97a1e8;
  and
- summary SHA-256
  1b90d1ec7519ea2e590d623a58fc69b2fe422b8e60543520964df8c657475490.

The schema-v1 directory was not overwritten. The authoritative schema-v2
study resides separately below ai-14:~/mca_study/plan17-b0aa6c5.

## 9. Formal business results

| Route | Automated pass | Automated failure | Execution failure | Selected structures |
| --- | ---: | ---: | ---: | ---: |
| O0 | 180 | 2 | 0 | 180 |
| F1 | 180 | 1 | 1 | 180 |
| F2 | 180 | 1 | 1 | 180 |
| C1 | 172 | 2 | 8 | 172 |
| C2 | 157 | 4 | 21 | 157 |
| F1 repeat | 180 | 1 | 1 | 180 |

### 9.1 O0, F1, and F2

O0 failures:

- case 24: finite selected state rejected by A08 for atom proximity and
  bond-geometry failures; and
- case 96: finite selected state rejected by A08 for short-bond and
  bond-length-ratio failures.

F1 and F2 have the same two failed cases:

- case 10: fixed-seed, single-proposal ETKDG returned no conformer; and
- case 29: downstream optimization returned a finite state that failed A08 for
  atom proximity, short-bond, and bond-length-ratio checks.

F2 independently generated and froze the ETKDG coordinates before replaying
them through the O0 downstream path. Its identical failed-case set strongly
localizes the remaining F1 outcomes to proposal/downstream compatibility
rather than the in-worker adapter seam.

The F1 repeat is complete:

- comparable cases: 182/182;
- primary/repeat business classification mismatches: 0; and
- primary/repeat A01-A10 signature mismatches: 0.

F1 therefore has stable 180/182 behavior under the recorded configuration,
but stability does not satisfy the 182/182 correctness threshold.

### 9.2 C1

C1 whole-complex ETKDG returned no conformer for:

    10, 22, 25, 28, 96, 101, 115, 117

Case 21 reached a finite selected state but failed A08 geometry. Case 70
encountered an unavailable gradient after topology blocking, had no successful
established recovery, and could not continue optimization. It correctly failed
A02, A04, A05, A06, A08, and A10.

### 9.3 C2

C2 whole-complex ETKDG returned no conformer for:

    10, 28, 96, 101, 115

The fixed coordination-bounds matrix failed triangle smoothing for:

    22, 25, 64, 66, 67, 71, 72, 74,
    75, 79, 80, 81, 103, 104, 105, 109

Cases 21, 54, and 63 returned finite selected states but failed A08 geometry.
Case 70 had the same unrecovered unavailable-gradient outcome as C1.

The triangle-smoothing failures reject this fixed-bounds C2 hypothesis. They do
not prove that the chemical complexes cannot exist, and they must not be hidden
by a fallback to C1.

### 9.4 Complete failure-frame audit

The 41 primary failed route-cases have now been audited individually:

| Failure evidence class | Count |
| --- | ---: |
| Nondegenerate rejected complex 3D frame | 10 |
| Recorded but degenerate attempt only | 1 |
| No recorded route coordinate frame | 30 |

The ten rejected complex frames are O0 cases 24 and 96; F1/F2 case 29; C1
cases 21 and 70; and C2 cases 21, 54, 63, and 70. F1 case 10 archived two
ligand-build events, but both reference one 174-atom all-zero coordinate
revision with zero coordinate extent. It is a recorded failure state, not a
valid 3D conformer. The remaining 30 failures occurred before a coordinate
frame existed.

The complete 41-row inventory, selected numerical values, diagnostic-image
roles, and route efficiency analysis are recorded in
`formal_failure_analysis.md`. The 15 generic zero-conformer results cannot be
assigned a more specific RDKit distance-geometry subtype from the frozen
evidence because the native wrapper did not transport RDKit's internal
embedding-failure counters. The exact supported reason is therefore
fixed-seed ligand or whole-complex ETKDG returning zero conformers; no chemical
root cause is inferred from visual appearance.

## 10. Numerical acceptance audit

The primary matrix admits 869 selected structures. Including the F1 repeat,
1049 selected structures were written.

All 1049 selected structures have:

- finite coordinates;
- finite selected best_energy;
- finite selected RMS and maximum gradients;
- no selected-frame explosion;
- a passing standard geometry report;
- selected-frame/report consistency; and
- an exact selected-structure asset.

There is no unavailable-gradient event in an admitted structure. The only
unavailable-gradient evidence occurs in C1/C2 case 70. Neither route contains
a successful standard recovery followed by resumed optimization, so both
results correctly remain hard failures.

The admitted-value ranges are evidence of numerical availability, not
convergence or quantitative force-field accuracy:

| Run | Selected count | Energy min | Energy max | Max RMS gradient | Max gradient |
| --- | ---: | ---: | ---: | ---: | ---: |
| O0 | 180 | 62.853 | 14,610.968 | 28,024.798 | 171,047.964 |
| F1 | 180 | 39.150 | 17,955.045 | 384.315 | 2,340.265 |
| F2 | 180 | 39.150 | 19,541.058 | 513.692 | 3,149.060 |
| C1 | 172 | 31.921 | 9,084.880 | 50.022 | 319.331 |
| C2 | 157 | 31.665 | 9,084.115 | 26.375 | 173.074 |
| F1 repeat | 180 | 39.150 | 16,945.409 | 393.076 | 2,435.799 |

The owner contract deliberately does not impose an absolute energy or gradient
ceiling once values are finite and the standard geometry gate passes. Some
selected structures exhausted their budgets and are not demonstrated local
minima. The supported description is therefore “Plan-17 accepted initial
structures,” not “converged physical ground states.”

## 11. Timing and resource report

The formal matrix wall interval from manifest creation to summary generation
was 506.107 seconds. Per-case elapsed sums are not wall time; they include work
executed concurrently by 16 spawned workers.

| Run | p50 s | p95 s | max s | elapsed sum s | parent-worker user CPU s | Max observed process high-water RSS KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| O0 | 8.027 | 19.810 | 44.125 | 1,755.171 | 925.531 | 227,148 |
| F1 | 7.474 | 19.213 | 36.617 | 1,549.921 | 760.927 | 227,148 |
| F2 | 7.458 | 19.295 | 36.828 | 1,550.421 | 781.470 | 231,164 |
| C1 | 2.486 | 9.919 | 36.097 | 691.132 | 690.533 | 236,616 |
| C2 | 2.239 | 8.927 | 36.680 | 610.513 | 609.867 | 236,616 |
| F1 repeat | 7.403 | 18.451 | 36.813 | 1,546.564 | 758.739 | 241,032 |

Overall:

- per-case elapsed sum: 7,703.722 seconds;
- parent-worker user CPU sum: 4,527.066 seconds;
- parent-worker system CPU sum: 2.579 seconds; and
- maximum observed process high-water mark: 241,032 KiB.

F1's elapsed sum is approximately 11.7 percent below O0 in this run. That
observation does not override the case-10 and case-29 correctness failures.
It also must not be presented as an isolated ETKDG speedup: F1 sampling occurs
inside the ligand worker and is included in ligand-build timing, whereas F2
records its frozen proposal separately.

C1/C2 timing is not a production throughput comparison. Those routes enter at
a different stage and several cases terminate before force-field optimization.

The attempt-denominator efficiency view is:

| Route | Passed | Mean s/case | s/accepted | Elapsed/O0 | Observed completion span s | Attempted cases/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| O0 | 180/182 | 9.644 | 9.751 | 1.000 | 121.898 | 1.493 |
| F1 | 180/182 | 8.516 | 8.611 | 0.883 | 112.151 | 1.623 |
| F2 | 180/182 | 8.519 | 8.613 | 0.883 | 114.142 | 1.595 |
| C1 | 172/182 | 3.797 | 4.018 | 0.394 | 64.023 | 2.843 |
| C2 | 157/182 | 3.354 | 3.889 | 0.348 | 55.625 | 3.272 |

F1's elapsed sum and mean are 11.694 percent below O0; F2 is 11.666 percent
below O0. The O0-to-F1 reduction is 205.250 seconds, of which 163.146 seconds
is in optimization and 38.265 seconds in ligand build. F1 sampling itself is
inside the ligand worker and has no separate clock, so this is a complete-route
result rather than an isolated ETKDG speed claim.

The reconstructed completion spans overlap at route boundaries in the shared
16-worker pool and are not additive. The exact complete-matrix result remains
1,092 results in 506.107 wall seconds, or 2.157 attempted route-cases/s. Its
1,049 admitted structures correspond to 2.073 admitted structures/s.

The recorded CPU column is `RUSAGE_SELF` for each matrix worker. It excludes
the spawned ligand-worker CPU used by O0/F1/F2 and therefore is not total route
CPU consumption. The RSS column is likewise a process high-water observation,
not a per-case memory delta; later cases inherit earlier worker high-water
state. Neither field supports a route-level CPU or allocation-efficiency
ranking without child/cgroup accounting. C1/C2's lower elapsed values also
include their different entry stage and 10/25 early failures and must not be
reported as production speedups.

## 12. Gallery and human-review status

The isolated ai-14 gallery environment rendered five exhaustive route sheets
without modifying the existing hp environment.

| Route | Rendered selected | Scientific placeholder | Render error |
| --- | ---: | ---: | ---: |
| O0 | 180 | 2 | 0 |
| F1 | 180 | 2 | 0 |
| F2 | 180 | 2 | 0 |
| C1 | 172 | 10 | 0 |
| C2 | 157 | 25 | 0 |
| Total | 869 | 41 | 0 |

The acceptance gallery deliberately exposes no rejected or substitute
structure. Each failed result receives a labeled placeholder, and the renderer
never searches trajectory frames or another route to fill an accepted cell.
That protects the 869-asset admission boundary.

The supplementary failure-diagnostic gallery separately renders every failed
route-case. It uses the failed route's own rejected frame when one exists,
shows the collapsed F1 case-10 attempt beside its input graph, and otherwise
shows an input-graph panel labeled `NO ROUTE 3D FRAME`. These images are
diagnostic evidence only and cannot become selected dataset assets.

Gallery identities:

| Artifact | SHA-256 |
| --- | --- |
| gallery.json | 1eebd77a35849da2bb65502d85f30c94a1eb668290857e4ec1cafd032451055a |
| O0 contact sheet | d636dcdbdc7055092117fa0af56203e59cb26d775c002bfda8cdb5c210c7842b |
| F1 contact sheet | 4c63b7251fa75339aea6e1327b20c84d4b03f3947e32085bf6121da1b0838608 |
| F2 contact sheet | 46b452dc14aa5ae5dfe3dc3b9f39e01450bb5024323d456744274842cd027f73 |
| C1 contact sheet | 105342409aac2445be579f51772ab6bfcb8578078df45acd5ffd7052f29733aa |
| C2 contact sheet | 2b2793a54270abf024b9d1431d9aab33c16071e0adb88c87b3defad4904fe66a |
| Pending review template | cac4741f2c0e2dc2533b46a2c2dfed90e392c8bbea9191f0ef1717ccd7b64bff |

Failure-diagnostic identities:

| Artifact | SHA-256 |
| --- | --- |
| failure-analysis.json | feb795712f715ad3ecd8147a73fae845582caf6ba2ba69bacd40113498b2b224 |
| failure-analysis.md | cfc3189ad074635af71feb6b66a97f6e8ef08010c7588a95983e063271e58cfe |
| O0 failure sheet | 75a823b70986247c52f9284fd73186534b09eb67045791fe0b0e924f9f430c8d |
| F1 failure sheet | c8c66769941455994802c83bf9e322610975504eefea0a736be29eb5cd50000b |
| F2 failure sheet | 7855b6330131be415486a070678aefb51f5eba00feb08bd4dcc95e436407575d |
| C1 failure sheet | 517e814139bb5bf96f53fe68026ab00d505f58a639f6be38a87e2ea21213719a |
| C2 failure sheet | 23936b762486372284a21d25d838efe84e3e684e624b5c581f795b84f8ab01be |
| Combined failure sheet | 492efa16a9d044ba744ad7b6879e7255fac09e8296360453b0e88e894b798dc5 |

A non-authoritative technical contact-sheet scan did not add a visible
nonphysical flag. That scan is not an attributable human review. The template
correctly remains:

- status: pending_human_review;
- expected rows: 910;
- reviewed rows: 0; and
- reviewer/completion time: unset.

The final report must not interpret “no automated flag” or a technical scan as
human completion.

## 13. Dataset publication status

The owner authorized publication of the five primary route-case results as a
governed Hotpot CBond-to-force-field benchmark.

The publication matrix is fixed at:

- 182 cases by five primary routes;
- 910 result records;
- currently 869 admitted selected-coordinate assets; and
- 41 explicit failed results without a coordinate asset.

The F1 repeat is control evidence and cannot become a sixth publication route
or a coordinate source. If the human reviewer flags an otherwise admitted
structure, that result becomes a manual-review failure and the admitted asset
count decreases explicitly.

The validation-to-publication bridge already verifies run identity, cohort and
source identities, exact selected.sdf siblings, proposal artifacts, trajectory
tree hashes, F1 repeat controls, gallery/review hashes, and Bundle evidence
before atomic publication.

No formal-data conversion or Bundle has yet been created because the bridge
requires a completed attributable human review. That review is the only known
prerequisite blocker for the dataset path. After it exists, the remaining
work is to execute conversion, prepare and revalidate the Bundle, synchronize,
certify, ingest with the owner authorization, and record the resulting
inventory/certificate identities.

## 14. Original-goal final status

| ID | Objective | Implementation | Validation | Final conclusion |
| --- | --- | --- | --- | --- |
| G01 | Pin one Hotpot release to RDKit 2026.09.1 | Implemented | Validated | Native, Python metadata, lock, and runtime agree |
| G02 | Lock exact offline native closure | Implemented | Validated | Six archives and native build identity are locked |
| G03 | Public standalone Python rdWrappers | Implemented | Validated | API, native fixture, R0, and prior wheel tests pass |
| G04 | Independent C ABI and C++17 facade | Implemented | Validated | C11 plus ABI-0/ABI-1 C++17 consumers pass |
| G05 | Avoid Python RDKit translation in native use | Implemented | Validated | Native graph path and source fences pass |
| G06 | Generic field-selective molecule transport | Implemented | Validated | Python/C++ common views and consumer policies pass |
| G07 | Preserve public Open Babel component | Implemented | Validated | API retained; O0 supplies new population evidence |
| G08 | Coordinate-only nonmutating standalone sampling | Implemented | Validated | R0 and focused tests prove the boundary |
| G09 | Run canonical R0 at 1/8/32 | Implemented | Validated | 559 success, two partial, no failed/unsupported |
| G10 | Private backend-neutral provider seam | Implemented | Validated | Worker crossing and full routes exercise it |
| G11 | Replace only initial ligand provider | Implemented | Validated | O0/F1/F2 share downstream policy and evidence |
| G12 | Run complete 182-case F1 | Implemented | Validated | 180 pass, one A08 failure, one no-conformer failure |
| G13 | Owner-visible default decision | No cutover by design | Resolved | F1 is below 182/182; Open Babel remains default |
| G14 | Whole-complex C1 experiment | Implemented | Validated | 182 cases continued through force-field policy |
| G15 | Coordination-bounded C2 experiment | Implemented | Validated | 182 cases; fixed-bounds hypothesis rejected for 16 |
| G16 | Level-1 transport/lifecycle | Implemented | Fixture/golden validated | R0 corpus has empty represented payload |
| G17 | Level-2 Hotpot stereo semantics | Implemented | Fixture/golden validated | Seven gates complete; source importer remains separate |
| G18 | Preserve order/topology/charge/direction/stereo | Implemented | Validated for generated/executed results | All evaluated results pass A09; pre-generation failures are explicit |
| G19 | Retain obWrappers and rdWrappers | Implemented | Validated | Both public components remain |
| G20 | One public force-field interface | Implemented | Validated | No backend/profile; candidate_count remains reserved |
| G21 | Stage/resource reporting | Implemented | Validated | Formal route and stage metrics are complete |
| G22 | CPython 3.10-3.14 wheels | Implemented | Validated locally at current checkpoint | Five b0aa6c5 workflow-equivalent builds and audits pass |
| G23 | GitHub-hosted CI provenance | Not implemented | Not validated | Separate release task |
| G24 | Strict standalone auditwheel policy | Explicit non-requirement | Not claimed | Open Babel remains a controlled cross-wheel dependency |
| G25 | Recover byte-identical historical cohort | Obsolete | Not required | Current semantic cohort replaces the false prerequisite |
| G26 | Normalized current 187/182 cohort | Implemented | Validated | Schema v2 exact 182/5 partition and round trip |
| G27 | Exhaustive route gallery review | Accepted and failed diagnostics implemented | Partially validated | 910 acceptance cells plus 41 explicit failure diagnostics rendered; attributable human review pending |
| G28 | Selected-state/recovery acceptance contract | Implemented | Validated | Formal matrix and case 70 exercise hard-failure semantics |
| G29 | Frozen F2 replay control | Implemented | Validated | Complete 182-case primary route |

## 15. Validation-experiment final status

| ID | Experiment | Final status | Evidence or remaining action |
| --- | --- | --- | --- |
| V01 | Affected force-field regressions | Validated | Current affected selection: 181 passed |
| V02 | Exact runtime/source identity | Validated | Formal ai-14 gate passed at b0aa6c5 |
| V03 | Python/C/C++ standalone API/SDK | Validated | Existing native/SDK consumers remain green |
| V04 | Common-buffer/Open Babel parity | Validated | Existing parity and mixed-invalid tests retained |
| V05 | Canonical R0 187 x 1/8/32 | Validated | 559 success and two deterministic partial results |
| V06 | Current semantic cohort reconstruction | Validated | Schema v2, exact source hash and 182/5 partition |
| V07 | Fresh O0 complete workflow | Complete | 180/182 automated passes |
| V08 | F1 complete workflow | Complete, cutover failed | 180/182 is below the fixed threshold |
| V09 | O0/F1/F2 post-provider policy parity | Validated | Shared route implementation and evidence schema |
| V10 | Selected energy/gradient audit | Validated | Every admitted selected state is finite |
| V11 | Unavailable-gradient continuation audit | Validated | Only C1/C2 case 70; no recovery, correct hard failure |
| V12 | Intermediate numerical-excursion classification | Validated | Focused evaluator tests and formal evidence |
| V13 | Standard terminal acceptance | Complete | All five routes evaluated over 182 cases |
| V14 | Semantic graph/stereo parity | Validated for evaluated results | No A09 mutation in a completed execution |
| V15 | Five-route manual gallery review | Pending human review | Acceptance gallery 910/910 plus 41/41 failure diagnostics; review 0/910 |
| V16 | Timing/resource comparison | Complete | Per-route/stage and aggregate evidence recorded |
| V17 | F1 repeated-run consistency | Complete | 182/182 classifications and gates agree |
| V18 | C1 generation plus force field | Complete | 172 pass, 10 explicit failures |
| V19 | C2 generation plus force field | Complete | 157 pass, 25 explicit failures |
| V20 | Source-SMILES stereo preservation | Not run | Separate governed importer/sidecar scope |
| V21 | CPython 3.10-3.14 wheel matrix | Complete locally | Five b0aa6c5 builds, installed-wheel smokes, content and ELF audits pass |
| V22 | GitHub-hosted CI provenance | Not run | Separate release task |
| V23 | xTB/DFT/QM handoff | Excluded | Explicit owner scope exclusion |
| V24 | F2 precomputed replay | Complete | Full 182-case frozen-proposal route |

## 16. Test and verification summary

The schema-v2 correction and final formal path were verified by:

| Scope | Result |
| --- | --- |
| Focused cohort plus compiled-native tests | 29 passed |
| Full affected coordination/force-field/native selection | 181 passed |
| Focused Ruff E/F/I | Passed |
| git diff --check | Passed |
| ai-14 exact formal runtime gate | Passed |
| Schema-v2 case-26 five-route smoke | O0/F1/F2/C1/C2 all A01-A10 pass |
| Formal schema-v2 matrix | 1092/1092 complete |
| F1 repeat comparison | 182/182 consistent |
| Gallery asset validation | 910 cells, zero render errors |
| Failure analysis unit selection | 5 passed |
| Failure/gallery/publication regression | 18 passed |
| Failure diagnostic audit | 41/41 classified and rendered; 10 rejected 3D, one degenerate attempt, 30 input-graph-only |
| Validation-publication/dataset fixture selection | 13 passed |
| Current-checkpoint wheel matrix | CPython 3.10-3.14 all passed |
| Wheel content/integrity audit | Five passed; zero duplicate members |
| `_rd_native` wheel ELF audit | Five passed; no RPATH/RUNPATH or dynamic RDKit/Boost/Conda dependency |

The affected local selection used the real compiled native rdWrapper extension.
The formal ai-14 matrix exercised the exact Python 3.13/native runtime rather
than mutating the existing hp environment to add pytest.

Historical broader evidence remains recorded in the implementation log,
including the standalone component, stereo, SDK, native-build, import-order,
and prior wheel selections. It is not restated as current wheel closure where
the source checkpoint differs.

Outstanding verification is explicit:

- no attributable human review has been completed;
- no formal-data dataset conversion/ingest has run;
- GitHub-hosted CI and xTB/QM remain separate non-executed scopes.

## 17. Implementation checkpoint chronology

The complete per-step record remains in implementation_log.md. The major
dependency checkpoints are:

| Steps | Result | Representative checkpoint |
| --- | --- | --- |
| 0-23 | API, common transport, Level 1, native core, Python/C/C++ SDK | 42d7c4c |
| 24-29 | R0, private provider seam, C1/C2 generator, wheel integration | 531130c |
| 30-44 | Level-2 Gates 4-7 and atomic public cutover | a948279 |
| 45-51 | R0 closure, toolkit observations, local native/SDK and prior wheel matrix | 24cee99 |
| 52-58 | Corrected validation contract, source restoration, evaluator, F2, publisher, gallery, cohort | ecaff12 |
| 59-61 | Five routes, semantic dataset identity, artifact-grounded A01-A10 evidence | 537784b |
| 62 | Exact formal Python/native runtime binding | fa828b5 |
| 63 | Deterministic cohort atom-ID normalization | 682e801 |
| 64 | Force-field-only fully materialized-H projection | 9b31908 |
| 65 | Validation-to-dataset evidence-chain sealing | 8986654 |
| 66 | Schema-v2 current implicit-H preservation | b0aa6c5 |
| 67 | Authoritative schema-v2 matrix and exhaustive gallery | source b0aa6c5 |
| 68 | Current-checkpoint CPython 3.10-3.14 wheel rerun | source b0aa6c5 |
| 69 | Per-failure diagnostics, failure gallery, and efficiency qualification | source b0aa6c5; documentation/tooling only |

The first schema-v1 formal execution occurred after Step 64 and exposed the
cohort omission. Step 66 invalidated that cohort identity before final
interpretation and required the complete schema-v2 rebuild and rerun described
in this report.

## 18. Supported claims and explicit non-claims

### 18.1 Supported claims

- Native ETKDG is available through independently usable Python and C/C++
  interfaces without a Python RDKit graph translation.
- One pinned native RDKit 2026.09.1 core serves both interfaces.
- The canonical R0 matrix returned 7,649 finite atom-row-exact conformers with
  559 successful and two partial trials.
- Open Babel and rdWrappers remain independent public components.
- The private force-field seam changes only the coordinate provider for
  O0/F1/F2 and introduces no public workflow selector.
- The schema-v2 current semantic cohort is exact, deterministic, and
  reconstructable across 182 positive cases.
- The formal O0/F1/F2/C1/C2 plus F1-repeat matrix is complete and
  artifact-grounded.
- F1 has stable 180/182 behavior and is not eligible to replace O0.
- Every admitted selected structure satisfies the Plan-17 finite numerical and
  standard geometry contract.
- Every failed primary route-case now has a named failure stage/reason and a
  diagnostic view that distinguishes rejected 3D, degenerate attempt, and no
  route-generated coordinate frame.
- Level-1 structural transport and all seven declared Level-2 semantic gates
  are implemented and directly tested.

### 18.2 Explicit non-claims

- RDKit is not approved as the default complex coordinate provider.
- F1 is not 182/182 and its two failed cases are not hidden by fallback.
- ETKDG is not claimed to be an Am-complex force field.
- C1/C2 results do not establish universal whole-complex embedding accuracy.
- C2 fixed bounds are not generally feasible for this cohort.
- A finite accepted force-field state is not necessarily a converged local
  minimum, ground state, experimental conformer, or quantitatively accurate
  energy.
- The schema-v1 matrix is not formal scientific evidence.
- R0 does not prove source-SMILES stereo preservation.
- The manual physical-plausibility review is not complete.
- The owner-authorized dataset has not yet been converted or published.
- The local wheel matrix does not establish GitHub-hosted CI provenance or
  strict standalone auditwheel-manylinux compliance.
- xTB, DFT, and QM handoff are outside this validation.

## 19. Remaining closure sequence

The remaining actions are narrow and must preserve the evidence boundaries:

1. an attributable human reviewer completes all 910 acceptance-gallery rows,
   recording any visible nonphysical accepted structures, with the separate
   41-case failure-diagnostic sheets available for failed-case inspection;
2. validate that completed review against the immutable gallery manifest;
3. build the 910-record conversion, prepare/freeze the governed Bundle,
   synchronize, certify, and ingest it with the owner authorization;
4. record the resulting dataset inventory, selected-asset count, evidence
   index, and certificate identities;
5. issue the final Plan-17 release closure without changing the Open Babel
   default.

No recovery of a byte-identical historical manifest, no coordinate match to
9640d906, and no new xTB/QM experiment is required.
