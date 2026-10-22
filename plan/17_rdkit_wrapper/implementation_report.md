# Plan 17 native RDKit wrapper implementation and validation report

Report date: 2026-10-22

Branch: `feature/rdWrapper`

Implementation baseline: `9640d906a9d13e969152d0f3a4d92a3a16b9e0ce`

Evidence checkpoint: `1bac0a8`

Normative sources: `rdkit_wrapper_implementation_plan.md` and
`implementation_log.md`

## 1. Executive status

Plan 17 has delivered the standalone native RDKit conformer component, its
Python and C/C++ interfaces, the consumer-neutral molecule transport, the
private force-field provider seam, coordination-aware research embedding, and
the complete seven-gate Hotpot stereochemistry implementation. The canonical
187-ligand standalone `R0` matrix is complete and reproducible. A fresh local
native rebuild, SDK install, ABI audit, and broad functional regression are
also complete.

**Stage 17 is not release-complete.** The following release decisions still
have no formal evidence:

- `O0`/`F1` comparison on the frozen 182-complex cohort;
- formal `C1`/`C2` results on that same cohort;
- selection of RDKit as the default force-field coordinate provider; and
- the CPython 3.10-3.14 manylinux wheel build/install/audit matrix.

The frozen 182-case manifest and matching native graph sidecars are absent,
and there is no dedicated formal `O0`/`F1` runner. Consequently Open Babel
remains the composed force-field default. No public backend/profile selector
was added, and no existing force-field stage was removed, reordered, or
weakened.

## 2. Final architecture and implementation route

```text
Hotpot Molecule
  -> field-selective NativeMoleculeBuffers
  -> Python _rd_native binding
  -> versioned C ABI v1
  -> one private C++20 RDKit 2026.09.1 ETKDG core
  -> owned status + (K, N, 3) coordinates

External C/C++ caller
  -> C ABI v1 or header-only C++17 facade
  -> the same C ABI and private ETKDG core

Existing force-field workflow
  -> private ConformerSampler seam
     -> OpenBabelConformerAdapter (current default)
     -> RDKitConformerAdapter (implemented, injectable)
  -> unchanged ligand checks/recovery/refinement
  -> unchanged placement/restoration/complex optimization/acceptance
```

There is one sampler implementation. The Python binding and installed C++ SDK
reuse the same private core and C API; no Python `rdkit.Chem.Mol` is constructed
in the native sampling or force-field path. The standalone sampler returns
coordinates only and performs no Hotpot geometry gate, perturbation, candidate
selection, RDKit UFF/MMFF optimization, or Open Babel optimization.

Open Babel was retained as an independent component. Its Python packer and C++
adapter now consume the common field-selective representation, while Open
Babel-specific compatibility policy remains behind the adapter. The legacy
Open Babel public behavior and mixed-invalid first-error order are preserved.

## 3. Public APIs

### 3.1 Python `hotpot.cheminfo.rdWrappers`

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

Generation is nonmutating: it does not change topology, current coordinates,
or `mol.conformers`. Public values contain only Hotpot/NumPy types.

### 3.2 Stable C ABI v1

The ABI publishes fixed-width scalars and pointer/length views only. No STL,
Boost, RDKit type, exception, or Python object crosses the shared-library
boundary.

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
`hotpot_rd_bond_transport_v1`, and `hotpot_rd_conformer_record_v1`.
Result pointers are borrowed from an opaque `hotpot_rd_result` and remain valid
until `hotpot_rd_result_destroy()`.

### 3.3 Header-only C++17 facade

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

The facade copies result-owned values into consumer-side STL objects. Both
`_GLIBCXX_USE_CXX11_ABI=0` and `=1` consumers passed against the same installed
SDK library. This proves facade/DSO boundary compatibility, not compatibility
with arbitrary old `libstdc++` runtimes.

### 3.4 Hotpot stereochemistry API

Level 1 is exact structural transport:

```text
StereoAnchorKind, StereoCarrierKind, StereoGroupKind
StereoGroupRef, StereoCarrier, StereoAnchor, StereoRecord
HydrogenTransportState, AtomTransportRecord, BondTransportRecord
StereoPayload
get_stereo_payload(mol) -> StereoPayload
set_stereo_payload(mol, payload) -> None
```

Level 2 is Hotpot-owned semantic interpretation:

```text
StereoType, StereoIssueCode
StereoValidationIssue, StereoValidationReport
StereoDerivedDescriptor, StereoAssignmentReport, StereoAlignmentReport
validate_stereochemistry(mol, *, coordinates=None)
assign_stereochemistry(mol, *, coordinates=None)
align_stereochemistry(reference, target, atom_mapping)
```

## 4. Common native molecule view

The former Open-Babel-local transport was generalized without introducing a
second RDKit-specific contiguous-buffer system.

Python provides `NativeMoleculeBuffers` and `pack_molecule(mol, fields)` with
independently selectable topology, coordinate, Level-1 stereo, atom-transport,
and bond-transport fields. Missing fields remain distinguishable from present
but empty fields. Static topology is packed once; conformer coordinates travel
separately as a batch.

C++ provides `ArrayView<T>`, `MoleculeBufferView`, `OwnedMoleculeData`, the
three stereo/atom/bond transport views and owners, and `ConformerBatchView`.
Synchronous operations borrow data; long-lived sessions own it explicitly.
The force-field `ComplexSessionInput` remains a flat aggregate and exposes a
zero-copy ligand view, avoiding broad source/API churn.

Generic validation accepts toolkit-neutral structure, including dative and
zero-order bonds. Consumer policy remains local: Open Babel performs its own
lossless-representation checks, while RDKit performs its pinned graph,
sanitization, hydrogen, display-direction, and stereo checks.

## 5. Pinned dependency, build, and packaging result

One Hotpot release supports exactly RDKit `2026.09.1` / Python package
`rdkit==2026.9.1`. Upgrades are owner-directed and deliberately lazy. The
supported Python metadata range is 3.10-3.14 because the pinned distribution
does not provide the required CPython 3.9 artifact.

The complete offline source closure is SHA-locked: RDKit, Boost, Eigen,
better-enums, Catch2, and RingDecomposerLib. The private core is C++20; the
public C/C++ boundary and existing native modules remain C++17-compatible.
Configuration forbids upstream network fallback and moving revisions.

Two packaging products are intentionally separate:

- the Python wheel absorbs the private static core into `_rd_native` and does
  not install the C++ SDK; and
- the versioned developer SDK installs the C ABI library, C/C++ headers, CMake
  package metadata, notices, and licenses, without private RDKit headers.

The release profile is pinned to
`manylinux_2_28_x86_64@sha256:39df0042...b4ff82`, GCC/G++ 14.2.1,
binutils 2.41, CMake 4.4.3, and Ninja 1.13.2. The profile and workflow exist,
and its native target graph built in the pinned container, but the full
CPython 3.10-3.14 PEP-517 wheel/install/audit matrix has not run to completion.

## 6. Force-field component substitution status

The private backend-neutral provider seam and both Open Babel/RDKit adapters
are implemented. Only the initial ligand coordinate-provider call is
replaceable. Worker seed propagation, conformer count, deterministic minimum
ordinal selection, and every downstream ligand/complex stage remain unchanged.
RDKit failure is explicit; there is no hidden Open Babel fallback.

Focused seam and complex/force-field regression reported 177 passes. This is
implementation evidence, not the required formal `F1` business result.

Current production decision:

- Open Babel remains the default provider;
- `candidate_count` remains reserved;
- no public `backend=`, `sampler=`, profile, or complete-workflow switch was
  added;
- no existing geometry review, recovery, placement, optimization, or final
  acceptance stage was removed; and
- no default-provider cutover is authorized until formal `O0`/`F1` evidence
  is available and reviewed.

## 7. Standalone canonical R0 result

The canonical input contains 187 physical SMILES lines with SHA-256
`e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`.
The clean-tree rerun used commit
`90dd0923199c57b083cb8f8d1f952a86ba090b85`, Python 3.11.16, RDKit
2026.09.1, one outer worker, one ETKDG internal thread, fixed seeds, and
requested conformer counts `1 / 8 / 32`.

| Requested count | Trials | Success | Partial | Returned conformers |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 187 | 187 | 0 | 187 |
| 8 | 187 | 186 | 1 | 1,492 |
| 32 | 187 | 186 | 1 | 5,970 |
| **Total** | **561** | **559** | **2** | **7,649** |

Both partial results are deterministic physical case 10 outcomes: 4/8 and
18/32 conformers. There were no unsupported or failed trials.

| Metric | Exact Step 45 result |
| --- | ---: |
| Primary generation time | 78.004250637 s |
| Matrix trials per second | 7.191915766 |
| Returned conformers per second | 98.058758818 |
| Detached Level-2 observation time | 0.017287096 s |
| Finite/atom-row-exact returned conformers | 7,649 / 7,649 |
| Atom-order failures | 0 |
| Topology failures | 0 |
| Level-1 transport failures | 0 |
| Input-mutation failures | 0 |
| Same-seed mismatch failures | 0 |

Stereo interpretation is explicitly non-vacuous in the report schema. All
561 input trials had zero stereo records, atom transport, and bond transport,
including the source row containing SMILES `@` markers. Therefore all 7,649
returned conformers are `NOT_APPLICABLE` for Level 2; assessed, valid, invalid,
equivalent, and non-equivalent Level-2 counts are all zero. The zero Level-1
failure count proves empty-payload nonmutation only; it does **not** prove that
source SMILES chirality was imported and preserved.

## 8. Coordination-aware C1/C2 research status

`C1` (stock connected whole-complex coordinate-only ETKDG) and `C2` (the same
route with explicit directed donor-to-metal and optional donor-pair bounds)
are implemented as standalone research routes. `C2` validates existing
donor-to-metal dative edges, applies Angstrom bounds, and triangle-smooths the
custom matrix before embedding. Neither route calls Hotpot force fields,
Open Babel, RDKit UFF/MMFF, geometry gates, repair, selection, or perturbation.

The fresh current-native one-case Eu smoke rerun labels itself
`evidence_class=development-smoke` and `formal_claim_allowed=false`. All six
C1/C2 cells at requested counts `1 / 8 / 32` returned full `SUCCESS` batches
with deterministic coordinates and input nonmutation.

| Route | Count 1 max / strict | Count 8 max / strict | Count 32 max / strict |
| --- | --- | --- | --- |
| C1 | 0.3595 A / 0 of 1 | 0.4384 A / 2 of 8 | 0.6428 A / 7 of 32 |
| C2 | 0.001949029236 A / 0 of 1 | 0.001949029236 A / 7 of 8 | 0.001949029236 A / 29 of 32 |

Here "max" is the largest detached bound violation and "strict" is the
number of returned conformers with exactly zero violation. This confirms that
the C2 bounds strongly guide embedding but do not promise exact interval
satisfaction.

These numbers are **not** formal 182-case evidence. The smoke manifest is
permanently labelled non-formal, and no `C1`/`C2` route enters `hotpot.ff` in
this stage. A read-only search of `ai-14:~/mca_test` and `~/mca_study` by the
exact expected formal-manifest SHA found no matching artifact.

## 9. Stereochemistry completion

### 9.1 Level 1

Level 1 now preserves ordered anchors/carriers, opaque source values, enhanced
group lineage, isotope/radical facts, current versus source hydrogen ledgers,
and ordered bond display direction through molecule copy, components,
assembly, removal, hide/recover, working-copy commit/rollback, Python/native
packing, RDKit conversion, and native generation. `AXIS` stores the complete
ordered cumulene path. Structural lifecycle repairs also preserve hidden bonds
and independent atom-array ownership.

Level 1 remains transport only: it does not assert validity, stereogenicity,
coordinate consistency, equivalence, or CIP identity.

### 9.2 Level 2 seven-gate result

| Gate | Implemented result |
| --- | --- |
| 1 | Frozen TH/DB/AX/SP/TBP/SPY/OH descriptor and permutation algebra |
| 2 | Deterministic graph validation for declared TH/DB semantics |
| 3 | Explicit-mapping alignment and parity transformation |
| 4 | Hotpot-owned coordinate assignment, degeneracy, and contradiction reports |
| 5 | Ordered axial/atropisomeric and enhanced ABS/AND/OR group semantics |
| 6 | Complete represented-graph projection, native RI equivalence, direct-priority derived names |
| 7 | Atomic SP/TBP/SPY/OH public semantic cutover, including chelates |

Gate 7 uses the frozen Hotpot ideal-site conventions, determinant-positive
center-rooted fitting, 12-degree angular-RMS acceptance, a 2-degree same-family
uniqueness margin, and a 3-degree TBP/SPY margin. It validates exact
represented first-shell coverage, current implicit-H virtual edges,
same-center conflicts, multicenter coupling, and chelate partition preservation.
It neither infers undeclared centers nor changes descriptor family to fit
coordinates.

Gate 6 derives `R/S` or `E/Z` only when direct atomic-number/isotope priority
fully resolves ordering. Recursive sequence rules, rings, pseudoasymmetry, and
axial CIP remain explicit unresolved cases rather than false labels.

Pinned RDKit independently observed all 53 SP/TBP/OH ideal configurations
through a frozen Hotpot-to-RDKit permutation lookup. Open Babel 3.2.x was used
only as an isolated TH/DB/SP parsing-and-distinction witness. Neither toolkit
is claimed as the Hotpot semantic oracle; SPY, AXIS, chelate partitions,
enhanced groups, and Hotpot's 12/2/3-degree policy are outside those external
cross-check claims.

## 10. Implementation checkpoint chronology

The implementation used small production and documentation checkpoints. This
table records the primary dependency order; the complete one-commit-per-step
detail remains in `implementation_log.md`.

| Step(s) | Result | Primary checkpoint(s) |
| --- | --- | --- |
| 0-1 | Live log, corrected executable plan, exact RDKit/Python metadata | `066e6b2`, `39978fb` |
| 2-3 | Python sampler API and two-level stereo API frozen | `c232d1d`, `18d26ca` |
| 4-5 | Generic C++ view and field-selective Python buffers | `1707bbf`, `8f269cb` |
| 6 | Standalone C ABI/C++17 SDK surface | `356fda8` |
| 7-8 | Open Babel migrated; native policy isolated | `3b3a2b0`, `2625072` |
| 9-10 | Group/H contracts and exact offline native closure | `2801e79`, `ffb2924` |
| 11-13 | Ownership, Level-1 remapping, structure | `26f2d41`, `19c7555`, `9c0433d` |
| 14-15 | Force-field view and Level-1 transaction | `d167b00`, `1dacca3` |
| 16-17 | Level-1 packing and lifecycle | `9eb8684`, `570d756` |
| 18 | Level-2 Gate-1 algebra | `7d2debf` |
| 19-21 | Native ETKDG core, RDKit adapter, common C++ views | `1c60a70`, `56c0b42`, `32a6064` |
| 22-23 | Direction/H closure, Python binding, and SDK | `36069ed`, `42d7c4c` |
| 24 | Detached R0 runner and legacy-H closure | `7f4c60d`, `8e8d2f8` |
| 25 | Private force-field provider seam | `097749a` |
| 26-27 | Level-2 graph validation and explicit-mapping alignment | `b61e98a`, `b086577` |
| 28 | Standalone C1/C2 constrained ETKDG | `edc15be` |
| 29 | Locked manylinux wheel integration | `531130c` |
| 30 | Gate-4 coordinate semantics | `31fe1a4` |
| 31-34 | Model revision, AXIS, axial/enhanced Gate 5 | `1079b40`, `7817fab`, `b471052`, `631ff90` |
| 35-36 | Native RI and represented graph | `0a82896`, `3328f53` |
| 37-38 | Graph equivalence and direct names | `ca401d5`, `9c8f4b8` |
| 39-41 | SPY algebra and complete Gate-7 contract | `0eb3536`, `d2655b9`, `019acc7` |
| 42-44 | Gate-7 graph, coordinates, atomic public cutover | `e099a3d`, `035e2c2`, `a948279` |
| 45 | Honest Level-2 R0 reporting and canonical rerun | `90dd092`, `c1e16e7` |
| 46 | Bounded external-toolkit observations | `c79d02c`, `6673be2` |
| 47 | Supported import-order tests and local release regression | `10bc417`, `6343713` |
| 48 | Current-native C1/C2 smoke and remote artifact search | `1bac0a8` |
| 49 | Final implementation report and release-gate handoff | reporting-only |

## 11. Validation results

### 11.1 Focused and cumulative test evidence

| Checkpoint/scope | Result |
| --- | --- |
| Unchanged focused baseline after required output directory | 298 passed, 5 skipped |
| Private provider seam plus force-field/complex selection | 177 passed |
| R0 runner selection after Level-2 separation | 12 passed |
| Stereo + `rdWrappers` at Step 45 | 534 passed, 2 skipped |
| Toolkit-observation selection | 65 passed |
| Final complete stereo + `rdWrappers` selection | 601 passed, 2 skipped |
| Fresh extension/SDK, C/C++ consumer, public API, packaging selection | 71 passed |
| Installed C11/C++17 consumers | Passed, including C++ ABI 0 and ABI 1 |

### 11.2 Fresh local Phase-8 build and audit

| Item | Exact Step 47 result |
| --- | --- |
| Curated functional first pass | 2,435 passed; 12 skipped; 49 subtests; 2 failed; 9 errors |
| Cause of all 11 nonpassing cases | Read-only default model/cache paths, not product assertions |
| Writable-cache rerun | 55 passed |
| Final MCA stdout case after `HF_HOME`/`XDG_CACHE_HOME` redirection | Passed |
| Fresh offline native build | 164 Ninja steps; all 6 archives SHA-verified |
| Fresh extension/SDK focused selection | 71 passed |
| Export audit | SDK: exactly 15 `hotpot_rd_*` functions; extension: only `PyInit__rd_native` |
| Dependency audit | No RPATH/RUNPATH or dynamic RDKit/Boost; SDK `ldd -r` clean |
| Installed SDK content | 4 public headers; 8 license/notice files |
| Fresh runtime identity | Hotpot 0.5.4.0; native RDKit 2026.09.1 |

`tests/run_coverage.sh` could not emit coverage data because the optional
`coverage` module is absent. Its functional targets were run directly with
pytest. The environmental cache failures were rerun successfully, but this is
reported as staged closure rather than one initially green coverage command.

## 12. Known limitations and open release gates

Release blockers:

- The frozen 182-case manifest and graph sidecars are unavailable. The expected
  manifest SHA-256 is
  `4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377`.
  Recover it or obtain owner approval for a new cohort identity.
- There is no dedicated formal `O0`/`F1` runner. Implement and freeze it, prove
  downstream trace equality, and execute both routes on the formal cohort.
- Formal `F1` evidence and the default-provider decision are absent. Open Babel
  must remain the default until the owner reviews correctness and resources.
- CPython 3.10-3.14 wheel CI/audit has not run. Build, install, audit, and
  coexistence-test all five pinned manylinux wheels.

Evidence and compatibility boundaries:

- The formal 182-case `C1`/`C2` matrix is absent. The Eu result remains smoke
  evidence only.
- OpenBabel-first then Python RDKit exits with SIGSEGV 139 in the unmodified
  third-party wheels. Keep it as an isolated unsupported diagnostic; normal
  tests cover the supported Hotpot/RDKit-first orders.
- The local GCC 14.3 probe SDK requires GLIBCXX/CXXABI versions newer than the
  host GCC 9 runtime. Use the recorded probe `libstdc++`; do not distribute the
  probe binary as a generally compatible SDK.
- The canonical 187 importer produces empty Level-1/Level-2 payloads. Do not
  claim source chirality preservation. Future semantic R0 evidence needs a
  separately hash-governed sidecar or importer route.
- The optional `coverage` module is absent locally. Release CI must generate
  coverage; the local functional assertions were run directly.

## 13. Claims and non-claims

### Supported claims

- Native ETKDG is available through independently usable Python and C/C++
  interfaces without Python RDKit graph translation.
- The exact RDKit/native source identity is locked, offline-rebuildable, and
  runtime-reportable.
- The canonical standalone 187-ligand `R0` matrix completed with 559 successes,
  two deterministic partial batches, and zero failed/unsupported trials.
- Open Babel and the existing force-field route remain available and default.
- The private provider seam changes only initial coordinates.
- Level-1 structural transport and all seven declared Level-2 semantic gates
  are implemented and directly tested.
- A fresh local extension and SDK passed C11/C++17 API, symbol, dependency,
  RPATH, license, and supported import-order checks.

### Explicit non-claims

- Stage 17 is not release-complete.
- RDKit is not yet approved as the default force-field coordinate provider.
- No formal 182-complex `O0`, `F1`, `C1`, or `C2` outcome is claimed.
- The Eu smoke cohort is not a substitute for the frozen 182 cohort.
- ETKDG is not claimed to be a metal-complex force field.
- R0 does not prove source-SMILES chirality preservation because the imported
  stereo payload is empty.
- The CPython 3.10-3.14 manylinux wheel matrix has not passed.
- ABI-0/ABI-1 facade success does not promise compatibility with arbitrary old
  `libstdc++` runtimes.
- Open Babel is not a complete semantic oracle, and RDKit does not validate
  Hotpot-specific SPY/chelate/enhanced-group/tolerance policy.

## 14. Artifact identities

Plan and log:

- Owner-approved original plan:
  `1c502f8ed8a48589d7fc67f9d0dd0bf8b882f8ab017a4ef4fdc4617ed2fc4604`
- Current implementation-grounded plan:
  `9a3be507e47cad7f2e13a45bb22931368a80bf39d7f53b906025696fe4945dce`
- Implementation log at report preparation:
  `90a00b9ba69b2fbc76a4c58c4b342b0389c27bb1b7e9ce709338be87a8f387af`

Native source closure:

- RDKit: tag `Release_2026_09_1`, commit
  `fece8caa860bdf6c9c82bdb9253f22a0cf87150c`, archive
  `86711e340759a3a0678a2e62f54f26f7ff564c3a02de0460f60b1ce89ac1bfe9`
- Boost 1.86.0:
  `1bed88e40401b2cb7a1f76d4bab499e352fa4d0c5f31c0dbae64e24d34d7513b`
- Catch2 3.4.0:
  `122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb`
- better-enums 0.11.3:
  `9b78dcef7f88d1345b6f25335bfbcba5f024b08990c7d6dc605b833f4128b8dd`
- Eigen 3.4.0:
  `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`
- RingDecomposerLib:
  `944b5816712a48bbf88aa25d4300ce11871ddf6e971218eac08f90ed2192f715`

Benchmarks, goldens, and fresh binaries:

- Canonical 187-ligand corpus:
  `e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`
- R0 `r0_cases.jsonl`:
  `2b630f6d492955a31467daaa8525c7c811d9f3954391da0be17ce14b428dc39a`
- R0 `r0_summary.json`:
  `63ac2945abb8c5b82236f6303951873ec31bfdcedae1f90611077514953640f7`
- Fresh C1/C2 development-smoke `report.json`:
  `3c05f6b3862a1c19baddd1e3ad1776ea94992d5c6c00e96e7dbfae01d439070e`
- Gate-1 algebra golden:
  `8d7856e0b593ff7af510e885b6f3fe2ef0ca047cd34a3563bde086af9afbcf4b`
- Gate-7 SPY golden:
  `fbda20439136c68ac50a2d508acc9970003e3721d73524c1c721ffff022a473d`
- Gate-7 coordination golden:
  `a565dafee783d2fe9ffb0575b757ebb6869cb2634bcdfa5c1f9c2ca3055e8dc5`
- Step-47 fresh extension:
  `0763fe8fad5464e858e745097950179d9062a4ab26c107a2d44749eddf7f2248`
- Step-47 fresh SDK library:
  `dc8d1a7a53cc6dfa4f47995eef29a48ad8af557ba458654aebadf859152d86a4`

## 15. Release recommendation

Accept the native component, common communication layer, stereo system, and
private provider seam as implemented checkpoints. Do not declare Stage 17
released and do not change the force-field default yet. The shortest remaining
critical path is:

1. recover or explicitly replace the frozen 182-case artifact identity;
2. implement and freeze the missing formal `O0`/`F1` runner;
3. execute `O0/F1/C1/C2` and obtain the owner decision on the default provider;
4. run the five-version manylinux wheel/install/audit matrix; and
5. issue the final release sign-off from those artifacts without reusing the
   development Eu smoke result or the GCC 14.3 probe binary as substitutes.
