# Public-output and facade simplification — 2026-11-06

Owner-authorized continuation of the coordinate/query refinement at
`16c01664b1fa2c6caab4e447bfba0ba421d7b7ca`, on
`feat/automatic-coordination-state-workflow`. The review extends beyond
`Molecule.build3d` to related FF, inference, xTB, carrier and writer interfaces.
The objective is to remove repeated inputs, constant placeholders, ignored
parameters and redundant forwarding while retaining actual observations and
control/attribution contracts. No chemical algorithm, default state-selection
rule, model feature encoding, native ABI or trajectory/SDF format is changed.

## Implemented migration

| Owner | Simplification | Meaning retained |
| --- | --- | --- |
| FF run/workflow/setup reports | Remove repeated `requested_forcefield`; run also drops `convergence_level` and permanently unavailable `steps_completed`. | Effective method, actual submissions/initialization/epochs, stopping and selected/terminal observations remain. Exact completed backend work stays explicitly unknown in evidence recorders. |
| FF validation/build/ring/restoration reports | Remove validation `level`, build `atom_count`, ring `attempt_limit`/`resolved`, restoration `attempt_limit`/`bond_count`. | Invocation settings remain with callers; actual attempts, piercing observations, coverage findings, relocation results and added H remain. |
| FF JSON and maintained benchmark serializers | Optimized workflows serialize final quality once at `optimization.quality_report`; build-only workflows retain top-level quality. | Existing Python `workflow.quality_report` access and the initial-build quality observation remain. |
| FF builders and conformer seam | Remove ignored public `candidate_count` and private `ConformerSamplingRequest.conformer_count`. | One-placement seam, seeds and all active budgets stay. Real native `ETKDGOptions.conformer_count` still controls generation. |
| Atom/Molecule/operations | Remove unused `_calc_implicit_hydrogens` and `_set_remapped_stereo_payload`; remove the `_add_hydrogens` carrier bounce and unused `_update_topology(clear_conformers=...)` argument. | Live pure-rule/operation seams, public H operations, remapping, topology/Crystal pruning, source frames and transactional rollback stay. Tests spy on the live operation rather than retired wrappers. |
| Molecular writer | Forward the documented `ob_opt` mapping instead of silently discarding it; document actual string/None returns. | Default writing is unchanged. The real Open Babel option is verified with isotopic SMILES without mutating molecule facts. |
| CBond stage | Remove repeated `metal` from its single-path stage report. | Saved pipeline argv and independently checked output chemistry identify the requested metal; scores, path ordering and sample accounting remain. |
| MCA result projection | Serialize nested predictions once. | Ordered five-key JSON schema, empty-list behavior, nested container isolation and all predicted values remain. |
| xTB stage | Retain a failed native report once through the existing `report_index`; omit constant projection/process-success tags and repeated method/task inputs. | Original failure cause/log/report, state-selection findings, source binding and sample denominators remain. |
| xTB execution report | Consolidate process/workspace facts at the existing `ProcessProvenance`; remove derived process-success and repeated requested-method fields. | Return code, native streams/timing/argv/workspace, effective method, finite observations, convergence, artifacts, commit and cleanup semantics retain their owners. |

Constructors, field introspection and newly emitted report JSON deliberately
change. There are no compatibility aliases, generic result envelopes or new
provenance machinery. Historical evidence is not reserialized to the new schema.

Ordinary FF optimization now records its actual native-adapter duration instead
of leaving an unmeasured zero. The timing boundary is documented separately
from native Stage3 time. The scientific comparator follows the final-quality
owner, retaining monitor observations after JSON consolidation.

## Deliberately retained interfaces

Scientific-state detailed envelopes have actual composition, rejection,
projection and persistence readers and remain under the accepted A9 contract.
MCA site/Atom joins and detached DTO identities support real consumers. Conformer
source frames and per-field observation attribution describe different lifetimes
and mixed-method assignments. View snapshots enforce currentness. Native raw
reports remain transport contracts. None is retired merely because its fields
look similar or a bounded search finds no additional reader.

The initial [build3d assessment](../../stage08/forcefields/build3d_field_assessment.md)
is retained as historical discovery; this migration supersedes its decision to
defer cleanup. The focused [inference/xTB record](../../stage08/inference/validation/output-simplification/report.md)
contains its connected schema and consumer decisions. P20-D047 is the owning
FF verbosity issue in the unified deferred index and is CLOSED by this bounded migration.

## Verification and limits

| Boundary | Actual outcome | Coverage |
| --- | --- | --- |
| Current-source carrier/H/stereo/topology/I-O | 1,080 passed, 3 known xfailed | 28 affected modules; isotope writer option, H operations, transport/currentness and rollback. |
| Current-source FF and connected evidence readers | 838 passed, 2 opt-in skipped | 50 modules; full affected Python map, native adapters, serializers, timer, A08 and scientific comparator. |
| Current-source xTB/pipeline/CBond/MCA projections | 443 passed, 10 external skipped | Complete default xTB selection and connected CLI, batch containment, benchmark and MCA serialization. |
| Isolated installed CP311 | 292 distinct cases passed | 291 initial passes plus the one missing-support case after repair; same wheel/site and unchanged product. |
| Actual source builders | Organic CC and coordination [Zn](N)N returned | Tiny budgets and quality-off: schema/plumbing only; high finite coordination energy and placement warning retained. |
| Actual installed automatic workflow | UFF → GFN-FF → GFN2-xTB water completed | Both xTB stages converged; graph/atom order, finite geometry, legal automatic state, report fields and commits checked. |

Source selections overlap and are not an additive unique-test total. The three
xfails are the existing ring-order aromaticity, fused-azulene perception and
shared-edge joint-ring limitations. The FF skips require scientific-boundary
and high-compute options; the ten xTB skips require `--run-external`. A separate
actual installed water calculation supplies one narrow real-backend observation,
not qualification of those unexecuted fixtures or a new corpus pass.

The installed workflow supplied no charge, UHF, multiplicity or state override.
Its imported Open Babel charge fact retained precedence. Hotpot selected charge0;
GFN-FF uses charge-only projection and GFN2 selected UHF0 by its execution default.
Scientific resolution remains reported as unresolved; backend convergence does
not establish a unique scientific state. Finite energies were
−0.327656101804197 Hartree (GFN-FF) and −5.07054436 Hartree (GFN2). Total workflow
time was0.435800s, a recorded smoke observation rather than a performance result.

Original failures remain archived: missing loader path, unguarded private pytest
launcher, stale constructor/field fixtures and missing copied benchmark support.
The invalid recursive-launcher observations are not qualification. Final FF and
non-FF freezes match current source. Carrier-owned source/tests stayed fixed;
seven later FF/xTB changes are covered by their final suites and installed
qualification. The installed snapshot differs only in the subsequently clarified
`hotpot/commands/ff.md`; all production Python and six reused natives match.


The associated [checkpoint](checkpoint.json) records actual commands, source/native/wheel
bindings, results, original failed attempts and evidence locations. Current
checks prioritize CP311 business behavior. No report-size or speed improvement
is quantified, and this Python-interface migration makes no fresh full-corpus,
model-arithmetic, clean-native-build or other-Python acceptance claim. Previously
recorded chemical/numerical limitations remain independent of this cleanup.
