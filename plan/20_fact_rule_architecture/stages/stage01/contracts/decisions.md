# Stage1 concrete interface decisions

This record supplements the captured API/consumer inventory. The latest
implementation instruction authorizes the plan's concrete route. The A2
constructor parameter disposition below was not specified by that plan and is
presented for the owner decision required by main-plan Section16.4. The owner
**accepted the exact A2 proposal on 2026-10-28** through the implementation
conversation. Its stage-specific cutovers still require their named gates.

## A2 — Raw Atom constructor input and fixed numeric projections

### Proposed final public contract

```python
Atom(mol: Molecule | None = None, **kwargs)
```

Accepted keyword facts are `atomic_number`/`symbol`, `formal_charge`,
`implicit_hydrogens`, `coordinates`/`x`/`y`/`z`, and `id`. Existing constraint
inputs `constraint`, `x_constraint`, `y_constraint`, `z_constraint` remain
accepted and commit to the Molecule-owned ConstraintSet in the same
construction transaction. Overlapping aliases must agree. Construction keeps
the characterized zero-number placeholder and zero-coordinate defaults;
presence records whether the caller explicitly supplied a value.

Remove the two explicit constructor parameters `attrs_array` and
`update_electron_config`. These names, arbitrary unknown keywords, and
derived/result inputs `n`, `s`, `p`, `d`, `f`, `g`,
`electron_configuration`, `valence`, `is_aromatic`, `partial_charge` raise
`TypeError`; they are not silently ignored or forwarded through arbitrary
`setattr`. Public `isotope` is not added. The private builder continues to
transport supplied isotope/stereo facts through the existing contracts.

Keep `Atom.attrs` and expose the owner-approved `Molecule.atom_matrix` as
properties returning independent float64 snapshots in the existing 19-column
order. They have no assignment or write-back path; writing a returned array
does not change the molecule. Do not store a second 19-column table. Optional
result slots without a current selected value encode `NaN`; an explicitly
present zero stays zero. This missing-value encoding is an explicit observable
change accompanying A8's `Atom.partial_charge -> Optional[float]` contract.

The original A2 decision retained `Molecule.atom_attr_matrix`. The owner's later
instruction explicitly asks for `Molecule.atom_matrix` and says to check existing
facts and avoid duplication. Apply that latest naming instruction as a direct
rename: migrate the existing entry and its consumers, without an old-name alias.
This resolves the earlier optional clarification from existing authorization;
no new user reply is asserted or inferred from elapsed time.
Both row and complete-matrix access use one schema/projection implementation.
Field-selective native/model snapshots remain private; no second public export
method, persistent matrix, validity path or calculated fact is introduced.

### Why an exact constructor decision is needed

The current 19-column array includes electron configuration, valence,
aromaticity and partial charge. `update_electron_config=False` currently lets
callers preserve supplied configuration columns without recomputation. The
planned final permissions make these calculated/result projections read-only.
Preserving the parameter would require an import override state, a separate
compatibility read path, or silently ignoring its meaning. Explicit removal
and complete private imports keep one source of truth.

Keeping the read projections avoids unnecessarily deleting real read APIs.
The legacy zero partial-charge slot cannot itself distinguish an omitted value
from an explicit zero. Converters must supply presence from their source
format and complete result records, rather than infer it from numeric zero.

### Coupled consumer migration and stage order

| Consumer | Required migration |
|---|---|
| `cheminfo/obconvert.py` and `plugins/ccdc_api/convert.py` | Decode external facts into one private complete MoleculeRecord/import builder; retain source presence, stereo, coordinates, labels and supplied result values |
| `core.py` row construction, copy and components | Use the same factual builder and explicit row/UID remap; do not discard existing factual columns during table extraction |
| `forcefields/working_copy.py` | Replace raw-array copies and restores with the one factual snapshot/commit path, preserving rollback and caller state |
| `AImodels/data_extract.py`, `plugins/PyG/data/utils.py` | Keep the existing model tensor shape/feature order and explicit task-side missing-value encoding; never feed new NaNs to an unchanged model by accident |
| CBond's `atomic_number_only=True` extraction | Read only the atomic-number column; do not project all 19 fields to obtain column zero |
| Existing snapshot/lifecycle tests and other captured readers/writers | Map the former array assertions to value, isolation, presence and rollback contracts; migrate every writer before removing its path |

The decision describes the final interface, not permission to discard values
early. Stage3 establishes the sole fact table, alias isolation and private
builder while preserving the facts needed by still-unmigrated owners.
Stage5 changes electron/valence/aromatic ownership with its consumers and
removes the two old constructor parameters after their imports are migrated.
Stage7C moves partial charges into the selected complete result owner and
establishes the stated absence encoding. The constructor cutover occurs only
when its supplied-fact consumers have an equivalent private import path; its
exact commit and old-to-new assertion mapping are recorded in the log.

Required evidence includes constructor signatures and rejected keywords,
scalar/batch alias consistency, buffer isolation, full import/working-copy
rollback, fixed 19-column projection order, explicit-zero versus absent-result
behavior, and unchanged model tensor encoding. No compatibility constructor,
duplicate authoritative array, new isotope API or scientific algorithm is
introduced by this proposal.

Owner decision: **accepted on 2026-10-28** ("采用该方案（推荐）"). This covers
the precise constructor removal, fixed independent snapshots, missing-result
NaN encoding and preservation of current model input encoding stated above.
It does not advance production work ahead of the Stage1 baseline or merge
the separate Stage3/5/7C ownership changes into one unvalidated change.

## Concrete implementation mappings for the owning-stage review

The following mappings resolve implementation detail without adding a common
base class, a new generic rule protocol, or optional public aliases. They are
recorded before the relevant production stage; any genuinely new result class
or schema proposal remains explicitly identified below.

### A4/A5 and current versus historical reads

- `Molecule.updata_atoms(*, atoms=None, **columns) -> None` accepts
  `atomic_number`, `symbol`, `formal_charge`, `implicit_hydrogens`,
  `coordinates`, `x`, `y`, `z`, `id`. Constraints retain their existing
  Molecule-owned operations. `None` selects all current atoms in order;
  explicit selections preserve their order and cannot contain duplicate,
  detached or foreign members. Scalar fields broadcast; a coordinate triple
  broadcasts or an `(N,3)` array supplies the whole selection. Empty writes
  return None. Unknown fields raise TypeError, read-only fields AttributeError,
  shape/alias conflicts ValueError; member failures reuse the applicable
  existing membership exceptions. The complete write is one transaction.
- `Molecule.atom_matrix` is a read-only property returning an independent
  float64 ndarray in current Atom order and the existing19-column order.
  It accepts no fields/dtype parameters; an empty molecule gives `(0,19)`.
  Optional selected-result absence uses the A2 NaN policy at Stage7C; explicit
  zero remains zero. It adds no MCA column, private UID or unreviewed feature.
  Pure dependencies are prepared once for the batch; no implicit model or
  external execution occurs. `Atom.attrs` uses the same field projection.
- Keep the existing AttributeError for absent MCA predictions; absent selected
  partial charge returns None as specified by A8. A source-dependent selected
  result that has become stale raises RuntimeError on its direct current-value
  access, with the changed dependency and explicit recalculation operation
  identified. The approved fixed numeric export encodes an unavailable slot
  as NaN. Historical returned result values retain their own content.
- Source-bound view calculations reject a changed relevant source with
  RuntimeError; their immutable member identities remain inspectable.
  No new `StaleStructureViewError` class or `AtomPairTable` alias is introduced.
  Scientific uncertainty alone is not a stale-view exception.

Owner decision on2026-10-28: the two capabilities are accepted with the exact
names `updata_atoms` and `atom_matrix`; the latter is a full-array float64
property. The earlier update_atom_facts/atom_feature_matrix proposals are
superseded, not additional aliases. The requested duplication audit found the
existing full atom_attr_matrix projection, so only one implementation remains.

### Crystal, external thermo results and conformer projections

- Keep `Crystal(a,b,c,alpha,beta,gamma)` and `from_matrix(matrix)` with the
  characterized convention: the current alpha is the angle between the
  first two matrix rows. Apply the plan's removal of `mol`, `obcell`,
  `space_group`, `lattice_group` and LatticeGroup; do not add a matrix property.
- Preserve `Molecule.get_thermo(temp=298.15, pressure=101325.0) -> Thermo`
  and the existing Thermo proxy behavior in the external adapter. The
  constructor and `get_chem`/`mol_to_thermo` helpers move to their assigned
  adapter path. Thermo has 181 declared properties and unrestricted forwarding
  to Chemical, including executable external objects. It remains a
  caller-owned task result and is not retained as carrier state. Replacing
  this with a small frozen record of README properties would silently delete
  real interfaces and is not part of the move. Pure values selected for
  persistent attachment use the planned typed result owner separately.
- Conformer projection order is source-frame insertion order followed by
  result attachment order and each ensemble's original frame order, with no
  deduplication, sort or second persisted coordinate array. Integer access
  retains a dict with `(N,3)` coordinates; slice access a dict with `(K,N,3)`.
  Keep the nine keys `coordinates`, `energy`, `partial_charge`, `gibbs`,
  `force`, `zero_point`, `spin_mult`, `thermo`, `capacity` and None returns
  from add/load/clear. Load commits only current coordinates; clear removes
  both source frames and attached ensembles. Preserve existing value units.
- Baseline probing found two existing conformer bugs: empty iteration raises
  TypeError, and adding an energy-bearing frame after a frame without energy
  can write coordinates before raising ValueError. Stage7D must retain their
  characterization as defects and verify atomic failure in the replacement;
  it must not reproduce partial mutation as a required behavior.

### Exact D01–D05 result-family proposals

These records refine the already documented consolidation candidates. D01–D03
public construction/schema decisions remain unapproved and require concrete
review before dependent Stage8A publication; the A2 approval above does not
approve these separately.

2026-10-30 alignment: The [current goal](../../../goal.md) supersedes D01's generic
public admissibility/confidence-envelope proposal. It was never approved or
implemented; its exact historical proposal is retained below. D01 remains open
for a concrete design of requested domain values and all producers, consumers
and codecs. Preserve candidate data and necessary internal hard-rule,
provenance, source-binding and lifecycle records. This alignment specifies no
replacement public schema/signature and grants no approval.

| Item | Retained/new definition and exact behavior | Schema/consumer obligation |
|---|---|---|
| D01 | **Superseded, unapproved historical proposal (2026-10-30):** `rules/chemistry/scientific_state/contracts.py::StateInferenceResult[T]`; field order `quantity, identity, facts_identity, candidates, admissibility, resolution, confidence, findings`; quantity reuses the existing ScientificQuantity domain, with the existing candidate element types | **Historical migration proposal, also superseded:** Replace the five old result classes and their return annotations; no empty alias subclasses. `asdict` gains quantity explicitly, while existing explicit candidate-space/decision codecs retain their schemas. Authority/control policy is unchanged. Current D01 review follows the dated alignment above. |
| D02 | Same module: `StateFinding(code: str, message: str, related_identities: tuple[str,...]=())` | Preserve current code strings and context roles. Existing static-eligibility codec stays code/message only; other reason codecs retain related identities. Do not replace those distinct explicit codecs with indiscriminate asdict. |
| D03 | `operations/calculation/scientific_state/contracts.py::StateActivation(namespace: Literal["provider","rule"], identifier: str, enabled: bool, configuration: tuple[tuple[str,ProviderValue],...]=())` | Replace provider_id/rule_id with explicit domain and identifier; keep separate ID namespaces, defaults and permission checks. Identity digests retain their existing domain-normalized payload. |
| D04 | Retain `operations/forcefields/contracts.py::OptimizationStoppingCriteria` with its five existing fields/defaults/validation | Remove the Python-only native OptimizationStoppingOptions and copying helper; preserve the real C++/pybind ABI type. |
| D05 | Retain one `structure/snapshots.py::BondTopology(atom_indices: tuple[int,int], bond_order: float, bond_kind: str)` with frozen/order behavior and ascending endpoints | Remove BondTopologySignature, use the retained record for transaction references and trajectories, preserve three-field historical JSON. UID/index mapping belongs in the complete snapshot. Any retained from_bond facade delegates to the shared collection operation. |

D06, cosmetic MCA-message renaming and unrelated plugin record changes remain
outside implementation. D07's historically reserved/public dormant records
retain separate disposition; this does not delay the plan's confirmed dead
implementation cleanup. Definitions are booked as actual removals/additions
only when their consumer migration and owning-stage commit have landed.
