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

Keep the existing read names `Atom.attrs` and `Molecule.atom_attr_matrix` as
properties returning independent float64 snapshots in the existing 19-column
order. They have no assignment or write-back path; writing a returned array
does not change the molecule. Do not store a second 19-column table. Optional
result slots without a current selected value encode `NaN`; an explicitly
present zero stays zero. This missing-value encoding is an explicit observable
change accompanying A8's `Atom.partial_charge -> Optional[float]` contract.

The old fixed projection and the planned explicit `atom_feature_matrix` use
the same schema and field-value implementation. Their export policies differ:
the fixed projection encodes absence; `atom_feature_matrix` rejects an unknown
selected value as already specified by A5. Neither export invents a fact or
adds its own validity/recalculation path.

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
