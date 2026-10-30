# Stage5 concrete implicit-H read and default decision

Status: the owner's 2026-10-30 direction requires valence-based defaults when no
count was supplied and retention of an explicitly supplied count. The previous
optional-scalar/NaN-export proposal is superseded and was never approved.
The bounded scientific recommendation and its evidence are maintained in
[hydrogen-defaults/decisions.md](../hydrogen-defaults/decisions.md),
[review.md](../hydrogen-defaults/review.md) and
[evidence.json](../hydrogen-defaults/evidence.json).

The recommended public signature and existing assignment permission are:

```python
Atom.implicit_hydrogens: int
```

A supplied explicit/source count wins, including zero. The integer scalar
codec remains; independent `Atom.attrs`/`Molecule.atom_matrix` numeric snapshots
retain supplied float64 bytes, including characterized legacy counts outside
the usual chemical-count domain. Their shape, dtype and19-column order remain.
No optional scalar, H-to-NaN export, public result type, exception class,
validity API or fallback calculation is authorized by this decision.

An absent count on an ordinary known element invokes the one named local pure
preparation lazily. It commits the finite default with the existing producer,
reuses it until consumed inputs change, and does not overwrite charge, radical,
isotope or supplied hydrogen declarations. The finite charge-sensitive table,
radical policy, aromatic and metal/dative cases are specified in the owning
scientific decision. A real isolated H/H+/H- has H0; it is not an unresolved
request for a missing neighbour. Scientific nonuniqueness, known unusual
valence and unsupported known states receive the documented conservative
no-addition convention; they do not become exceptions.

For genuinely missing required facts (Z0/absent element, an element-unknown
neighbour needed for completion, UNKNOWN bond kind, missing needed dative
direction), the private rule returns no proposal, H presence stays absent, and
the group remains UNRESOLVED without repeatedly running unchanged inputs.
Keep the historical public display: the scalar remains integer0 and a
permissive `Atom.attrs`/`Molecule.atom_matrix` H slot remains float64 zero for
that absent slot. The placeholder is not a calculated/supplied H fact; reading
it does not set presence, source or CURRENT, and does not return a stale
inferred value as current. An explicit/source zero or a valid inferred zero
has presence and CURRENT, even though its numeric display is the same.

Strict H consumers use `_require_implicit_hydrogens` and its existing
`ValueError` before chemistry or strict native/toolkit/model projection. A
public display value is not their readiness test. Explicit H still reads on
Z0; another elemental consumer may separately require a real element.
CONFLICT continues through its existing error boundary.

A2 specifies NaN for optional selected-result absence, in particular
A8/Stage7C partial charges; it does not change this P2 H count to an optional
result or require NaN here. The generic UNRESOLVED rule prevents a zero fact
or strict zero fallback; the retained legacy display does neither. This
route preserves the established public boundary and needs no new owner
approval. The prior scalar/projection ValueError proposal is superseded,
as is the unapproved optional/NaN proposal. A future visible unknown-H change
would still require its concrete Section16.4 owner review.

A real complete representation conflict retains facts and CONFLICT; its
existing `RuntimeError` names the explicit resolution operation. Scientific
uncertainty alone is neither a conflict nor an exception. Strict hydrogen
addition/toolkit/native/SMARTS/model consumers prepare their requested rows
once before executing; they use the same missing-input/conflict boundaries.

For this bounded default correction, retain the legacy **ordinary-nonmetal
inferred site-total-H target** and raw supplied values. Metal inference is an
explicit non-total raw0 exception even with graph hydrides; real H retains its
separate isolated-H/H2 rule. Do not turn metal graph H into a new raw/model/
SMARTS feature value. Existing toolkit imports and
some consumers use incompatible non-graph-H meanings. The connected import and
consumer normalization in the owning decision is explicitly a **proposal**:
it would change imported field14 when graph H already exists and must not be
published as an approved default/read implementation detail. The present
default/read correction requires no imported-count normalization.
Retain supplied source/model encodings and scope residual-expansion correction
to the new inferred targets. Universal explicit-H/transport coherence remains
unclaimed; resolving the inherited mismatch is separate connected work, not a
new gate imposed on this bounded default correction.

The H-specific input extraction and closed source binding must include every
newly consumed formal-charge/Bond-kind/dative/aromatic/radical/current-H input.
Radicals come from the existing transport payload. The final tuple is fixed:
`(uid, effective_radical, current_bracket_H, current_no_implicit)`; a radical-only
binding does not implement the chosen algorithm. Its writer seeds the existing
H group only for changed consumed fields. Bind no opaque payload or historical
ledger and add no carrier schema.

For automatic materialization of a new inferred ordinary-nonmetal target,
create target-minus-graph H and consume current bracket H in the same outer
transaction through the existing current-H payload commit: set current
implicit/bracket counts to0 and preserve current no-implicit. Keep the
source_hydrogens ledger and raw inferred target/source unchanged. Graph H and
remaining current bracket H must never overlap. The next preparation on the
new graph/current tuple returns the same target and automatic add-H is
idempotent. Include graph creation, payload replacement, topology finalization
and dependency invalidation in both Atom and Molecule transaction coverage;
partial failure restores their published state. The owning decision Section7
specifies the exact boundary and evidence.
