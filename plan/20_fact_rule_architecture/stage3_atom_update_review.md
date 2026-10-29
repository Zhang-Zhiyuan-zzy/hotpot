# Stage3 closed Atom update preparation

Entry: `stage3_cutover_entry_review.json`, following `d7989b3`. The main plan
and four normative companions were read completely before this work. Stage3
remains in progress; no production table cutover or public batch publication
is claimed by this preparation.

N05 is one private `_AtomFactUpdate` in `structure/_mutation.py`, with fields
in this order: `source, rows, atom_uids, fields, value_bytes, presence, sources`.
It composes the existing `_SourceIdentity`; row/UID pairs occur once per
update, never per field. Numeric candidate columns are stored as private
immutable bytes, including after pickle/deepcopy. The `values` property creates
temporary float64 ndarray headers; changing their dtype/shape cannot change
the saved candidate. A retained read-only ndarray alone is insufficient because
NumPy permits those metadata changes. Metadata is
one bool and one fixed producer code per affected column. Complete imports,
copy and FF restoration retain heterogeneous row metadata through the complete
record/table path rather than flattening it into one producer code.

The normalizer accepts only the already approved batch fields in `_fields.py`:
atomic number/symbol, formal charge, implicit H, source/display id and
coordinates/x/y/z. Scalar and selected-row batch input share this preparation.
Coordinate triples broadcast; `(N,3)` input preserves selection order. Symbol
translation uses the existing immutable element-symbol tuple and Open Babel
spelling convention (first-letter case, D/T as H, unrecognized symbols as0),
without a toolkit import or introducing isotope facts. The JSON property asset
has differing atomic-number values for In/Tb/Yb/Bk, so it is not substituted
for the existing symbol order; correcting that asset is outside this change.
Overlapping aliases
must agree after translation; canonical output follows the shared19-field
order. Existing wide/raw float64 numeric storage is not narrowed.

Unknown fields raise TypeError; known non-writable fields raise AttributeError;
shape/alias/duplicate-selection mismatches raise ValueError. Member selection
and the existing membership exceptions belong at the Molecule boundary, where
owner, row and UID are checked under its existing lock. Ordinary writes use
owner-only binding and no molecular digest; long operations can supply named
dependencies. Preparation itself never reads/mutates a carrier or installs a
table. No installer placeholder, arbitrary FactPatch or extra public method
is created at this checkpoint.

The coherent production cutover will consume this record in the one
`_commit_atom_facts`, checking row/UID alignment before `table.update`, and
connect scalar/batch setters together. Old chemical side effects remain until
Stage5; readonly generated properties and selected-result ownership retain
their assigned later stages. Actual member-selection, molecular rollback,
dependency marking and integrated performance gates remain open.

Before that cutover, the existing private table's pickle/deepcopy path must
serialize its unique blocks and metadata and rebuild column views. Serializing
the views independently creates conflicting reads from the same table.
