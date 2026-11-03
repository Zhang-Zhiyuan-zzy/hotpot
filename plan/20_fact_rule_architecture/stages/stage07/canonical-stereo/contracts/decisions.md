# Stage7E — Canonical stereo storage

Root Sol/max entry decision, 2026-11-03. Stage7D CP311 owning scope is accepted
at c4f6bec. This releases the connected authoring partitions below, not runtime,
Stage8, global ownership or final acceptance. The effective goal, main-plan5.3
and Stage7E, Stage1 A2, Stage5 H conventions and Stage6 extraction/public
contracts remain authoritative. Public StereoPayload/classes/signatures/pickle
spelling and public19/native prefix do not change. No new class, dataclass,
Protocol, enum, named native struct or ABI mirror is authorized.

## E1 — Canonical current row plane

The existing _AtomFactTable gains `_stereo_facts`: a capacity-aligned list of
immutable tuples, one per Atom row. Each row retains **all occurrences** in
transport order; an occurrence is exactly:

`(isotope, radical_electrons, current_h_present, current_implicit_hydrogens,
current_bracket_explicit_hydrogens, current_no_implicit, source_namespace)`.

Optional exact builtin ints retain arbitrary magnitude/sign; bool and int remain
distinct. The third slot is an exact bool; absent H container has three None
fields. Namespaces retain None versus empty string. None in each optional field
is its absent declaration; an empty row tuple is absent transport row. A present
all-None occurrence remains present. Multiple declarations remain multiple;
storage never selects a duplicate winner. Values/presence/current namespace
exist only here, never in Atom dictionaries, a numeric mirror or source ledger.
Legacy numeric slot14 remains the separate supplied/inferred site-total target.
Transport omitted-H never becomes a new rule input or overwrites slot14.

Reuse N05 _AtomFactUpdate with trailing `private_values=()`; when nonempty it is
one immutable replacement row tuple per selected row, aligned with rows/uids.
Numeric fields/value_bytes/presence/source retain their existing contract and
public batch permissions. Native-created numeric updates initialize this slot.
This is a storage ingress used inside the complete stereo publication; it is not
a new public or standalone transport setter. Source identities inspect canonical
row/display/layout owners directly. The existing native Undo gains reference-owning list-row before/after entries,
not another journal/type. One prepared update can atomically publish both numeric
and private rows, callback invalidation and rollback, including failure injection.
Configuration names the existing table private slot. Changed private rows notify
`stereo_facts` representation invalidation; exact H seeds remain the stereo
operation's responsibility. Table growth/remap/take/pickle/detachment carry this
plane; scalar writes touch selected rows rather than copying the molecule.

## E2 — Source-only layout and Bond display

Molecule owns `_stereo_schema`, `_stereo_records`, `_stereo_source` only. Source
is an immutable `(atom_layout, bond_layout)` tuple. An atom layout entry is either
an original unbound AtomTransportRecord (out-of-range row), or
`(atom_uid, occurrence_index, source_hydrogens)`. Bound entries contain no current
values/namespace. A bond layout entry is either an original unbound
BondTransportRecord (missing endpoints/edge), or `(bond_uid, occurrence_index)`.
Each Bond owns `_display_facts`, an immutable occurrence tuple of
`(first_atom_uid, second_atom_uid, display_direction)` preserving orientation,
None/empty/unknown display strings. Dative `_bond_direction` is unchanged.
Parallel structural bonds are already rejected by the topology foundation.
Repeated transport rows remain separate entries in their canonical owners.
Records/schema retain opaque schema47/anchor81 without interpretation.

`_decompose_stereo(mol, sealed)` returns
`(schema, records, source, atom_rows, bond_rows)`; atom_rows aligns every Atom,
bond_rows every complete Bond. `_install_stereo_storage(mol, sealed)` installs
that prepared split on a freshly built table, no chemistry. `_get_stereo_payload`
assembles from current rows/Bonds and source-only layout under the owner lock;
it never prepares H/aromatic facts, caches the output or promises payload identity.
`_stereo_storage_state` / `_restore_stereo_storage` retain exact schema/records/
source and Bond display owners for the existing transaction. Native Undo owns
in-place private row restoration. Raw UID/layout integrity failures remain errors.

## E3 — Isotope and H consumption

Delete sparse Atom isotope truth and N02 atom_isotopes. Existing isotope reads
project table occurrences. With no occurrence, raise AttributeError as the former
missing attribute did; an existing occurrence with None projects None. Equal
isotope declarations project that value; conflicting duplicates fail precisely
at this scalar consumer rather than selecting a storage winner or rejecting set.
Direct isotope assignment normalizes the existing integral transport domain with
operator.index (or None), replaces isotope in **every** occurrence and preserves
all other fields/source history; when absent it appends a minimal transport row.
No new public constructor/batch field is added.
RDKit registers the exact native isotope-only subset: namespace=None and radical/
current-H/source-H all absent. Isotope is unambiguous in that subset; apply the
existing isotope map quietly, with no added runtime label. Other unregistered
H/radical/source conventions and unknown namespaces retain adapter rejection.
The original MCA isotope/symmetry fixture demonstrated the need for this connected
map when native isotope values moved into transport. A missing isotope in MCA's numeric
encoding remains zero; a row whose isotope is None uses that same absent encoding.

Preserve existing effective H consumption (radical/bracket/no-implicit with
0/0/False defaults). Its historic last-occurrence reduction is **operation-local**,
not a canonical storage winner. Preserve all-occurrence H materialization and
Gate6 duplicate/out-of-range diagnosis before H/native preparation. New regression
fixtures make this existing distinction explicit. Get/set reassembly must not
calculate H, normalize isotope/H scopes or alter scientific algorithms.

The existing implementation accepts set(None): preserve it as explicit cleared
storage (`_stereo_schema=None`, empty other owners/rows); getter returns None.
Its public annotation stays unchanged and this does not add adapter support.
N02 None continues to mean no supplied transport: factual installation normalizes
that absence to empty schema1 as before; no record/adapter roundtrip guarantee
is added for the undeclared direct-set(None) edge.
A later direct isotope assignment creates schema1 transport for the new fact.

## E4 — Connected transport and lifecycle

N02 is Record24/tagged-v5: remove atom_isotopes, retain one **transient** assembled
stereo_payload for complete immutable transfer. Reject outdated private pickle
arity/version explicitly; no compatibility mirror. All constructors, MCA DTO,
CCDC/RDKit/OpenBabel, capture/install/merge and cold pickle migrate together.
Whole canonical-order factual capture preserves opaque/unbound entries exactly;
partial/reordered/component selection uses the existing sole pure remap/filter
algorithm. Component policy does not fabricate bindings for unknown references.
Reinstall split bindings after cold-pickle UID restoration. Detached Atom tables
retain immutable private rows; no owning object-pointer array is byte-sealed.
Conformer/H transport, native packing and force-field working-copy publication
use on-demand assembly; rollback restores original owners and exact table cells. Removed Bond handles
retain their exact immutable display owner; only current Molecule transport is
filtered. The original source reproduction observed display loss before repair.

## E5 — Authoring partitions and acceptance

A: local Sol/high owns _atom_table.py, _mutation.py, structure/_native/facts.cpp
and a new focused table/native test file. B: local Sol/high owns core.py,
_builder.py, MCA serialization and toolkit/CCDC record constructors plus bounded
builder/lifecycle tests. C: local Sol/high owns _stereo.py and canonical storage
regression tests plus mechanical migration of transitional stereo identity tests.
Root owns _commit.py, _identity.py, _results.py, operations, capture/native/conformer/
working-copy integration, cross-partition review, records and Git. Writers share
one workspace; no reverts, nested agents, product imports, builds/runtime or Git
before root integration/freeze release. No external helper is active.

Required evidence: exact lossless values/types/presence/source/order/schema,
large/negative ints, opaque/unbound/duplicate fixtures, direct isotope consistency,
H seeds/scopes/side-effect ordering, reference-aware failure/outer rollback,
copy/components/merge/remap/cold pickle, toolkit/native/conformer/FF publication,
zero-new-type/no-persistent-payload audit; affected CP311 source and installed
selection with rebuilt factual native; actual FF/pipeline/MCA handoffs; bounded
allocation/throughput. Reuse valid unchanged evidence. Non-CP311 adaptation and
clean all-native/ABI/SDK/sdist/global science/performance remain honestly open.
