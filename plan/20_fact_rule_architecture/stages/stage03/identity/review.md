# Stage3 identity and metadata checkpoint

This continues Stage3 from `40312df`; it does not enter Stage4. On 2026-10-29
the main plan and all four normative companions were read in full again.
`entry_review.json` records hashes, retained complete read copies,
the clean synchronized branch and the missing gates. Scientific rules, MCA
backend selection and the Stage13 experiment retain their assigned stages.

## Active member identity

Molecule now owns an opaque owner token, a private reentrant lock and separate
non-reusing Atom/Bond UID allocators. Every actual Atom/Bond constructor assigns
one private UID and its owner binding. Direct Bond construction consumes one
UID; registration does not allocate another. Public `id`, current `idx`, Python
object identity and the internal UID remain distinct.

Compaction and source-label changes retain member UIDs. Removal never reduces
high-water marks. A failed operation may consume identities; it must not make
them reusable. Direct Bond endpoints must be current members. Construction and
the controlled insertion paths establish the owner binding; the existing
topology index retains its membership checks. No persistent UID-to-row map or
second topology index is introduced.

Foreign insertion copies its factual row into a local Atom and rejects a
same-owner duplicate, following the concrete lifecycle route in main §7.5.
Replacement retains the detached old handle and installs a new local handle.
Component merge recreates local endpoints and Bond identities, including both
temporary hidden partitions. Copy and components reuse the raw-row copying
path, preserving supplied electron slots and the existing sparse isotope and
`frac_valence` inputs. These changes do not decide pending charge/spin/cell,
annotations or generated-result copy policies. The final complete builder
still has to consolidate these lifecycle writers.

Replacement retargets active and hidden Bond endpoints while preserving Bond
UID/direction. Single-Atom public insertion/replacement also remaps the existing
AtomTransportRecord (isotope, radical, current/source H and namespace) to the
destination row and removes the replaced row's old transport fact. Multi-atom
stereo records are not imported from unrelated source members. Whole-copy and
merge paths retain their existing one-shot full-payload remapping.

Current-version pickle omits the lock, recreates it on loading and rebinds the
existing topology index's Python-object keys to the restored members. Member
UIDs and allocator high-water marks survive. This is not a historical old-core
pickle migration mechanism. The lock import stays private.

## Source binding

One private `_SourceIdentity(owner, dependencies)` record serves six named
domains. Owner checks use `is`. An owner-only binding inspects no molecular
facts and computes no digest. A long operation can capture the exact required
domains once and recheck them before installation; no persistent digest cache
or general revision counter is stored.

| Domain | Current captured content |
|---|---|
| topology | ordered member UIDs and complete directed Bond endpoints |
| representation | topology, elemental/charge/H/aromatic fields, Bond representation/source, current stereo transport, legacy isotope and charge/spin inputs |
| geometry | topology and XYZ; independent of charge/source-label changes |
| cell | the six current cell parameters |
| constraints | topology, coordinate/Bond constraints and existing constrained angle/torsion entries; no view generation during capture |
| source_frames | topology and the existing ordered coordinate-frame array |

Until Stage7B, complete Bond order is active + hidden-metal + hidden-covalent;
partition changes may conservatively invalidate a capture. Until Stage7D the
old frame owner mixes roles, so this primitive does not establish final P1/P3
frame provenance. Explicit spin declarations take precedence; current selected
legacy-frame spin remains a representation input where no declaration exists.
Energy/force result arrays do not enter the frame-coordinate identity.

The capture/check primitives hold the new lock. Existing raw writable-array
aliases and all legacy writers have not yet been replaced by the full commit
boundary, so this checkpoint does not claim complete concurrent snapshots or
automatic validity protection for existing scientific results.

## Numeric presence and producer provenance

The private table candidate additionally owns one uint32 presence mask per row
and a uint8 producer-code block aligned to the 19 fields. A fixed immutable
mapping names the actual default/explicit/raw-array/toolkit and existing-rule
producers. It is not a runtime registration API or a per-field record family.

Raw-array input marks every supplied slot present, including zero. Defaults
can remain absent. Toolkit imports can supply exact masks/codes. An explicitly
supplied source is independent of presence. Scalar or per-field presence/source
arguments are normalized before numeric installation; a failure restores
numeric values and metadata together, including the currently interrupted write.
Append, remap, take and copy retain metadata and buffer isolation. No UID map,
isotope duplication or shadow result-status owner is added to the table.

The candidate is still not installed on production Molecules. Its extra
metadata work is included explicitly in the private-table measurement and
cannot be presented as an old mechanism with a fabricated baseline.

## Evidence and remaining work

`evidence.json` binds the default suite, full affected
Python3.10–3.14 source selection, B20 native-binary preparation, failed attempts,
source/class census and actual performance records. An initial matrix exposed
missing matching graph binaries in the source checkout; four baseline wheels
were SHA-verified and their unchanged graph binaries extracted before rerun.
This is not a candidate installed-wheel/native rebuild.

Complete factual imports, row/presence/source ownership cutover, closed molecular
transactions, all writer migrations, dependency expansion/reads and integrated
performance acceptance remain required before Stage3 completes. Construction
failure's old partial registration is still characterized rather than declared
fixed. The later owner rename/no-duplication instruction resolves the old matrix
name: `atom_matrix` directly replaces it and migrates live consumers, with no
old-name alias. The approved batch method still awaits the closed commit path.
