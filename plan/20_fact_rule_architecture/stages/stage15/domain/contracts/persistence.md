# Stage15E — portable current facts and CIF entry contract

Root decision 2026-11-05, following15D2b CP311 source/installed acceptance.
Reuse retained complete-plan/companion alignment; reread effective goal,15A–D
contracts,review15E/6,actual record/builder/cell/group/capture and connected
I/O/Stage14 owners. This is the settled implementation of15A's accepted
Crystal.from_cif(source,*,block=None) and Crystal.to_cif() signatures.
No external chemistry/symmetry/toolkit API and no guessed connectivity.

## Sole owners and codec

The existing immutable nine-field hotpot.crystal/1 remains the only Crystal
record. A private _crystal_codec serializes it to inspectable JSON primitives;
exact Fraction entries use rational strings, operations use exact xyz, source
and active group contexts remain distinct. Decode exclusively through existing
SpaceGroup and _canonical_crystal_record constructors. No second algebra,
retained external crystal object, arbitrary Python constructor or pickle loader.
This portable cell codec is shared with later14D, not a second schema owner.

CIF standard fields carry lengths/standard angles, explicit current operators,
site labels/types/fractional coordinates/occupancy/disorder and actual supplied
Bond pairs/types/image codes. A closed _hotpot_state_json (hotpot.cif/1) stores
otherwise unavailable current Hotpot facts: full oriented record and source
binding/maps, raw current Atom presence/producer/values, full Bond facts/metadata,
charge/spin declarations, Level1 stereo/isotope/H occurrences, constraints,
identifier, Wiberg observations and selected partial-charge record/markers.
Existing record buffers use hex; this is a versioned persistence extension,
not model features or a second retained fact owner. Pure Level1 stereo encoding
will be shared with14D instead of maintaining two parsers. Only the fixed known
record types can be reconstructed, never executable names/imports.

A native-CIF extension must agree with the actual standard site order/labels,
elements, coordinates, metric,group,occupancy/disorder,Bond pairs/types/images
within documented emitted numeric precision. Inconsistent edited dual data rejects
locally rather than silently restoring stale hidden facts. A bare standard CIF
can be imported by removing that extension and using the declared supported map.
No universal byte equality or historical hash requirement is a chemistry oracle.

## External CIF mapping and honest limits

Gemmi0.7.5-compatible is lazy and syntax-only: read_string,scalar/loop iteration,
as_string,quote,Document/Block/Loop construction and text output. No Gemmi Op,
SpaceGroup,Structure,neighbors,cell math or chemistry. The optional crystal extra
contains gemmi>=0.7.5,<0.8; absence produces an identified local dependency error
and leaves mathematical/molecular Crystal usable. Actual version is attributed
in validation. Do not modify the established CP311 environment; use the isolated
task dependency-site already installed from the pinned wheel.

Input is CIF text or a pathlib/os path; multiple blocks require exact selection.
Tag case and declared core/mmCIF spelling aliases are normalized. Missing/?/.
active facts remain absent; exact input text and selected block are archived in
Molecule.properties['cif_source'] (standalone cell-only Crystal has no Molecule,
so its archive is private _cif_source; rejected ValueError carries cif_source/
cif_block attributes), not treated as active facts. Nonowned source fields such
as ADPs/refinement/measurement metadata remain in that archive. No ADP or
occupancy filtering, molecular perception,charge inference or disorder collapse.

Parse finite cell/Cartesian geometry and numeric esds without inventing accuracy;
raw source keeps the esds. Fractional coordinates use the supplied full oriented
basis when available, otherwise the documented conventional standard cell basis.
Cartesian coordinates need their explicit CIF transform; an unsupported frame
rejects instead of assuming alignment. Atom types take precedence over label
component0. Support ordinary single-element/oxidation suffix and D/T isotope
codes under the CIF label convention; oxidation text is not formal charge.
Composite/dummy unrepresentable species reject precisely; explicit unknown
single-site element can remain absent. No implicit H/formal charge defaults.

Explicit complete operator lists are authoritative for the active group. A
verified exact reference match may bind its number/metadata; anonymous/changed
settings retain their actual operators without inventing reference-frame maps.
A reference label alone cannot fabricate an origin/setting. Number/symbol-only
inputs with no verified setting leave the active group unknown and emit the
precise projection limitation; no P1 fallback. Retain original operator IDs for
symmetry-code decoding, independent of canonical operation order.

Every supplied Bond maps the two actual source labels. A missing Bond loop gives
no Bonds. Missing periodic-image information givesNone. Explicit '.' has its
known no-operation meaning. Decode n_klm using the source operator ID and integer
translation. Bring both endpoints into the first endpoint's frame; only a
relative identity rotation with integer translation fits the current voltage
edge. Unknown images remain unknown; nontranslational relative operations,
self/parallel/repeated quotient edges or unmapped types reject the dependent
full projection with original source retained. Never silently collapse, enlarge
or synthesize sites. Conventional output codes are emitted when representable;
otherwise the exact Hotpot extension retains the full image and the unsupported
standard projection is reported. All-zero images remain explicit zeros.

Bare-CIF Bond type S/D/T/A maps to single/double/triple/aromatic; no type maps to
unknown with an unavailable numeric order, never a fabricated single/zero.
A supplied numeric Hotpot Bond kind/order/image field must agree with known
standard declarations. Native Hotpot extension preserves dative direction and
all semantic Bond facts without a toolkit. No geometry-based bond inference.

CIF exports current structural facts, not an entire live Python session.
Additional source/conformer observations, MCA attachments and arbitrary runtime
properties/extras are outside this format; discovered nonempty omitted material
gets a precise CRYSTAL-UNSUPPORTED-PROJECTION warning. Preserve available results
and input; never claim full session persistence. Current-version pickle remains
the complete existing lifecycle path. Standalone cell-only Crystal roundtrips
remain supported. Output has17significant digits; comparisons use1e-10Angstrom
geometry and existing relative/absolute metric tolerances, not SHA identity.

## Evidence and gate

Independent positive/negative/boundary fixtures: rotated/skew full matrix,
reference/anonymous/changed-origin/centred/screw settings, supplied equivalences,
occupancy/disorder, complete generated historical maps, known/unknown/nonzero
images, standard code ID order, isotope/H/stereo/dative,absent versus zero facts,
malformed references/schema and conflicting dual records. Actual11repository
CIFs retain the denominator and no-Bond cases; unsupported cases remain exact
failures. Source nonmutation,optional dependency isolation and forbidden toolkit
imports are checked. Actual source and outside-checkout installed CP311 checks,
precision and complete read/write costs precede bounded15E acceptance.
Then publish format closure to14D;15F/final integration/support stay open.
Three unsuccessful attempts on one issue stop that issue with dependency impact;
advance independent work. Root owns all integration/design/acceptance/Git.
