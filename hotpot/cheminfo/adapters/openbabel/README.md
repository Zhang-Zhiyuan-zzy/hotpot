# Open Babel factual conversion and stereo transport

Conversion projects prepared Hotpot facts into a fresh Open Babel molecule or
imports an independently supplied native molecule through the factual builder.
It does not assign stereochemistry from coordinates, sanitize authoritative
facts, or use a text-format/toolkit intermediary. Public row mappings use
Open Babel's one-based atom indices; native stereo references use persistent
`GetId()` values and are mapped separately.

## Stereo codec

The existing immutable `StereoRecord` carries native Open Babel data under
`openbabel-stereo/1`. This is a versioned transport codec, not a quality label or
another stored stereo owner. Records are sparse: only native stereo data actually
present are imported, without invoking perception. Their source remains Open
Babel. Atom isotope/H/source facts retain the existing `openbabel` convention.

`source_code` and the raw `descriptor` use these normalized forms:

| Code | Anchor and ordered carriers | Meaning |
| --- | --- | --- |
| `TH/CW/s`, `TH/CCW/s`, `TH/UNKNOWN/s` | Atom centre; `[from, ref1, ref2, ref3]` | Tetrahedral config in ViewFrom orientation; winding is the named native enum. |
| `CT/U/s` | Ordered double-bond endpoints; four native ShapeU references, two per side | Cis/trans configuration; pairs 0/2 and 1/3 are opposite. |
| `SP/U/s` | Atom centre; four native ShapeU references | Square-planar configuration. |

`s` is `1` for specified and `0` for unspecified. Native `GetConfig()` provides
the canonical view/shape and equivalent ordered references. Explicit unknown
winding and unspecified records remain distinguishable in Hotpot and native
round trips; neither creates an assigned stereoisomer.

Atom references are transported as Hotpot rows. Open Babel's unsigned
`ImplicitRef` is a virtual H when the corresponding atom has a non-graph H;
otherwise the native H/lone-pair convention is retained as a lone-pair carrier.
Square-planar implicit references have no registered translation and fail
locally. A missing/unsupported reference is never replaced with an atom. No
CIP designation is inferred by this codec; raw source descriptors are not
silently promoted to Hotpot Level-2 descriptors.

## Projection rules

Hotpot-to-Open-Babel projection writes native stereo data after graph, aromatic,
H and isotope preparation, then marks chirality perceived. Thus serialization
does not substitute a coordinate-derived stereoisomer for the declarations or
invent stereo on an unspecified graph. Native TH/CT/SP configs are restored
directly. Existing RDKit ordered tetrahedral and alkene records translate by
registered carrier-order and cis/trans rules without importing RDKit or using
RDKit as a conversion intermediary. Three graph carriers use the native virtual
fourth reference under the source convention; graph H and isotopic H keep their
actual identities. Unsupported namespaces, codes, enhanced groups and other
stereo families fail at the affected adapter boundary instead of losing data.

Open-Babel-origin TH records project to RDKit using actual target neighbour order
and permutation parity. CT records choose explicit carriers from their ordered
sides and set the relative cis/trans value plus native neighbour directions.
Unknown/unspecified declarations do not become assigned configurations. Native
SP records remain available for native OB round trips; the RDKit projection
rejects them because no approved square-planar permutation codec is supplied.
These are adapter support limits, not claims that a molecule cannot exist.

## Ownership and verification

Projection leaves canonical records, Atom/Bond facts and source objects intact.
Copy, component, merge, reorder, pickle and H materialization use the existing
source-independent carrier remapping and canonical storage. No new DTO, stored
payload, ABI field, model feature or public interface is introduced. Lossless
native-config comparisons apply to registered transport; chemistry comparisons
use independent opposite-isomer fixtures, allowed ordinary-H projection and
explicit source orders rather than canonical SMILES spelling alone.

The rules follow the native config contracts in Open Babel's stereochemistry
API. Tests and execution records identify actual toolkit versions and observed
coverage; this document does not establish unexecuted support or acceptance.
