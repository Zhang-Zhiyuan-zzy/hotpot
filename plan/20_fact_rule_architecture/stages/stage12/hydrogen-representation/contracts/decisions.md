# Stage12B — Current hydrogen inventory

Root decision for P20-D026, 2026-11-04. Implementation is released at
`9d9c6564d8c771d7b25ab3028a26b21d5f336bca`; acceptance requires the evidence below.
This concretely selects Stage5 §5's normalization proposal and supersedes only
its deferred import/consumer/materialization boundaries. Default valences,
aromatic/radical rules, integer API, missing-value display and strict readiness
remain unchanged. No new public symbol, class, numeric column or native ABI.

## B1 — Count owner and allowed public differences

Let E be the number of graph-H neighbours. For ordinary nonmetal heavy atoms,
existing scalar/field14 is the site target T, including represented graph H.
The non-graph count is R=max(T−E,0); represented atoms are never counted twice.
A supplied T below E does not delete graph atoms or prove chemical impossibility.
Preserve supplied bytes, presence and producer; a named graph operation remains
separate from inference. Real H/H2 contribute their existing graph vertices and
no additional non-graph H; retain their historical raw count. Inferred metals
retain raw0; supplied metal counts keep their existing non-graph scope. Ordinary
Molecule automatic H completion continues to exclude metals and H vertices.

New RDKit/Open Babel ordinary-nonmetal imports translate R to T=E+R. RDKit
retains exact original I/J/noImplicit in current and immutable source transport.
Open Babel gains an existing AtomTransportRecord source ledger (namespace
`openbabel`, original implicit count; no invented RDKit decomposition), including
its available isotope. The scalar remains the only target owner; history is
never substituted for a live count. Other supplied producer values are not
silently inferred from provenance or normalized after the fact.

Imported methane with E0/E1/E4 therefore has field14=4/4/4 rather than4/3/0.
Methanol site targets are3/1 across graph representations. Generic/PyG tensors
expose these intentional field14 changes; do not keep a second legacy tensor
truth. All other columns, order, dtypes and supplied nonintegral float64 bytes
retain their existing contracts. The actual preentry fixtures show no CBond
consumed-array or MCA complete-feature change from this normalization alone;
custom field14-consuming checkpoints are uncharacterized and need regenerated
inputs/revalidation, not a fabricated compatibility guarantee.

## B2 — Current declarations and toolkit maps

Current bracket J and noImplicit remain declared representation inputs. Current
implicit I is checked transport output, not another input to the default rule.
A supplied scalar does not silently erase J, noImplicit, isotope or radicals.
Physical consumers reject the reproduced contradictory case J>R with
`HYDROGEN.DECLARATION_CONFLICT` and the row/target/graph/bracket values. The scalar
and permissive numeric displays still expose their supplied fact. Do not make an
arbitrary stored I block default calculation or materialization.

RDKit preserves registered current decomposition and its existing implicit-count
verification. It must also reproduce R=I+J after the real property-cache update;
otherwise issue an adapter-scoped inventory failure, retaining the facts. An
isotope-only/source-only record cannot suppress count projection. Without a
current RDKit decomposition, encode R as explicit non-graph H with noImplicit
True; this prevents RDKit from generating extra H on real H, metals or an
undercoordinated supplied target. Open Babel receives exactly R through its
implicit-H setter after graph construction and the available isotope. No
perception call may silently replace those prepared counts.

Native packing uses the same prepared R. The existing conformer-only current
state I0/J0/noImplicitTrue under canonical/Open Babel namespaces projects to
the same fixed logical0; it does not invent an RDKit source convention. Retain the existing namespace-free
IMPLICIT-only logical-count transport form, now explicitly fixed-count: the
RDKit native builder sets explicit R and noImplicit True. Its canonical
isotope/source-only form must work too; original RDKit current decomposition
retains strict verification. The registered Open Babel source-only map is
allowed; arbitrary namespaces remain unsupported. Preserve the C API layout and
all native source/current attribution. The affected RDKit native binary must be
rebuilt and actually exercised before its capability is accepted.

## B3 — Materialization and deletion

Automatic ordinary-nonmetal completion expands R once, without the historical
second metal-neighbour subtraction. Supplied O target2 beside Zn therefore
requests two H; the represented bond/charge and sanitizer's own limits remain.
All prepared rows are captured and checked before any write, and the complete
operation shares the existing outer factual transaction/rollback. Consume current
I/J into the graph for supplied/imported as well as inferred rows, preserving
source history, isotope, radicals and previous noImplicit. Clear both counts to0
when all R is materialized; repeat addition creates none. Remap a materialized
virtual stereo-H carrier to its actual new graph H in the same commit, preserving
ordered carrier position and descriptor.

Explicit num is a declared graph edit. At an ordinary nonmetal, consume at most
that many current bracket/implicit H (bracket first), preserving T and source
history; excess new graph H remains an explicit graph edit, not an excuse to
invent a negative residual. num=0 is a no-op. Metal/H exceptional paths retain
their separate graph-edit scope, with no reinterpretation of supplied raw counts.

remove_hydrogens remains graph-vertex deletion, not a universal inverse or a
source-history reconstruction. Invalidate retained current I output on affected
anchors, keeping J/noImplicit and supplied T; inferred targets are recomputed by
the unchanged default rule. No all-cases physical-conservation claim follows.
If retained declarations cannot encode the resulting target in a toolkit, retain
the exact local failure rather than changing them to force success. Polar-only
removal retains its declared topology scope.

## B4 — Connected consumers and persistence

All physical consumers use the same pure non-graph projection, including SMARTS
H/X/v (Python and native target packing), Gate6 virtual adjacency, and
scientific-state nuclear/electron/bond-valence counting. The historical molecular
implicit-H sum remains the sum of raw targets, as required by the existing
default-charge heuristic (completed graph-H count minus target sum). Default
charge always completes its working copy, including partially materialized
inputs; it preserves the original graph, target/source facts and transport.
HydrogenInventory keeps its existing raw target field and graph-count fields;
its readers compute residuals instead of suppressing all H whenever E>0. Keep
None for unavailable counts and existing conflict boundaries. No new inference
algorithm or change of electron/charge/spin rules is authorized. Composition
coverage and charge derivation retain uncovered non-graph protons whenever R>0;
xTB metadata counts their nuclei/electrons, while its graph-only input adapter
rejects any remaining R (including supplied metal H) rather than omitting it.

Assembly now reads physical site total E+R uniformly, without producer-based
import reinterpretation. Its named transformations consume caps and write their
product targets through the existing atomic owner, preserving source ledgers
and its existing isotope/stereo/capacity/integrity restrictions. T<E alone does
not erase graph caps. Present current decomposition still must agree with R;
missing checked output I is not reinterpreted as a new authoritative zero.

Conformer working-copy projection keeps its selected scalar0/noImplicit True
policy and stereo remapping. Copy/components/merge and current-version pickle
preserve current factual bytes and source history through their sole owners.
The private MoleculeRecord pickle generation changes5→6, retaining24fields;
old generation5 is rejected explicitly. No producer/history-based migration of
ambiguous old materialized objects is safe or authorized. This is a private
persistence cut, not a new public schema or old-pickle compatibility route.

## B5 — Evidence and completion

Baseline evidence is the two executable preentry directories indexed by the
entry record. Tests use independently counted methane E0/E1/E4, methanol,
aromatic/bracket sites, radicals, charged atoms, supplied zero, D/T, real H/H2,
metal graph hydrides, and the actual scalar/bracket conflicts. Expected physical
inventories, permitted target/tensor changes, exact source preservation and
rejection conditions precede candidate execution. Include repeated/partial
materialization, remove/add declared behavior, atomic late rollback, lifecycle,
SMARTS, stereo, nuclear/electron counts, actual model feature inputs and the
actual registered CBond/MCA plus FF/conformer routes where affected.

Run focused CP311 source first, then the complete affected source/installed
selection outside the checkout with a bound rebuilt RDKit native and actual
wheel/sdist. Retain original failures. No all-version, model-generalization,
universal chemical truth or global performance-parity claim follows. Measure a
small representative completed workload for usable cost. Acceptance requires
root review of actual inventories, coupled state and all affected evidence;
D026, whole Stage12, final versions/release, Plan12 and global gates stay open
until their own closure records.
