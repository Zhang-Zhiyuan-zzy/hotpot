# Stage8E assembly decisions

## Entry and fixed boundary

Stage8D owning CP311 gate is accepted and pushed as1f348a2. Root Sol/max owns
this stage and all exact transformation/public decisions. Current goal, main9.2,
Stage8E/Decision20, Stage1 asset review, Stage5 count conventions and Stage7E
current/source transport remain authoritative. Preentry reports are inventory,
not executed chemistry or acceptance. Preserve one topology, one chemistry
preparation/commit owner and existing result/mapping/rollback contracts.

## E1 — One operation and explicit public surface

Select the concrete Decision20 route: Fragment, EdgeShoulder, AtomLink,
AlkylGraft, BondAdding, AtomReplace, RingWedge and AssembleFactory are the eight
assembly classes explicitly available through the existing chemistry/root
facades. The selected advanced operation path is
`hotpot.cheminfo.operations.assembly`; it also exports alkyl_generator,
atom_link_atom_action, shoulder_bond_action, bond_order_add, atom_replace and
ring_wedge. Helpers/typing/module names do not leak through wildcard exports.
Move each existing implementation once and delete the old mol_assemble package
in the connected cutover; no old-path alias or second implementation. Update
actual tests/examples/docs and facade consumers. Broader carrier facade work
remains Stage9/12.

Delete only the incomplete chemistry mol_statistics.ComplexStatistics and its
facade export, after the recorded absence of a direct consumer. The separate
plugins.ccdc_api.statistics.ComplexStatistics is retained.

The active26-row FragTemplete.json alone becomes
operations/assembly/data/fragment_template.json, byte-for-byte, loaded through
importlib.resources and included in wheel/sdist. README/SVG documentation follows
the operation. Distinct unused legacy asset schemas remain Stage11-governed;
never concatenate them into another assembly default.

## E2 — H transformations and action integrity

P20-D024 is a required chemical correction: methane/water graft must consume
one H per attached site and yield methanol H3/1 with charges0/0 and one C–O
single bond, preserving the original inputs. Generic calc_implicit_hydrogens
intentionally preserves supplied counts; changing it globally or forcing all
rows to an inferred value is not a solution. Transformation changes belong at
participating product sites, inside the existing whole-action _FactTransaction.
Unrelated authoritative H and immutable source-H ledgers remain intact; current
bracket/no-implicit/graph H and affected stereo must follow the actual declared
transformation. No source-enum/record/cache is added merely to label an action.

Before authoring, characterize the existing five actions and alkyl enumeration
against independent simple chemical oracles, including supplied/imported/
inferred and explicit-H graphs. Inferred field14 is site-total; toolkit fields
exclude graph H and their complete normalization remains an unapproved separate
proposal. Do not infer a universal residual solely from producer codes or refill
current H from source history. Root will record exact consumption/scaffold/
stereo decisions from the evidence before dependent implementation.

## E3 — Factory, documentation and validation

Repair demonstrated local defects in the selected existing workflow (loader
return, batch iteration, worker completion/error/timeout cleanup and complete
small enumeration) within an attributable route. No new stochastic algorithm or
silent unknown-error fallback is authorized. Existing inert random/seed/weight
controls must be documented honestly or receive an explicit later disposition;
mode names alone are not implemented behavior. Keep failures visible and
independent sample attribution stable.

Documentation must distinguish executable examples from schema illustrations
and match actual signatures. Gate includes exact exports, default resource
loading, six configured assemblers/custom actions/finite enumeration, real H
counts/charges/adjacency, input isolation, whole-action rollback and source/
installed CP311 behavior. Reuse valid unchanged evidence; full native rebuild/
SDK/sdist/multi-version/global final gates remain separately open. No authoring
or runtime acceptance follows from this entry record alone.

## E4 — Factory execution and rooted alkyl enumeration release

Actual baseline attempt01 proves the ExStrct AtomReplace missing return and
MolBatch duplication/empty-batch defects. Corrected attempt02 binds1501
source/test/dependency files and records the existing1,1,2,3 rooted alkyl counts
and incorrect capped numeric H. Original failed checker captures are retained.

Keep factory signatures and ordinary set[str] results. Correct iteration, loader
returns, empty input, relative catch paths and periodic save counters/cadence.
Preserve serial/full-set versus parallel/new-frontier execution and strict
max_running early stopping; None disables that optional bound. Existing mode,
seed and sample_weights remain inert configuration, documented without claiming
stochastic sampling. No new public mode or result envelope is introduced.

A worker must either publish its complete batch or fail visibly. Unknown errors
are transported/rethrown with original exception type where pickle permits and
original chained traceback text. Cross-process frame identity is not promised.
An unpicklable error must retain its original traceback in an explicit transport
failure, never become empty success. Abnormal exit/missing completion is a
batch-attributed execution error. Timeout=0 disables the bound; a real timeout
raises TimeoutError identifying epoch/batch/input SMILES/time bound and completed
work. Preserve already obtained results through catch_path when configured; do
not return an apparently complete subset after omitting a timed-out batch.
Always join/reap owned children and close owned channels/progress bars on normal
completion, cap, error, timeout, start failure or interruption. Never terminate
unrelated active children. A private tuple channel is sufficient; no new DTO.

Enumerate rooted carbon trees directly, with attachment Atom0 preserved. Start
from one inferred, capped carbon and grow only sites with degree<4. Deduplicate
by the recursively sorted rooted-tree connectivity, not unrooted SMILES or an
unmarked sacrificial anchor. Every returned length n fragment is the capped
alkane CnH(2n+2), with rooted counts1,1,2,4 at n1–4; a link consumes one cap H
at the retained root. Keep the existing defaultdict[int,list[AlkylGraft]] return.
This corrects demonstrated missing attachment isomers and stale supplied H; it
adds neither a stochastic algorithm nor a persistent second topology index.

Release the canonical filesystem/public/resource migration and bounded factory/
generator authoring under these decisions. Dependent H/scaffold action authoring
remains held until E5 records its exact reaction domain and source convention.

## E5 — Declared action-local H and scaffold transformations

Root selects the following bounded reaction semantics after corrected baseline
attempt02 and the focused H-inventory consultation. No global import/add-H/H-rule
normalization, model feature rewrite, source enum or new record is authorized.

**Input inventory.** Capture strict current H, the raw count/producer, graph-H
identities/isotopes and all current transport occurrences before action writes.
Ordinary inferred and explicit/user site-total rows mean T, with non-graph
N=T-E. Initial RDKit/OpenBabel imported rows mean non-graph R, with T=E+R.
Use the registered current decomposition when present and require its distinct
implicit/bracket components to agree with that input representation. This is an
action contract for coherent inputs, not a universal producer-code residual
helper. Source-H remains immutable history. Negative/nonintegral inventories,
contradictory decompositions or insufficient H fail locally. Arbitrary legacy
imported-materialized histories have lost overlap information: do not recover
it by min/max or source history. Observable overfull registered inventories fail
under the selected cap-replacement valence convention; observationally identical
histories cannot be distinguished and receive no recovery guarantee. An explicit
user site-total declaration establishes the intended count. Other source scopes
without a registered count interpretation fail at the action consumer.

**Product count owner.** Each participating changed product site receives the
explicit source1 **site-total** count T' through the existing factual commit.
This deliberate local representation of a newly transformed product avoids
falsely attributing a new count to the toolkit and makes subsequent actions
unambiguous. It changes neither imported inputs nor unrelated count rows. Update
all existing current-H occurrences to the residual N'=T'-E', retaining the prior
no_implicit and assigning bracket J' only as the actual transformation requires;
current implicit output is N'-J'. Preserve occurrence order, isotope/radical/
namespace and exact source-H. Never synthesize a source history or clear
no_implicit silently. A retained no_implicit with nonzero residual must use an
explicit bracket representation; the action's declared projection may materialize
that residual as bracket H without inventing additional H.

**Cap consumption.** Link adds one SINGLE bond and consumes exactly one H at
each capped site; bond+1 consumes one per endpoint and accepts only existing
nonaromatic SINGLE/DOUBLE bonds. Prefer ordinary non-graph H, consuming the
current implicit part before bracket H, then graph H. Preserve graph isotopes;
choose ordinary graph H first. Equivalent remaining isotope choices may use
stable input order, but choices yielding different isotope products fail with a
precise ambiguity rather than inventing isotope selectivity. Existing positive
count/adjacency contracts remain necessary reaction-domain checks. No change
to generic Atom.link_with is needed.

**Scaffold/element completion.** Element replacement changes only the selected
parent element; retain its charge, isotope, radical and source ledger. Recompute
only that site's product H under the existing Stage5 rule and the final heavy
connectivity. Remove only excess capped graph H; current bracket inventory is
reduced where required by the new element capacity, never filled from history.
Retain no_implicit: it permits retained graph/bracket H but no automatic extra H.
The ordinary neutral oracles are methane C->N = NH3 and terminal ethane C->O =
methanol. No tautomer, charge or radical choice is added.

Shoulder deletes the two selected parent endpoints and their owned graph H,
transfers only outside heavy bonds to the corresponding fragment anchors with
bond kind/order/direction/source metadata preserved, and completes only the two
new anchor H inventories. Fragment caps are not added to transferred parent caps.
Deletion includes isotope facts on the explicitly replaced sites/caps. Preserve
existing aromatic restoration; butane central edge plus ethene gives but-2-ene
H3/1/1/3 and C4H8. Ring-wedge **retains the original parent terminal Atom**, its
external heavy bond and source ledger, merges the ring fragment, transfers the
fragment anchor's ring-neighbour bonds to that parent, then deletes only the
displaced fragment anchor and its graph H. Complete the retained site's H;
ethane plus cyclopropane gives methylcyclopropane C4H8 with retained-site H1.
No detached-parent reinsertion or replace_atom return-value assumption remains.

**Stereo and atomicity.** These actions make no retention/inversion or E/Z
claim. Reject affected specified or uninterpreted stereo (anchor or carrier
touches changed/deleted sites) until a separately declared transformation exists;
ordinary RDKit CHI_UNSPECIFIED/STEREONONE/STEREOANY are unspecified. Preserve
unaffected records via the existing remapper. Affected isotope ambiguity and
unsupported stereo fail locally, not through an arbitrary configuration choice.

Each direct action, including H preparation, writes, merge/removal/aromatic work
and postconditions, has one outer _FactTransaction. Capture a fragment working
copy before acquiring the parent transaction; prepare only working copies and
leave original fragment inputs untouched. Fragment.graft prepares/searches a
frame copy and applies actions on copies, preserving the original frame's raw
facts/status/seeds. Every exception/interrupt restores the full parent action.
Local diagnostics identify action/site/reason with ASSEMBLY.HYDROGEN_INVENTORY,
ASSEMBLY.HYDROGEN_UNAVAILABLE, ASSEMBLY.ISOTOPE_AMBIGUITY or
ASSEMBLY.STEREO_TRANSFORMATION_UNDEFINED; no generic failure envelope.

The known generic inferred+graph RDKit H double-count projection and ambiguous
legacy supplied add-H encoding remain separate Stage5/12 consumer work. Exact
source/installed action facts must be checked directly; SMILES or an external
export alone is not the H oracle. No global H representation closure follows.
E5 now releases the connected action/Fragment implementation and chemical tests.

E5 implementation clarification: a declared neutral aromatic [nH] cap is valid
although Stage5 correctly infers no *unspecified* H on neutral aromatic N. Its
action-local upper capacity uses the same ordinary N sigma-valence convention
for existing caps; it never creates aromatic-N H or changes the pure default.
The explicit [nH] substitution fixture checks consumption with source history
retained. Lossless outside-bond transfer also preserves the raw constraint/id
projection, not only its boolean display, alongside kind/order/direction/source.

E5 ring clarification: the retained parent Atom adopts the displaced ring
anchor's aromatic flag through the existing aromatic-restoration owner, alongside
the transferred ring bonds. This is the declared scaffold substitution, not a
new aromaticity rule or a rewrite of unrelated parent sites. The benzene-to-
toluene-scaffold fixture checks the retained identity and source-H ledger.

E5 runtime correction: source continuation02 proves that restoring the adopted
ring flag as a derived aromaticity assignment lets subsequent H preparation
re-infer it from the previous terminal-site history. Commit the retained site's
adopted flag as an explicit source1 new-product fact through the existing atomic
writer before completing H. Keep other sites and source-H history unchanged.
The focused probe demonstrates aromatic survival through completion, lazy reads
and repeated H preparation; injected failure after the new commit restores the
whole parent. This corrects the declared transformation, with no global aromatic
rule change or status bypass.

The same actual seam is reproduced for benzene shoulder replacement with ethene:
H completion clears two newly adopted anchor flags and rejects a valid inventory.
Commit only replacement-anchor flags that differ from their incoming fragment
flags as explicit product facts before either H write. Unchanged and unrelated
source flags remain intact. The new fixture requires benzene adjacency, all six
aromatic atoms/bonds, H1 at every carbon, source-H preservation and input isolation.
