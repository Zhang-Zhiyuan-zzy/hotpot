# Stage5 hydrogen defaults and read contract

Status: bounded design recommendation for `p20-s5-h-default-design-01`,
2026-10-30. The owner's latest instruction requires automatic valence-based
H defaults whenever no count was supplied. It authorizes the scientific
correction to the old H heuristic. Production implementation and validation
are pending. The earlier `Atom.implicit_hydrogens: int | None` proposal was
not approved and is not adopted here.

## 1. Decisions and exact public effects

Retain the public integer property and existing setter:

```python
Atom.implicit_hydrogens: int
```

A present explicit/source value wins, including zero. The getter and explicit
assignment retain their existing integer codec; independent numeric snapshots
retain supplied float64 bytes, including characterized legacy values outside
the ordinary chemical-count domain. The default rule never rewrites those
values. A named explicit representation-conversion operation is distinct from
default inference and must preserve its source ledger.

An omitted count on a known ordinary element is not missing chemical input.
Prepare the one local pure rule lazily, commit a finite count with its existing
producer code, and reuse it until a relevant input changes. Do not require a
unique physical protonation state, tautomer or electronic state. No model,
external program, registry, additional DTO or public method is introduced.

Z0/absent element, an element-unknown neighbour needed by the rule, an UNKNOWN
bond, or a DATIVE without the required donor/acceptor direction is different:
the rule returns private None, H presence remains absent, and the group stays
UNRESOLVED. Preserve the existing public integer and permissive numeric-display
boundary: an absent H slot displays the historical0 placeholder in the scalar
and field14 of `Atom.attrs`/`Molecule.atom_matrix`. This display makes no H
proposal, creates no source/presence, and never marks the group CURRENT. Do not
return a stale inferred count as current when rendering that absent slot; a
previous derived value's unused bytes are not an authoritative H declaration.

Strict consumers continue through `_require_implicit_hydrogens`: genuinely
unavailable H raises its existing ValueError before chemistry, native/toolkit
packing or a strict model projection executes. They must not infer readiness
from an int/float display value. CONFLICT retains its existing RuntimeError
boundary and is not turned into the missing-slot placeholder. Repeated
unresolved display or strict reads do not rerun unchanged input. Explicitly
supplying a count still wins, even on Z0; an elemental consumer can separately
require a real element.

| H case | Internal fact state | Public scalar / permissive H slot | Strict H consumer |
|---|---|---|---|
| Supplied H0 | present, supplied source, CURRENT | integer0 / supplied float64 zero |0 |
| Ordinary absent C H | inferred4, producer present, CURRENT | integer4 / float64 four |4 |
| Real H/H+/H- without graph H | inferred0, producer present, CURRENT | integer0 / float64 zero |0 |
| Required element/Bond identity absent | no H fact, UNRESOLVED | integer0 / float64 zero, display placeholder only | existing ValueError |

A2's accepted NaN policy applies to **optional selected-result slots** and is
explicitly staged with A8/Stage7C partial-charge ownership. Implicit H is a P2
representation count, not that selected-result absence contract. Stage1 A2 also
retains legacy constructor placeholders with separate presence. Therefore this
preserved H display does not extend NaN to H, change A2, approve scalar None,
or require a new public decision. The companion UNRESOLVED no-zero-fill rule
continues at rule/commit/current-consumer boundaries: no unknown H becomes a
zero fact. Existing numeric display alone cannot distinguish absent H from a
real zero; preserve that limitation rather than invent a public validity API.

The earlier proposal to move ValueError onto the scalar and permissive numeric
projection was not approved and is superseded by this preservation route.
Section16.4 still applies if a future change makes unknown H visibly distinct
through None, NaN or a new scalar/projection exception. This round selects none
of those public changes and has no remaining H-read owner-decision gate.

## 2. Default valences

The table is a finite **hydrogen-completion convention**, not oxidation-state
inference or a declaration that these are all physically possible valences.
Select the lowest listed valence at least as large as occupied covalent
valence plus the represented radical-electron count. With no such valence,
add no unspecified H; retain all supplied bonds/charges/radicals and report
no chemical uniqueness error. Never repair charge or promote neutral N to
valence5 because an oxygen neighbour exists.

| Element(s) | q=-2 | q=-1 | q=0 | q=+1 | q=+2 |
|---|---|---|---|---|---|
| B | — | 4 | 3 | 2 | — |
| C | — | 3 | 4 | 3 | — |
| N | — | 2 | 3 | 4 | — |
| O | 0 | 1 | 2 | 3 | — |
| Si | — | 3,5 | 4 | 3 | — |
| P, As | — | 2,4,6 | 3,5 | 4 | — |
| S, Se, Te | 0 | 1,3,5 | 2,4,6 | 3,5 | 4 |
| F, Cl, Br, I, At | — | 0 | 1 | 2 | — |

A dash, another known element, or another known charge means **no automatic
unspecified-H addition**, not an unavailable element and not a valence error.
This deliberately narrow table does not synthesize metal hydrides, noble-gas
hydrides or rare hypervalent halogen H. Explicit/source values remain usable.
Hotpot's existing elemental classification is unchanged; notably its current
`is_metal` includes Ge/Sb/Po. This design does not silently reclassify them.

Absent formal charge uses the existing neutral model q=0; absent radical input
uses the usual nonradical model r=0. These are documented modelling defaults,
not new explicit charge/spin facts. A molecular charge/spin declaration does
not assign local charge or radical counts. Isotope changes no H valence: graph
D/T atoms have Z1, remain explicit, and retain their isotope; inferred new H
has the ordinary unspecified isotope convention.

For represented radicals, automatic filling by the valence deficit is supported
for B/C/N/O/Si and the listed halogens. Other known radical patterns, including
P/S/As/Se/Te radicals, receive no extra unspecified H. Aromatic radical N also
receives no unspecified H. This is a conservative default policy; existing
radical declarations are never cleared to obtain a closed-shell answer.

## 3. Bond and aromatic contributions

Use the already represented kind, order, aromatic flag and direction. Numeric
order alone is insufficient. Contributions at a nonmetal completion site are:

| Kind/context | Contribution to occupied valence |
|---|---:|
| SINGLE / DOUBLE / TRIPLE | 1 / 2 / 3, including a represented covalent metal–nonmetal bond |
| ZERO | 0; a known zero-order edge is not an unknown bond |
| DATIVE at donor | 0 |
| DATIVE at nonmetal acceptor | 1 |
| UNKNOWN, or DATIVE without direction | missing required input; do not invent its order/role |
| aromatic edge in aromatic completion | one sigma connection; use the aromatic completion table below |

A metal centre retains the historical inferred raw H0, even with represented
graph hydrides or current bracket-H transport facts. This is an explicit
metal-specific encoding exception, not a site-total H inventory. Graph/bracket
facts remain present in their existing owners; supplied scalar/source counts
still win. Molecule automatic add-H already skips metals. Changing metal raw0
to graph-H count would change model/native/SMARTS inputs and is not authorized
by this default correction. A nonmetal donor N→Zn is completed
on donor valence, whereas a represented covalent Zn–C single consumes one C
valence. This intentionally corrects the old blanket exclusion of all metal
neighbours from H completion. It does not redefine `Bond.is_covalent`, alter
the force-field algorithm, or silently reinterpret a supplied ligand count.
An operation requesting the metal-free ligand skeleton uses that explicit
selection. For a target produced by the new default rule, H expansion uses
`max(Htarget-E,0)` and bypasses the old polar-site metal subtraction: the rule
has already consumed the actual covalent/dative role. The implementation can
scope this correction to the inferred producer. Supplied legacy source counts
and their existing expansion interpretation are a separate characterized
boundary; do not silently relabel them or use their historical source ledger
as a live residual count.

For aromatic completion use these sigma targets:

| Atom/charge | Sigma target for automatic H |
|---|---:|
| C(0), Si(0), B(-1) | 3 |
| N(+1), nonradical | 3 |
| C(-1/+1); neutral/negative N; O/P/S/As/Se/Te; other aromatic atoms | no unspecified H |

Two aromatic C neighbours leave H1; an additional substituent or fused third
connection leaves H0. Charged aromatic C and neutral/negative `n` default
to no additional H; a supplied `[cH-]`/`[cH+]` or `[nH]` is retained. Bracket H or graph H
is explicit representation and is preserved. An aromatic `n` with two single
numeric Kekule edges is not by itself permission to choose the `[nH]` tautomer;
its supplied H or explicit graph H determines that role. These conventions do
not change the existing aromatic-perception/Kekule algorithm or its known
limitations. Carbon/B/Si radical electrons consume the corresponding sigma
H vacancy; other unsupported aromatic radical cases add no unspecified H.

## 4. Exact local algorithm and count meaning

For **ordinary nonmetal inference**, retain the historical **site-total-H
target**, rather than a literal count absent from the graph. This is not a
universal definition of raw field14: metal inference is the explicit legacy0
exception, and a real H atom retains the separately stated H2/isolated-H rule.
Supplied toolkit/user fields retain their existing encoding. Let E be graph H
neighbours, J current unmaterialized bracket H, B occupied graph valence under
Section3, and r represented radicals. E and J must describe non-overlapping
inventories. Bracket H consumes J valence as well as represented H. Historical
source-ledger counts are not current facts. The automatic materialization
operation maintains this non-overlap through the current-H update in Section7.

The pure rule can consume immutable tuples through the existing private
operation; it needs no new runtime data class. Its result is an integer target
or private None for genuinely missing inputs. Apply this exact order:

1. The operation retains a supplied scalar/source count before inference,
   including on a metal. Source-only history does not prohibit current default
   inference and does not become a current count.
2. If the element is absent, return private None. A metal returns legacy raw0
   regardless of graph/bracket H; its represented H facts are retained outside
   that non-total field. A real H adds no further H: the historical H–H site
   count is E (H2 gives1 each), isolated H/H+/H- gives0. This fixes B20's
   isolated-H indexing failure without assuming H2.
3. At an ordinary nonmetal site, current `no_implicit=True` prohibits extra
   unspecified H and returns Htarget=E+J. This requires enough neighbour
   identity to count E; it does not guess an unknown H inventory. The metal/H
   exceptions above take precedence. Existing current bracket H is retained;
   this branch does not choose a different protonation.
4. Otherwise, missing required local element/kind/direction yields private
   None. An unsupported known charge/element/radical convention yields
   Htarget=E+J, adding no unspecified H.
5. Sum B including graph-H bonds. For aromatic targets sum sigma contributions
   as in Section3; otherwise use ordinary integer valence. Include J in
   occupied valence. Select V for B+J+r. If none exists, extra=0; otherwise
   extra=V-(B+J+r). Return ordinary-nonmetal Htarget=E+J+extra.
6. Source/explicit counts are not recalculated. The operation checks its whole
   captured source and commits the affected proposal at the existing atomic
   boundary. The exceptions in step2 are not normalized to that formula.

The raw scalar, attrs/matrix field14 and legacy model field order are not
silently normalized to literal non-graph H. The actual existing producers and
consumers disagree about that field, so the complete conversion issue below
must be kept visible. A return-type choice cannot solve it.

## 5. Inherited toolkit inconsistency and separate normalization proposal

Ordinary-nonmetal B20/default Htarget includes graph H, while current RDKit imports store
`GetNumImplicitHs()+GetNumExplicitHs()` and Open Babel imports store
`GetImplicitHCount()`, both excluding graph H. Add-H subtracts graph H, but
SMARTS total-H/connectivity/valence and stereo virtual-H expansion add the raw
field to a graph that already includes those H. These are real incompatible
interpretations, not alternative scientific truths.

A source-code-only residual helper is **insufficient**: a source count R that
was non-graph H at import cannot remain a live non-graph count after those H
are expanded into graph atoms. Repeated add-H would duplicate them. Conversely,
subtracting current graph H from every toolkit source count under-adds for an
initial partially explicit source (E1+R3 methane needs three additional H).
Do not publish that tempting helper as a complete solution.

The smallest coherent normalization proposal is:

- retain the ordinary-nonmetal inferred site-total target and the explicit
  metal/H encoding exceptions; no uniform metal total-H field is selected;
- at ordinary-nonmetal toolkit import translate physical inventory to
  Htarget=E+R, preserve exact original R in the existing immutable source
  ledger, and never treat that ledger as a second current count owner;
- derive non-graph H for those normalized nonmetal targets at every affected
  chemistry/transport consumer as `max(Htarget-E,0)`, with current
  bracket/no-implicit decomposition maintained
  by the existing transport projection; keep raw user declarations unchanged;
- add-H expands precisely that residual once and retains Htarget; the next
  call adds none. Do not subtract metal neighbours again;
- preserve legacy model input encoding through its explicit task projection,
  binding the exact historical-source counts when required; do not silently
  replace trained features with newly normalized counts.

This is a **proposal**, not a claim that the owner approved changing imported
public/raw numeric values. It changes imported field14 when graph H already
exists and needs the parent to apply the public/encoding review before that
connected migration. No new public type, DTO or source ledger schema is
required, but Open Babel's missing ledger producer and every consumer must
migrate together. Literal non-graph canonical storage is an alternative with
more direct public/count and add-H changes; it is not selected implicitly.

The directly implementable boundary for this round is the single corrected
pure default, retained integer/placeholder display and strict missing-input boundary, complete
bindings and invalidation for its consumed inputs, and residual expansion for
its own inferred targets. Keep toolkit imports, supplied raw counts and model
feature projections on their existing encoding. No import normalization,
source-ledger writer addition or universal consumer reinterpretation is
required to implement this bounded correction.

The inherited toolkit/default inconsistency is not enlarged into a new
prerequisite for that patch. Existing Stage5 consumer/atomicity/default tests
still apply; universal hydrogen representation closure is not claimed. The
separate normalization proposal would require public/encoding review and
connected validation when selected. The owning review preserves the concrete
counterexamples so this limitation is not mistaken for closure.

## 6. Minimal input binding and invalidation

`operations.chemistry._local_inputs` currently supplies only element,
neighbour elements, numeric orders and the Atom aromatic flag. Its existing
valence/formal-charge/order-sum callers need not change. Add one private
hydrogen-only tuple extraction (or explicit arguments at those two H call
sites) carrying the actual local formal charge, Bond kind, edge aromatic flag,
site-relative dative role, represented radicals, and current bracket/no-implicit
facts needed above. It is an ephemeral tuple, not a new carrier or schema.

The existing `bonding_inputs` identity already binds formal-charge columns and
Bond kind/order/flag/direction. It does **not** bind `_stereo_payload`, where
`AtomTransportRecord.radical_electrons` lives. Bind only the effective inputs
actually consumed from that payload, in current Atom order:

```text
(atom_uid, effective_radical_count, effective_current_bracket_H, effective_no_implicit)
```

This four-entry tuple is mandatory for the final algorithm: radical, current
bracket H and current no-implicit all affect it. A radical-only tuple does not
implement this contract and is not an allowed minimal alternative. Missing
radical/bracket/no-implicit declarations use `0/0/False` as effective inputs.
Do not bind isotope, descriptors, source namespaces, historical
`source_hydrogens`, display annotations or the opaque payload as a whole.
`current_hydrogens.implicit_hydrogens` is transport output, not another input
count owner for this pure rule. The final rule and binding have exactly the
same consumed input set.

At the existing `_commit_stereo` write boundary, compare old/new effective
inputs by the current Atom UID before replacing the payload. On a changed
consumed value, seed the existing `IMPLICIT_H` group for those UIDs; its current
downstream fanout already includes `AROMATIC_REPRESENTATION`. Use the existing
transaction journal for payload, seeds and group status. Do not infer radicals
from molecular spin, invoke chemistry at the setter, create a fifth dependency
group, or dirty H for source-ledger/descriptor/isotope-only edits. The complete
record installer must seed changed consumed inputs when it retains supplied
dependency state; initialization without retained state must schedule the
existing groups. Copy/remap must retain matching UID/input/status bindings.
The current direct installer and rollback payload assignments are part of this
writer audit; rollback restores the old binding and its old dependency state.

Capture/check this closed binding around the complete H proposal, including a
no-proposal/UNRESOLVED attempt. A later row changing a consumed radical or dative
role must invalidate the whole proposal before any H values commit. Preserve
explicit/source H even when this queues their status for revalidation.

Required bounded checks: Z0 displays scalar/snapshot H0 while H presence stays
absent and status remains UNRESOLVED, strict consumption raises existing
ValueError without committing a count, and unchanged reads execute the missing
rule once. After a relevant element/Bond fact is supplied the rule retries;
explicit H0 and real isolated H0 are CURRENT with presence, not placeholders.
For C with one C single, H3 changes to H2 when its current radical count becomes1;
a supplied H0 remains0; current bracket or
no-implicit changes retry unchanged UNRESOLVED/default inputs; source-only or
stereo-descriptor-only edits neither change the H binding nor rerun its rule;
a consumed-input change during a multirow proposal rejects/rolls back all rows.
These are implementation requirements, not tests reported as run here.

## 7. Scoped inferred bracket-H materialization

The new conversion branch applies only when automatic add-H (`num=None`) uses
a CURRENT count with presence and the current
`ATOM_SOURCE_CODES['implicit_hydrogens']` producer. It does not recognize an
inferred producer from a historical ledger. Metals are excluded from Molecule
automatic addition and keep raw0; their graph hydrides are not erased or
recounted in field14. Supplied/toolkit counts and explicit `num` retain their
characterized paths; this is not imported-count normalization.

For an ordinary nonmetal, prepare/capture `(T,E,J,producer,consumed inputs)`
with T=E+J+extra and A=max(T-E,0)=J+extra. Validate the complete pre-mutation
source before adding any H, and materialize all A as ordinary graph H.
Capture every selected row's current producer/eligibility and J before the
first graph write. Later rows use that prepared immutable tuple even though
the operation's own earlier writes have queued dirty seeds; rechecking live
CURRENT after each row must not accidentally route later rows down the legacy
source path. This does not bypass validation of the original complete proposal.
In the same outer factual transaction, replace only each affected existing
`AtomTransportRecord.current_hydrogens` through the existing `set_stereo_payload`
/ `_commit_stereo` path with:

```text
implicit_hydrogens = 0
bracket_explicit_hydrogens = 0
no_implicit = the previous current value (including None/False/True)
```

This consumes the current unmaterialized bracket inventory into the newly
created graph H. Do not copy bracket counts from `source_hydrogens` back into
current state, edit that immutable ledger, clear isotopes/radicals, change a
supplied raw count, or force no-implicit True. If no current-H state exists and
J=0, no transport record needs to be synthesized just to hold zero. Use the
existing replacement/commit pattern, not the whole conformer working-copy
projection: that separate projection also clears raw counts and freezes its
own implicit-addition policy.

Retain raw inferred T and its producer. Let ordinary topology and the changed
four-entry payload input enqueue the existing dependency groups; do not mark H
CURRENT merely because a numeric target was retained. The next named
preparation checks the new binding. After success E'=E+A=T and J'=0. Under
unchanged other facts the rule returns T again (including no-implicit and
unsupported-known-state cases), so the next automatic add-H residual is0.
Keeping J after graph creation would instead count the same H twice and is
not a permitted implementation. Preserving the previous no-implicit value
also permits the existing default policy to operate after a later graph edit.

The graph creation, current-H replacement, inferred target/source/presence,
source checks, topology finalization and dependency seeds/status share one
outer `_FactTransaction`. `_commit_stereo` joins it and journals the old
payload. Molecule.add_hydrogens already has the topology transaction wrapper;
Atom.add_hydrogen currently lacks it and must receive equivalent outer
coverage including its final `_update_topology`. Keep one add-H calculation/
current-H update implementation across both entry paths. A later-row failure
restores all published graph/fact/payload/dependency state; normal burned-UID
high-water semantics remain as specified by the existing transaction contract.
Do not recapture a changed source in place of validating the original complete
proposal. Current-H commit after expected graph writes stays inside the same
owner lock/transaction.

Required implementation evidence: E0/J1/no-implicit True aromatic n gives
T1, creates one graph H, clears current J, preserves source J1, and repeats
with0 additions; E0/J1/no-implicit False C gives T4, creates4 graph H, clears
current J, retains False, and repeats with0 additions after relevant dirty
expansion. Include an initial graph-H case, both Atom and Molecule entry paths,
a batch failure after an earlier row's graph/current-H update, source-ledger
byte equality, no forced no-implicit change, and metal graph-H raw0 unchanged.
These are required assertions, not production tests executed by this design
owner. Existing broader stereo/toolkit total-versus-omitted limitations remain
as characterized in Section5; no carrier schema or source-ledger migration is
introduced here.

## 8. Public-source basis and limits

The finite table follows documented graph defaults, with a conservative
selection among conventions. OpenSMILES' older neutral N3/5 list is not copied
into automatic N filling: Open Babel's actual N branch explicitly avoids
rounding up to5, and the pinned RDKit table lists neutral N3. P3/5 and S2/4/6
remain useful representation conventions. No d-orbital expansion explanation,
pKa prediction, tautomer search, oxidation-state assignment or metal-hydride
inference is implied.

- [OpenSMILES §§3.1.2 Hydrogens,3.1.5 Organic Subset,3.5 Aromaticity,3.6 More about Hydrogen](https://raw.githubusercontent.com/opensmiles/OpenSMILES/master/opensmiles.html#hydrogens).
- [Daylight SMILES §3.2](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html).
- [RDKit Book, Valence calculation and allowed valences](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences).
- [RDKit Release_2026_09_1 Atom.cpp, calculateExplicitValence/calculateImplicitValence](https://raw.githubusercontent.com/rdkit/rdkit/Release_2026_09_1/Code/GraphMol/Atom.cpp).
- [RDKit same-release atomic_data.cpp valence list](https://raw.githubusercontent.com/rdkit/rdkit/Release_2026_09_1/Code/GraphMol/atomic_data.cpp).
- [Open Babel3.1.1 GetTypicalValence and OBAtomAssignTypicalImplicitHydrogens](https://raw.githubusercontent.com/openbabel/openbabel/openbabel-3-1-1/src/obfunctions.cpp).
- [IUPAC Gold Book, valence](https://goldbook.iupac.org/terms/view/V06588).

These are representations and models. They do not determine a unique physical
molecule from a partial graph. In particular known unusual valence, multiple
reasonable protonation states and scientific model uncertainty do not become
exceptions. Missing elemental/bond identity and broken transaction mappings
remain distinguishable technical conditions.
