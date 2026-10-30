# Stage5 hydrogen-default scientific and interface review

Status: design task `p20-s5-h-default-design-01`,2026-10-30. This record supplies
the scientific convention and implementation requirements requested by the
owner. Production and test changes belong to the Stage5 implementation owner;
this task made documentation changes only. Stage5/global acceptance remains
open. The older `int | None`/NaN proposal was not approved.

The owner's controlling instruction is: “明确设定依照设定；没有设定应该依照价键规则提供默认值。
（请高阶模型提供符合科学事实的默认规则）”. The concrete recommendation is an
integer property, protected explicit/source counts including zero, finite
ordinary valence defaults, and preserved public missing-slot zero display with
a separate existing strict missing-input boundary. The exact table and
algorithm are in [decisions.md](decisions.md). The bounded parent follow-up
selects preservation of the historical scalar/snapshot display, so no new
scalar/projection exception or optional/NaN return is proposed. Imported count
normalization remains separate and unapproved.

## Entry and scientific source coverage

The current main plan and all four normative companions were read in full
before substantive design; exact identities and bounded line ranges are in
[evidence.json](evidence.json). Stage4 owns its completed entry gates. Stage5
owns H extraction, coupled aromatic representation, lazy dependency/commit
semantics and the current consumer checks. Stage6 stereo splitting, Stage7
external-object/physical-carrier work, Plan21 and Stage13 are not entered.

Primary checks used OpenSMILES' explicit/default-H and aromatic conventions,
pinned RDKit `Release_2026_09_1` Atom/allowed-valence source, Open Babel3.1.1's
actual typical-valence implementation, Daylight SMILES and IUPAC's definition
of valence. Downloaded primary-source bytes and SHA256 identities are preserved
in the evidence. The finite table is a graph completion convention, not a
pKa/protonation/electronic-state solver. In particular older neutral N3/5 text
is not a reason to synthesize valence5 nitrogen automatically. Real bracket
hydrogen, explicit graph H and represented radicals must be retained.

## Actual owners and affected consumers

| Owner or consumer | Current source and relevant behaviour | Required boundary |
|---|---|---|
| Current count and source | `structure/_fields.py`, `_atom_table.py`, `_mutation.py`, `_commit.py`; field14, presence/source and integer scalar codec; `core._numeric_property` dispatch | One current count owner; protect explicit/source bytes and zero; preserve independent numeric snapshots. An absent H slot's legacy zero display does not create a current fact. |
| Local rule and preparation | `rules/chemistry/bonding/hydrogen.py`; `operations/chemistry._atom_implicit_proposal`, `_prepare_hydrogens`, `_require_implicit_hydrogens` | Replace one default heuristic; immutable local inputs; validate the complete source before one commit; no new cache/registry. |
| Charge/edge/radical dependencies | `_local_inputs`, `structure/_identity.py` `bonding_inputs`, `structure/_dependencies.py`; Bond kind/direction and current payload radical facts | Extend H-only tuple input; bind consumed payload fields and seed at `_commit_stereo`; no opaque payload hash. |
| Current versus historical transport | `stereo.AtomTransportRecord` / `HydrogenTransportState`, `rdconvert._extract_stereo_payload`, `_apply_atom_fact`, `get/set_stereo_payload` | Current radical/bracket/no-implicit fields are required completion inputs; immutable `source_hydrogens` is evidence, never a second current owner. |
| RDKit writer | `rdconvert` RDKit-import field14 assignment; field14=`GetNumImplicitHs()+GetNumExplicitHs()` | Toolkit count excludes graph H; exact source/current ledger exists. Connected target normalization is a separate public/encoding proposal. |
| Open Babel writer | `obconvert.obmol2mol`; field14=`GetImplicitHCount()`, original payload retained | Toolkit count excludes graph H; producer currently has no per-atom H source ledger. Do not claim a complete source-ledger conversion without implementing this writer. |
| H expansion | `operations/chemistry._add_atom_hydrogens` / `_add_hydrogens`, Atom/Molecule entry methods | Existing code subtracts graph H and then polar-site metal neighbours. For targets from the new bond-kind-aware inferred producer, expand target minus graph H and bypass the extra metal subtraction. Preserve supplied legacy source expansion/encoding as the characterized separate boundary. |
| Counts, charge and state descriptors | `core.sum_implicit_hydrogens`, `Atom.explicit_hydrogens`, `_default_molecule_charge`, `Atom` state construction | Their total/omitted interpretation must be declared in a connected normalization. Do not silently change a molecular charge heuristic with this default correction. |
| SMARTS | `search/smarts.py`, `search/_native/production.cpp`, native matcher/semantic packing | Current total-H predicate adds graph and raw H; H/X/v must consume one consistent omitted/total interpretation. Prepare once before matching. |
| Stereo virtual H | `stereo._gate6_*` graph projection/expansion | Current graph H plus raw count can double-count default-inferred targets. Retain explicit D/T atoms, current radical facts and implicit carrier identity. |
| Toolkit/native translation | `rdconvert.to_rdmol`, `obconvert.mol2obmol`, `_native/buffers.py`, `rdWrappers`/`obWrappers` buffer consumers | Existing payload/current-H states and fallback scalar packing must agree; no scalar None/NaN coercion or unreviewed ABI/schema change. |
| MCA/model/conformer working values | `AImodels/mca/featurizer.py`, `conformers/adapters._materialized_hydrogen_projection`, explicit task buffer projections | Preserve historical trained feature encodings/source values; the existing materialized-H working copy explicitly clears current omitted counts while keeping source evidence. |
| Copy/remap/restore | `core._install_molecule`, `_set_remapped_stereo_payload`, fact record merge/component/pickle paths; transaction rollback | Retained dependency state must match consumed radical/current-H input after install/remap. Rollback restores both payload and status/seeds. |

Names and line identities refer to the dirty source snapshot captured with the
probe, not a release claim. The production implementation owner can relocate
settled private helpers without changing this ownership requirement.

## Executed observations and their interpretation

The read-only probe compiled/executed the actual B20 Atom `get_valence` and
`_calc_implicit_hydrogens` AST bodies from commit
`e6082f5fd67cb1054726ddcf43db7f963f7091ed`, and invoked the current private H
rule. The separately labelled proposal arithmetic is not production code.
There were53 local observations and29 parsed RDKit reference SMILES. All seven
selected production-source hashes were identical before/after that run.

| Local represented case | Actual B20/current rule | Recommended default | Interpretation |
|---|---:|---:|---|
| Neutral N–O single | H4 / H4 | H2 | Correct unsupported nitrogen promotion; use neutral N valence3. |
| N+ with C single, O single, O double | H1 / H1 | H0 | Nitro N occupied valence4 already completes its q=+1 convention. |
| C, one radical and one C single | H3 / H3 | H2 | A represented radical consumes one valence vacancy; absent radical input defaults to0. |
| P four C singles | H0 / H0 | H1 | Multi-valence graph convention: choose5 above occupied4; supplied zero still wins. |
| P(=O)(O)O centre | H3 / H3 | H1 | Old oxygen-neighbour sum is not a suitable default H calculation. |
| P(=O)(O)(O)O centre | H4 / H4 | H0 | Occupied valence5; no extra H. |
| Real isolated H/H+/H- | IndexError / private None | H0 | Correct the missing-neighbour failure; do not make H2 automatically. |
| Fully explicit C–H4 | site target4 /4 | site target4 | RDKit omitted-H reference is0. This is a representation difference, not failed arithmetic. |
| Covalent Zn–C single at C | H4 / H4 | H3 | A represented covalent metal bond consumes C valence. N→Zn donor consumes no donor valence. |
| Known ZERO C–O versus UNKNOWN zero-order edge | H4 / H4 for both | finite4 versus private None | Zero-order knowledge differs from absent bond identity. |

A further12 RDKit reference observations cleared bracket H/no-implicit and
varied aromatic B/C/N charge. They support conservative no-addition for charged
aromatic C, H1 at neutral c with two aromatic neighbours, and H1 at ordinary
aromatic N+ when no H/no-implicit declaration is supplied. Charged aromatic
`[cH-]`/`[cH+]` and `[nH]` retain their declared H; the rule does not choose a
tautomer. This final boundary refines the unused charged-carbon entries in the
original arithmetic helper; the maintained decision table is authoritative.

## Count-coherence finding and bounded proposal

The ordinary-nonmetal legacy default is a site-total-H **target**; metal
inference has an explicit historical raw0 exception and real H a separate
H2/isolated-H rule. Toolkit sources are non-graph
counts; add-H subtracts graph H; SMARTS/stereo/transport may add the raw count to
a graph already containing H. Those paths cannot all be correct under one
unqualified raw-count interpretation. This is independent of the scalar's
integer return type.

For example a toolkit methane source with graph H1 and omitted count3 needs
three new H. Subtracting graph H from every raw source would add only two.
Conversely using the original source count3 forever as a live omitted count
will add H again after expansion. A source-code switch alone cannot track the
current physical inventory and is not a complete migration.

The separate target-preserving proposal translates ordinary-nonmetal toolkit
import inventory once to E+R, keeps original R in the immutable source ledger, and
makes all chemistry/transport consumers derive `max(target-E,0)`. Add-H is then
idempotent and removes no H because it double-subtracted metal neighbours.
Legacy model encoding must be maintained through an explicit task projection.
This changes imported public/raw field14 when E>0 and needs a connected writer/
consumer migration plus parent public/encoding review. It is **not approved**
by the owner's default-H instruction and is not hidden in the default patch.
The alternative of canonical literal omitted-H storage likewise requires an
explicit count-contract decision. Implement this round's corrected default,
integer reader and new-input invalidation while preserving toolkit/source/model
encodings. Scope target-minus-graph expansion to the new inferred targets so
the old additional polar-site metal subtraction does not apply twice. The
inherited explicit-H mismatch is characterized rather than promoted into a
new universal-closure prerequisite for the present correction.

## Dependency and public effects still requiring evidence

The retained public contract is `int`, with the existing setter. Z0/absent
element, a needed unknown neighbour, UNKNOWN Bond kind or needed dative
direction leaves H absent/UNRESOLVED. Its scalar and permissive H snapshot slot
keep the historical zero placeholder. The pure rule returns private None;
rendering zero does not create H presence/source or mark CURRENT. Strict H
consumers retain `_require_implicit_hydrogens`' existing `ValueError`. A real
inferred/supplied zero is CURRENT with presence; the display alone deliberately
cannot distinguish it from absence. Existing CONFLICT handling remains.

A2 was re-read at `stages/stage01/contracts/decisions.md:18-40,120-132`:
NaN is explicitly for optional selected-result absence and staged with
A8/Stage7C; construction retains placeholders and separate presence. It does
not require H NaN or a new scalar exception. This preservation route needs no
new public decision. The earlier recommendation of scalar/permissive-export
ValueError is superseded, not reported as owner-approved. Any future visible
unknown-H return/exception change still needs Section16.4 review. No global
count normalization follows from this bounded choice.

Implementation must prevent chemistry from using the display as evidence of
resolved H. Existing strict entry points are already named for add-H,
RDKit/native/SMARTS, working copies and MCA serialization. The direct scalar
read inventory includes count/charge/state descriptors, assembly predicates
and action helpers, stereo graph expansion, xTB checks and state-facts
extraction. Their required-group guards must be established at their task
boundary when they require current H; an unchecked `int(...)`/truthiness on
legacy display is not sufficient. This is existing consumer-coverage work,
not a new public API or imported-count migration. No production execution of
the new display implementation is claimed by this design review.

Radical-aware completion adds a real input dependency. The existing
`bonding_inputs` source binds formal-charge/Bond kinds/directions but excludes
`_stereo_payload`; `_commit_stereo` currently replaces that payload without
marking H dirty. The exact small extraction/seed/atomicity requirements are in
Section6 of the decision. The final rule consumes all three current payload
inputs, so `(uid, radical, current_bracket_H, current_no_implicit)` is mandatory;
radical-only binding is not an alternative. Pure descriptors, isotope/source-ledger-only edits
must not trigger H recalculation. A complete-record install retaining dependency
state must also schedule changed consumed values. No Stage6 carrier migration
or new DTO is needed.

## Scoped reviewer clarifications: metal exception and bracket inventory

The primary contract review task `p20-s5-h-contract-review-704283-11` identified
a real wording conflict: “site-total” was stated globally while the actual
algorithm/probe retained metal0. Keep B20 metal0 and explicitly scope the
total-target interpretation to ordinary nonmetal inference. Molecule automatic
add-H skips metals; `sum_implicit_hydrogens` includes heavy atoms; SMARTS
TotalHEq adds graph H and raw H; generic native/RDKit projections transport
raw field14. Changing Zn–graph-H from raw0 to1 would change those model/native/
SMARTS values and can double-count its represented H. Preserve graph hydrides
in the graph and current bracket facts in transport without treating metal
field14 as a physical H inventory. An explicitly supplied metal H count still
wins. This is a declared legacy exception, not permission to normalize imports.

The second finding is also accepted: the exact algorithm uses radicals,
current bracket H and current no-implicit, so the four-entry UID binding is
mandatory. The previous radical-only wording is removed.

A further materialization risk is resolved in Section7 of the decision: if
ordinary inferred T=E+J+extra is expanded but J survives, the next topology-dirty
preparation counts the same H as both graph H and current bracket H. The
minimal fix uses the existing current-H payload replacement, setting current
implicit/bracket counts0 while preserving current no-implicit. It shares the
outer graph transaction, retains T/producer, and never edits source_hydrogens.
E'=T and J'=0 then make repeat automatic add-H idempotent after preparation.
Do not force no-implicit True: that would silently disable later defaults.

Current-source inspection found Molecule.add_hydrogens covered by
`_transactional_legacy(topology=True)`, while Atom.add_hydrogen reaches
`_add_atom_hydrogens` and then `_update_topology` without equivalent outer
coverage. Both paths must encompass graph/current-H/final topology/dependencies
in one transaction; the existing `_commit_stereo` journal supplies payload
rollback when joined. The conformer materialized-H helper establishes the
current-state replacement pattern but also clears raw counts and forces its
working-copy policy, so the whole helper is not the selected public add-H
implementation. Existing immutable transport types and one shared update
implementation suffice; no Stage6 schema migration is involved.

This was a bounded source/contract refinement. The imported-count proposal and
accepted permissive-display/strict-read distinction remain unchanged. The
reviewer checked inline consistency rather than the downloaded external
primary-source bytes; it supplies no new scientific-source, execution or stage
approval claim. Exact result artifact and inspected source identities are in
the evidence. Required metal/current-J/idempotence/rollback tests remain the
production owner's checks.

## Actual commands, results and remaining gates

- `python /tmp/hotpot-p20-hydrogen-default-design-20261030/probe.py` failed with
  `ModuleNotFoundError: No module named 'hotpot.cheminfo.rules'`; script execution
  without checkout `PYTHONPATH` found the installed package. The failure is
  retained in the evidence and is not counted as coverage.
- `PYTHONPATH=/home/zhangzhiyuan/hotpot PYTHONDONTWRITEBYTECODE=1 python /tmp/hotpot-p20-hydrogen-default-design-20261030/probe.py`
  exited0:53 local observations,29 RDKit references,0.6454152259975672s,
  selected production bytes unchanged. Environment: Python3.11.16,
  RDKit2026.09.1,numpy2.4.6,Open Babel distribution3.2.1,pytest9.1.1.
- The recorded inline Python aromatic-charge reference command exited0 and
  produced12 conventional RDKit observations; its exact script, result and
  artifact hash are in the evidence.
- JSON parsing, document-local link checks, finite-table/contract assertions,
  and assigned-file `git diff --check` are recorded separately after writing.
  They verify documentation integrity, not production chemistry.

No pytest/default suite, scientific assignment suite, native rebuild, wheel,
five-ABI matrix or performance benchmark was executed by this design task.
Root/other-worker results are not attributed here. Production owner must check
source/explicit zero, charges, radicals, aromatic `[nH]`/charged c, real H/H2,
isotopes, metal/dative/ZERO/UNKNOWN distinctions, relevant retry/one-time lazy
preparation, missing-slot display versus strict-consumer rejection without a
zero fact, exact numeric snapshots and whole-proposal rollback. The inherited
explicit-H mismatch remains a recorded limitation; universal closure and its
separate normalization are not prerequisites for this bounded
correction. Root retains the existing Stage5 technical/SMARTS/transport/stereo/model,
default-suite, source/debug/current installed-wheel ABI, packaging/native,
scientific and comparable performance acceptance.
