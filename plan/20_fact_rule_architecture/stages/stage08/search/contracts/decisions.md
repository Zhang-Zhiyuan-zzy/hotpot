# Stage8C search decisions

## C1 — Canonical BondKind contract prerequisite

Root selects an early extraction of the existing seven-value `BondKind` strEnum
from core.py to `structure/bond_kind.py`. Lower chemical rules must not import
a concrete carrier owner to name a bond category. This is the already planned
structure-owned type, with one definition and no new members, class, scientific
rule or representation. Core and ordinary top-level exports refer to that one
object; existing coercion and kind/order mapping implementations stay unchanged.
Four SMARTS/IR imports now name the canonical contract, retiring their exact
legacy-core debt. This does not move the concrete Bond or other carriers early.

New pickles identify the actual physical class owner. The captured current
core-path enum pickles must still load through core's existing export during
this prerequisite; Stage9 retains the separately owned old-core deletion and
historical-asset decision. Add no compatibility module, alias class or duplicate
implementation. `hotpot.BondKind` remains the ordinary public entry point and
is identical to the object used by Molecule/Bond construction and copy.

Validation uses the actual pre-edit CP311 enum/pickle capture at c13970d, a
contract-leaf import without the execution tree, cold imports/pickle identity,
real bond construction/copy, existing SMARTS/native matching and structural
bonding/transport tests. An exact class-AST comparison checks this mechanical
type move; chemical meaning is judged by the actual behavior tests. Native
source/binary/schema/feature encoding does not change or rebuild for C1.

The full parser/recursive-query/chemical-predicate/RI/packing/public-operation
separation remains under read-only inventory. The current molecule-keyed
recursive SMARTS cache still requires its owning Stage8C treatment. No full
Stage8C, installed successor, native-package or global acceptance follows
from this prerequisite. Root will record the connected split before authoring.


## C2 — Connected search cutover selected by root

The C1 commit5800e6e is pushed. The completed Python/native inventories show
that moving mixed smarts.py/native.py/production.cpp wholesale would violate
ownership. Root selects the following connected extraction under Sections6/9.2.
It preserves current public names,48signature rows, matching/chemistry/order,
all24MCA rules and the sole shared CSR Graph. No new data-record class, graph,
backend selector, compatibility implementation or scientific algorithm is added.

| Responsibility | Canonical owner |
|---|---|
| Existing SmartsSemantics, Atom/Bond operation/field enums and two frozen typed-expression records | structure/substructure.py; retain fields/defaults/identity equality/repr. Existing carrier-based convenience methods and callable records are thin function-local delegates to named search operations. No carrier import in the contract. |
| Lexer, errors, grammar/Boolean/H/coordination-extension parsing and chemical comparison rules | rules/chemistry/substructure/; lexer uses existing reference element symbols. Pure parser returns operation-local atom kwargs/map/aromatic hints and edge/attrs tuples. Reuse existing expression family; do not create a retained second graph/DTO. |
| Searcher/Query/Substructure/Hits/Hit and SMARTS allocation/recursive execution | operations/search/; single existing class definitions and graph state machine. Bind recursive text expressions to the existing _RecursiveEvaluator at this operation boundary. |
| Native query encoding/requirements from mutable Query objects | adapters/structure/substructure_query.py; this is transport lowering, not a second chemistry IR or execution path. Preserve the current dict/tuple schema, recursive compilation and exact-owner type recognition. |
| Prepared molecule and exact-topology-label extraction | adapters/structure/substructure.py; no locks/materialization/writes. Operation retains lock and required H/aromatic preparation in its original ordering. |
| _NativeTarget/_NativeMappingSource, loader and Python/native translation/bindings | adapters/native/substructure/; one extension hotpot.cheminfo.adapters.native.substructure._search_native, unchanged basename and private Python classes. matching.py owns the Python wrappers. |
| Sole RI matcher, candidate domains and exact integer-label automorphism | rules/mathematics/graph_matching/_native/; no chemistry/Python/toolkit dependencies, same shared Graph source. |
| Native typed atom/bond comparison and task-local recursive predicate evaluation | rules/chemistry/substructure/_native/; prepared scalar structs and evaluator have no Python/adapter/carrier dependency. Python translation and host/callable leaves remain at the adapter. |

### Preserved lazy semantics and preparation

SmartsSemantics public Atom methods retain their exact signatures. Named
operations select/read the original rings/bonds/H/order values, then lower pure
rules calculate from explicit values. FULL_GRAPH valence retains the existing
integer all-order sum (not a new raw floating sum); ligand valence retains its
existing kind-filtered numeric sum. Native field-selective preparation remains
H then aromatic only when required. Ring descriptors retain Relevant Cycles and
FULL_GRAPH/LIGAND_SKELETON scope. No new guard, missing-H zero or inferred fact.

Existing typed expression __call__ enters a named operation. The pure comparison
rule receives an explicit lazy scalar/tuple supplier; only reached leaves obtain
values. This preserves Boolean short circuit and direct Probe/custom-object
behavior without passing a live carrier into a rule. Compiler recognition still
requires the exact canonical expression class; opaque user callables retain whole
Atom/Bond handles. Host sets/general containers and attribute preflight also
remain lazy Python leaves: the former shorthand that only callables reenter
Python is corrected. Missing-attribute checks precede reached callbacks; their
ordering, identity and exceptions remain unchanged. No user callback executes
at compile time.

The pure parser emits an unbound recursive body in the existing expression
value field. Operations bind it to _RecursiveEvaluator(smarts,semantics), whose
only retained inputs are smarts,semantics,substructure. Delete WeakKeyDictionary
and molecular signature/result functions. Direct recursive calls prepare/pack
and execute a fresh task, preserving full anchored membership; native recursive
programs retain their existing per-problem/target cache only. No global, weak,
Molecule-keyed or ambient task cache replaces it. Mutable Query inputs continue
to be snapshotted when compiled; MCA keeps legitimate immutable compiled-query
and one-target reuse.

### Native source and lifetime split

Keep the current TargetAtom/TargetBond/SemanticRecord/typed-expression family
once in the chemical native header, containing pure NativeValue fields only.
NativeTarget::Data owns separate aligned host-field/object vectors at the
adapter; each supplied field exists in its native or host slot, with original
Python handle identity retained. Move conversion, preflight, host/callable
evaluation and NativeCompatibility to the adapter. Native expression rules
request recursive execution through an explicit native callable bound once by
the adapter; its result cache stays within that query/target lifetime.

Move the existing NativeAutomorphism/ExactLabelCompatibility to mathematics;
its target becomes an aliasing shared_ptr<const Graph> that retains the original
NativeTarget lifetime, not a copied Graph. The binding converts the existing
NativeTarget constructor argument to this ownership handle; private Python
constructor/behavior remains unchanged. Move the ordinary compatibility-domain
loop to mathematics and retain existing RI traversal, anchors, zero-limit
probing, deterministic order and GIL release. Add a standalone matcher/orbit
source test alongside existing Graph/cycle component tests. Do not move or
redefine the shared Graph during this search cutover.

### Migration and validation boundary

Delete tracked old search sources/stub and migrate every active producer,
consumer, test, doc/example, setup/MANIFEST/CI and measurement import together.
No shim/reexport package, class/module spoof, fallback loader or old/new runtime
selector. Current public class metadata names actual canonical owners; cold
class identity and supported enum pickle checks do not imply arbitrary Query
instance serialization. The three existing core carrier imports move only from
search/search.py to operations/search/search.py until Stage9, using their exact
existing ratchet slots. Other retired imports do not supply debt growth credit.

Pre-cutover actual baseline: MCA24molecules/43sites and SMARTS serial/concurrent
equality with complete5040embeddings. Plan16 quick initially failed in its
obsolete subgraph_monomorphisms_iter measurement helper. Root authorized only
its actual iter_mappings query-to-target correction and key rename; successor
baseline completed all22checks with phase_breakdown.available=true. Original
failure remains, and passed MCA/SMARTS work was not repeated. These are source
baselines, not scientific/performance acceptance.

Root releases authoring only now, with disjoint Python/native writers and root
connected migrations. No candidate product imports, pytest, builds, performance
or model execution before static review and a fresh source freeze. Required
validation is full31ordinary Stage8C files plus connected MCA/stereo/dataset/
assembly/lexer and new owner/no-cache/native-source contracts, CP311 fresh search
extension and installed wheel outside checkout, conformance, deterministic
embedding/callback/native parity, actual business workflows and same-work
search/site/SMARTS throughput. Retain all failures and adverse usable costs.
Other versions, clean SDK/sdist/global scientific/performance gates retain their
own later obligations; no whole Stage8C gate closes at authoring or focused pass.


### C2 connected assembly observation — P20-D024

The first frozen focused run passed245cases and exposed one new positive
assembly oracle: imported methane/oxygen AtomLink graft retains H4/2 although
its C–O bond and requested site are correct; expected methanol H is3/1. A direct
link+calculation reproduces the source-dependent behavior without search; all
four relevant business function bodies are unchanged from5800e6e. Inferred-H
new-Atom sources give3/1. Root keeps the failing chemical oracle as a strict
expected failure and adds the passing inferred-H assembly route. This is an
identified Stage8E transformation obligation, not a search matcher defect or an
accepted chemical result. No blanket H rewrite, hidden correction or weakened
oracle is authorized. Source/installed C2 reports must retain that limitation
and count; Stage8E must resolve it before dependent assembly acceptance.


### C2 root CP311 acceptance and cost disposition

Root accepts the connected search migration within the C2 CP311 source and
outside-checkout installed boundary. Full source1159nodes yield1158passed plus
the strict P20-D024 expected failure, with49additional passed subtests. The
original stale stereo roster failure is corrected only for5already accepted
Stage8A classes and rerun once; earlier collection/import failures remain.
Installed954nodes yield953passed/the same expected failure plus49subtests.
The205source-only architecture/CI nodes explain the exact ordered collection
difference. Actual MCA4cases pass independently at both source and installed;
3standalone native harnesses pass. The private wheel has1fresh search binary
and5reused accepted binaries. Clean all-native/SDK/sdist, other versions and
later/global gates remain open.

All3benchmark recipes completed once with their original declared work.
Search22checks/5040complete embeddings, MCA24molecules/43ordered sites and
SMARTS24/120/720/5040embeddings with serial/concurrent agreement remain. Root
accepts observed usable cost: search medians0.926–1.042×, MCA1.021× and
SMARTS0.950–1.252×; all233adverse observations remain. These single historical
and candidate launches establish observations, not statistical significance or
global parity. P20-D025 retains consolidated final integrated cost review.
P20-D024 still requires the Stage8E imported-H transformation correction before
dependent assembly acceptance; no chemical3/1oracle is weakened.

Final delivery removes one trailing space in the migrated assembly Markdown
example and extra EOF blank lines in6new Python files first checked at staging.
All executable ASTs and test/native bytes retain validated content; the private
wheel keeps those original blank lines. Exact frozen-byte reconstruction and
AST equivalence qualify these formatting-only deltas without rerunning business
validation. Stage8C scope closes at this checkpoint, with the limitations
above and the evidence chain in validation/c2/checkpoint.json and report.md.
