# Stage9B physical carrier cutover decisions

2026-11-04; parent `958bdd9b4757160aa57c4dc8eef91a8bb71587d5`.
Root accepted this design and subsequently released its bounded source and
installed validation through [entry_review.json](../entry_review.json). The
[checkpoint](../validation/checkpoint.json) records accepted CP311 carrier-cutover
evidence. Whole Stage9 and later/global acceptance remain separate gates.

## S9B-01 — Sole definitions and concrete initialization order

Move the current definitions from the captured core bytes, SHA256
`ab458bd9d9f7dc7e2d8d7dff6218b943001d6becc8d8d7364cdbe0796a1a3aed`,
without creating another delivered class or a compatibility route.

| Definition | Sole owner |
|---|---|
| Molecule | `structure/molecule.py` |
| Atom | `structure/atom.py` |
| Bond | `structure/bond.py` |
| MolBlock | existing `structure/_fields.py` |
| NotInSameMolecule, ObjNotInMolecule | `structure/errors.py` |
| AtomSeq | `views/sequence.py` |
| Angle, Torsion | `views/geometry.py` |
| JointRing, Ring | `views/topology.py`, in that definition order |
| AtomPair, AtomPairs | `views/pairs.py` |
| BondKind, Conformers | their existing `structure/bond_kind.py` and `structure/conformer.py` owners |

The fixed startup order is common schema/error/constraint/topology/state leaves,
Molecule, Atom, AtomSeq, Bond, remaining views, declaration completion, then the
ordinary public exports. Atom eagerly binds the completed Molecule; AtomSeq
binds the completed Atom/Molecule; Bond retains exactly `(AtomSeq, MolBlock)`
and its existing MRO/subclass behavior. `views/__init__.py` stays non-loading.
Actual Bond/Ring dispatch in AtomSeq and actual Atom/Bond dispatch in MolBlock
use local imports of those same classes. Class-name tests, virtual registration,
sys.modules workarounds, new bases and new persistent storage are outside scope.

`structure/__init__.py` exports only Molecule, Atom, Bond, BondKind and Crystal.
`cheminfo/__init__.py` retains its existing public export set, binding those names
to the concrete leaves. AtomPairs keeps its name. Root alone deletes core with
the final consumer in the same integrated business commit; temporary uncommitted
movement is unavailable for runtime until that integration passes review.

## S9B-02 — Exact annotation/default objects and finite namespace completion

Preserve eager annotations, quoted ForwardRefs, generic alias shapes, decorators,
default objects and wrapper chains. There is no blanket future-annotation
conversion, replacement sentinel, copied enum default or copied graph default.
The existing `cython.int`, `types.ArrayLike`, `np.ndarray[int]`, Segment/Cycle,
force-field convergence member and relevant-cycle integer retain their objects
and evaluation forms. Existing unresolved hints, including Conformers.add's
missing `torch`, are not repaired by importing unrelated optional systems.

The private `operations/_carrier_signatures.py` owns four fixed accessors:
`_geometry_types()`, `_geometry_reports()`, `_graph_defaults()` and
`_forcefield_contracts()`. They return the existing geometry-rule,
geometry-adapter, graph-rule and force-field-operation namespaces. They import
no concrete carrier and perform no calculation. Declaration owners call them
through small function-local import binders before their class definitions;
those bound namespaces serve annotations/defaults only. Scientific methods call
named operations separately. The existing strict structure/view direct-rule,
direct-adapter and eager-operation import gates remain binding.

`structure/_declarations.py` owns `_complete_type_names()`. One explicit call
from `structure/__init__.py`, after all concrete definitions complete and before
public import returns, performs these fixed namespace bindings:

| Defining module namespace | Existing concrete names completed |
|---|---|
| `structure.molecule` | Atom, Bond, Angle, Torsion, Ring, JointRing, AtomPairs |
| `structure.atom` | Bond |
| `views.sequence` | Bond |

Default `typing.get_type_hints()` uses a function's defining globals; imports
inside a method or under TYPE_CHECKING cannot replace this step. This is the
narrow accepted clarification to main Section6.1(6), separate from factual
pickle/state ownership. There is no registry, discoverable-class loop,
class rebinding or alternate type. The finite direct assignments do not install
descriptors or rerun native bootstrap.

The existing geometry and force-field packages still load their transitive
implementations when their eager declaration objects are obtained. Retaining
that existing behavior is explicit Stage9B scope; it does not demonstrate final
backend laziness or close Stage12's eager/lazy obligations.

## S9B-03 — Factual methods, once-only installation and operation boundaries

Molecule keeps the exact UID allocation, owner token, lock, factual record,
mutation, invalidation, dependency, source-H/stereo/frame/result and constraint
machinery. Its seven observation descriptors install once in molecule.py.
`_transactional_legacy` and its 17 current wrappers install once there, preserving
`wraps`, topology/dependency/frame/constraint capture flags, nested transaction
joining and original exceptions. The complete Stage9A link transactions remain
in their existing operations.

Atom owns the sole numeric descriptor installation and `_ATOM_FACT_ENGINE`.
MolBlock's first Atom write reaches that same bootstrap and Atom.__setattr__.
Native sources, lazy `_commit._fact_engine`, scalar/batch callbacks, error types
and dispatcher identities remain unchanged. The two existing writable Angle and
Torsion constraint properties install once in geometry views. AtomPair's accepted
Wiberg assignment projection remains intact; immutable membership does not remove
these projections.

The final direct-call cleanup adds only named operation delegates. The 13
`chemistry._element_*` delegates pass the same scalar atomic number to the
existing element/valence rules. Atom retains its existing
`np.asarray(..., dtype=int)` electron projection. The static explicit-input
Molecule._longest_path calls `topology_queries._explicit_longest_path` with the
same five arguments. Molecule cid/name/cas call descriptors._cid/_name/_cas,
which invoke the existing PubChem service only on property access and evaluate
`mol.smiles` once. Molecule bond/ring queries call
`geometry._molecule_bond_ring_piercing_state` and
`geometry._molecule_bond_ring_relations` with unchanged keywords/defaults.
Bond, Angle/Torsion and pair views likewise enter their named geometry
operations. Preparation, locks, short-circuit/read order, stale-view errors,
AtomPairs' explicit `_items` branch and returned values remain unchanged.
There is no new chemistry or mathematics algorithm.

Every moved relative import resolves to its original real owner. In particular,
Molecule.to_pyg_data still reaches `hotpot.plugins.PyG.data.utils` through the
approved adapter route; this cutover introduces no plugin redesign.

## S9B-04 — Current pickle identity and the approved historical hard cut

`structure/_state.py` owns the eight existing molecule/member/detached/unregistered
capture/restore helpers, including `_bond_member_at`. Restore factories import
concrete handle helpers locally. Raw restoration still precedes extras; the
existing `views/_source.py` remains the sole view source/restoration machinery.

Current-version pickle/deepcopy must preserve owner separation, one reconstructed
lock, UID/high-water state, live-member and endpoint identity, detached readonly
tables/status/seeds, constraints, pair observations, sparse facts, source/canonical
stereo transport, frame/ensemble selection, current/invalidated results and
metadata/annotation references. Root order and equal UIDs across different owners
must not merge graphs. The new callable paths are the real defining owners.
Conformers loses only its old-core `__module__` override; settled stereo public
module identities remain unchanged.

Historical core-path pickles are deliberately rejected. There is no core shim,
legacy global rewriting, fallback restore or compatibility subclass. Historical
fixture bytes remain evidence; tests must assert that intentional boundary and
canonical current-version roundtrips without silently regenerating the fixture.

## S9B-05 — Ownership, expected public effects and evidence gates

Root owns all remaining consumers, string/dynamic entries, tests, audit/allowlist,
parser binding, core deletion, release/validation records and Git. The handles
worker owns common/handle and view/geometry files. The Sol-max global partition
owns Molecule, state/declarations, signature accessors, chemistry/descriptors/
topology delegates, Conformers' override removal, structure/cheminfo exports,
this decision record and bounded main-plan clarification. Shared-file edits must
preserve the other owners' work.

Expected public/API differences are defining module paths, actual current-version
pickle callable paths, deliberate old-core pickle rejection, the actual parser
implementation digest and named declaration/operation ownership. Signatures,
permissions, defaults, classes/MRO, chemical results and factual lifecycle are
preserved. The root-owned parser binding must hash the real new parser-affecting
owners; previous currentness certificates cannot be forged.

The static discovery receipt is
`/tmp/hotpot_plan20/medium/stage9b-design-20261104/inventory.json`:
117 import statements, 151 bindings in 113 files, including 40 production and
73 test files. All 151 `(path, scope, target, local)` identities exactly matched
the active legacy-core allowlist at entry; line shifts do not change debt
identity. All 35 module-level functions, four simple-name bindings and 13 core
classes have explicit sole owners and disjoint file partitions. The 24 unrelated
Stage12 allowlist entries remain outside this retirement.

The preserved root contract baseline is
`/tmp/hotpot_plan20/long/stage9b-entry-20261104/contracts/baseline.json`, SHA256
`407cb8a8b7c13cb2957d23693fcd9c27e586c02a53a102d9699bf8917decdaae`.
Its receipt covers 14 classes, 702 runtime members, 311 AST functions and
33 representative default hint evaluations: 32 succeed, with the existing
Conformers.add `NameError(torch)`. This authoring partition reuses that evidence
without rerunning production code.

Executed static command:
`python /tmp/hotpot_plan20/medium/stage9b-design-20261104/check_global_static.py`.
The receipt `global_static_checks.json` in that directory reports PASS over all
10 owned product files: 146 Molecule method signatures/decorators and bodies
matched the captured source after explicit import relocation and the eight named
delegate changes; all eight state helpers and both frame/transaction helpers
matched; seven observation installations and 17 wrappers were preserved. It also
checked three constant bindings, four signature accessors, nine finite namespace
assignments with one call, 13 scalar delegates, exact Conformers-only override
removal, the unchanged cheminfo export set and the PyG relative target. No
Molecule runtime global reference was left unbound in the static symbol table.
These are stdlib AST/source checks, with file hashes in that receipt; no Hotpot
module was imported or executed. They do not prove runtime initialization,
native dispatch, current pickle graphs, source/installed behavior, plugin imports
or scientific/performance parity. Root must freeze the
integrated bytes, remove core/last consumers and review static readiness before
releasing focused/full-default source execution, affected installed CP311 and
plugin/dataset evidence. No Stage9, Stage10–12, clean-native, other-version,
global science/performance/GPU/Plan12/P20-D026 gate is closed by this record.

## S9B-06 — Pure source identity contract and evidence refresh

2026-11-04; root accepted this bounded correction and released static authoring.
Runtime imports/tests/builds remain held until root's new integrated freeze.
Root reported the first integrated full-default run as 6,661 collected,
6,521 passed, 9 failed, 128 skipped and 3 xfailed, with no error, process leak or
source drift. The retained failure
`tests/test_cheminfo/calculator/test_scientific_state_ownership.py::test_pure_rule_import_closure_never_reaches_an_upper_layer`
exposed a real contract/lifecycle ownership violation. This author inspected the
source graph; the full-run transcript and its provenance remain root-owned.

The route `charge/contracts.py -> structure/snapshots.py -> _identity.py` made
immutable contracts depend on live capture/check code. It reached concrete
Atom/Bond dispatch through `_fields.py`; independently, it reached
`conformer.py -> _commit.py -> structure/__init__.py -> molecule.py`. The
monitor intentionally follows nested/TYPE_CHECKING imports and explicit package
targets, including `from . import _fact_native`; its traversal and forbidden
layers remain unchanged. Moving MolBlock alone cannot close the second route.

The preserved pre-correction source graph and in-memory variants give:

| Variant | Reached Python files | Forbidden import edges |
|---|---:|---:|
| Captured integrated source | 210 | 1,048 |
| Remove MolBlock from `_fields.py` only | 210 | 1,033 |
| Snapshot imports the pure source-identity leaf | 36 | 0 |

Move the one N01 `_SourceIdentity` definition to
`structure/_source_identity.py`, preserving its whole class AST, fields,
`@dataclass(frozen=True, eq=False)` and future-annotation form. The leaf imports
only stdlib declarations. `_identity.py` imports that same class for construction
and retains all 16 lifecycle/canonical functions and module constants unchanged;
`snapshots.py` changes only its import target, preserving every existing class
and annotation. MolBlock remains in `_fields.py`. There is no schema, field,
class-count, native-source or ABI growth.

The private defining/current-pickle path changes from
`hotpot.cheminfo.structure._identity._SourceIdentity` to
`hotpot.cheminfo.structure._source_identity._SourceIdentity`. Constructor shape,
frozen/identity equality, annotation forms, public exports and scientific
algorithms stay unchanged. There is no module override or compatibility route.
The existing canonicalizer labels dataclasses by their real defining module;
this exact private module-tag change is allowed if that type occurs in a
supported canonical value. Registered dependency-domain inputs inspected here
do not contain the identity object itself; their digest parity remains subject
to runtime validation. No broader digest normalization is authorized. The
parser implementation fingerprint must change when root binds the actual new
source bytes, preserving prior certificates as historical evidence.

This author owns the pure leaf, `_identity.py`, `snapshots.py`, the bounded main
plan clarification and this decision. Root owns the five other product import
consumers (`_commit.py`, `_mutation.py`, `_results.py`, `_selection.py` and
`operations/forcefields/working_copy.py`), the two direct test imports, both N01
current-owner budget entries, parser/package bindings, audit, entry/log records
and validation. `_commit._fact_engine` must configure the same sole relocated
class as `source_type`; native construction retains the existing class layout.

Static commands and receipts are retained under
`/tmp/hotpot_plan20/medium/stage9b-source-identity-20261104`. The command
`python /tmp/hotpot_plan20/medium/stage9b-source-identity-20261104/capture_before.py`
preserves the input manifest/source graph. The command
`python /tmp/hotpot_plan20/medium/stage9b-source-identity-20261104/check_static.py`
writes `static_receipt.json` and `variant_graph.json`.
The latter checks the unchanged whole class and lifecycle AST/source, constants,
snapshot declarations, MolBlock/native bytes, monitor traversal and actual
post-correction rule closure. These are stdlib source checks, not Hotpot runtime
execution or source/installed acceptance.

The failed first default run stays recorded. Its pre-correction source freeze
and parser binding cannot certify the corrected tree. Root must finish the
connected consumer/budget/parser migration, freeze the integrated bytes and
refresh affected closure, class/annotation, source-identity, native-dispatch,
current-pickle/lifecycle, full-default, installed CP311 and plugin evidence under
the existing gates. Unchanged artifacts retain only their original source scope;
this correction closes no Stage9/global gate by itself.
