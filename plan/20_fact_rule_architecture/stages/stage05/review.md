# Stage5 independent rules checkpoint

Status: independent implementation and known-input checks are ready for owner
review. Stage5 is **not complete**. The public implicit-H scalar decision in
[contracts/decisions.md](contracts/decisions.md) remains pending. In particular,
`operations.chemistry._implicit_hydrogen_read` is deliberately not implemented;
its scalar and unavailable numeric-export paths must be completed only after
that choice. Five existing scalar-H test nodes remain unexecuted in this
checkpoint selection. This is not a distributable production freeze.

Entry identities and the full normative-document read are preserved in
[entry_review.json](entry_review.json). The branch entry is
`ba5fa6d7357c4fcdb8b2847bb407274a62b4a5a8`. Exact current source/test/native
hashes, commands, environment, outcomes and remaining gates are indexed by
[rules_checkpoint_evidence.json](rules_checkpoint_evidence.json). Root owns
the main plan, stage index, global log/map/budget/allowlist and final validation.

## Approved public changes implemented

`Atom(mol=None, **kwargs)` accepts the reviewed facts and constraint inputs.
`attrs_array`, `update_electron_config`, arbitrary keywords and supplied
electron/valence/aromatic/partial-charge constructor inputs raise TypeError.
Construction remains factual, preserves explicit zero/presence and agreeing
aliases, and burns an allocated UID on failure. The existing private
`Molecule._create_atom_from_array` delegates to the sole factual row importer;
this is not a replacement public constructor or an old-name alias.

Electron configuration, n/s/p/d/f/g, valence and Atom/Ring aromatic projections
are read-only, including the native scalar dispatch. Explicit implicit-H
assignment remains available. The approved `updata_atoms` and full float64
`atom_matrix` spellings remain exact. Known numeric snapshots preserve raw
factual float64 values independently of scalar codecs: H65537.25 remains
65537.25 in snapshots/model features. Scalar `Atom.n` retains the characterized
zero-based shell calculation; the historical numeric n-column retains the
separate neutral electron-configuration table convention.

Ring's aromatic setter, `determine_aromatic` and `kekulize` are removed.
Molecule's complete named operations own assignment. The obsolete
`MolBlock.setattr`, `Atom.setattr`, `update_atoms_attrs_from_id_dict` and
`update_bonds_attrs_from_id_dict` write paths are removed with their callers.
No public exception, alias, validity API or result schema is introduced.

## Ownership and preparation

Seven P4 numeric slots have no Atom storage: n/s/p/d/f/g and valence. Their
presence/source bits are cleared on import and append. Raw19 transport places
NaN in those retired slots; the one public/private field projection supplies
computed values. There is no second persistent19-column array. Partial-charge
ownership and optional-result absence remain Stage7C work.

The four original finite dependency groups remain. VALENCE is an invalidation
fanout bucket, with no stored valence value or cached finding. Setters only
commit facts and queue UID seeds. Named preparation expands upstream seeds
before inspecting status, coalesces pending rows/components, preserves supplied
H declarations, retries UNRESOLVED/CONFLICT after relevant new input and avoids
repeating the unchanged rule. Default-false components do not opt into aromatic
perception because another component has aromatic input. Conflicting components
retain their facts and do not make a separate valid scalar component stale.

The private closed `bonding_inputs` identity binds the numeric/representation
and coordinate facts consumed by Stage5 rules and Open Babel perception.
Opaque Bond annotations and legacy result frames are excluded because these
calculations do not consume them. H proposals validate the whole captured
source before any commit; a later row's calculation cannot invalidate an
earlier proposal unnoticed. Open Babel perception also binds the cell. The
force-field input identity now includes the independent Bond aromatic flag.

Current known H values reach model/native/RDKit/MCA consumers through one named
strict preparation. Batch H addition prepares its heavy-atom selection once.
Missing required input reaches the existing error boundary without zero fill.
The explicit unknown scalar and numeric-export choice remains unimplemented.

## Coupled assignment and immutable views

`structure/snapshots.py` adds exactly two frozen values, `_BondingSnapshot` and
`_AromaticAssignment`. Pure rules return one complete selected Atom/Bond
proposal. The commit journals Atom values, Bond flags/kinds/orders and validity
together, validates source/member coverage and rolls back on failure. The
private complete MoleculeRecord gains trailing `bond_aromatic`; copy,
components, merge, pickle and imports retain Kekule Bond flags independently
of single/double kinds. This adds no carrier class.

Ring holds fixed closed Atom/Bond members plus source/geometry/representation
bindings. JointRing stores fixed first-occurrence tuples (`_member_rings` is
membership, not a Molecule cache), correctly includes member index0 and checks
live-neighbour/index/chemical reads. Changed relevant sources raise the existing
RuntimeError; held member identities remain inspectable. Molecule's four ring
caches and their invalidation/undo/pickle consumers are retired together.

The old six Python `cheminfo/kekulize` files and all five duplicate protocols
are removed. The retained per-ring model, greedy ordering, planarity ratio0.03,
joint check and historical `Ring.has_3d` branch selection are documented in
[Kekulize.md](../../../../hotpot/cheminfo/rules/chemistry/bonding/Kekulize.md).
Neighbour tuples are calculated once per proposal; the scientific formulas
are unchanged. Explicit complete assignment clears flags outside its selected
cycles. Normal kind/order changes follow the old greedy route; the existing
native kind-or-flag encoding remains unchanged. The three strict scientific
xfails remain: ring-order dependence, azulene and multi-edge shared paths.

## Actual checks and assertion mappings

- Current-source Python3.11.16 checkpoint: **477 passed,5 deselected,3 strict
  xfailed,1 GeometryQualityWarning**,3.717891s including recorder overhead.
  The warning is from the existing mocked optimizer's terminal geometry
  observation. The actual command/19 selected paths/five exact exclusions are
  in `local/independent-checkpoint-attempt4/command.json`.
- Current AST/static handoff:664 Python files plus the bound fixture parsed;
  **12 exact retired Stage5 allowlist entries,0 new debt,0 layer findings**.
  Root must remove those12 entries before claiming the global ratchet passes.
  Actual runtime/native/dependency identities are in
  `local/independent-static-attempt2/report.json`.
- B20-bound private observations:7 completed,0 execution errors, all selected
  factual bits/Bond3-f64/endpoints/kinds/flags bound to the B20 prepared input;
  all compared after-values have0 differences. All-single benzene starts H2
  and ends H1. Candidate provenance is separately declared by
  [bound_fixture.py](contracts/bound_fixture.py), never attributed to B20.
  This is bounded evidence, not a public-getter gate or universal chemical
  truth. See `local/bound-fixture-attempt3/{command,report}.json`.
- Four donor regressions retain their original function/parameter identities
  and H1/H2 numerical rule assertions. Their eight actual input paths also
  verify supplied-count preservation: CS assembled/parsed1/0, CP2/1, Se0/0,
  As0/0; all source codes are observed OpenBabel4. See
  `local/supplied-hydrogens-attempt3/{command,report}.json`. This permission
  effect replaces the old unconditional inferred-count overwrite assertion;
  no new xfail or removed node hides the difference.

All `local/` paths above are under
`.plan20-artifacts/20261029/stage5/`. Each run records exact argv, cwd, environment,
source commit/dirty hashes, interpreter/native/input identities, elapsed time
and before/after byte equality. Original failures are retained: erroneous
eq=False test comparisons; an obsolete retired-s-source assertion; the four
donor permission differences; dependency-state capture preceding lazy numeric
preparation; and one real opaque-annotation hashing failure fixed by the closed
numeric dependency domain. A local static-helper syntax failure is also kept.

Storage-worker source backups and static assertion/node-ID records are intact.
After ownership returned, five tests now use the existing private row-import
entry instead of introducing new core imports. Their function/parameter IDs
and assertions remain; current hashes are in this checkpoint evidence.

## Remaining gates

The owner's H scalar choice must close the missing reader and unavailable
attached/detached numeric snapshot handling, then all excluded and connected
current-read tests must run. The complete Stage5 SMARTS/transport/stereo/model
selection, default suite, all five source/debug/current installed-wheel ABI
checks, packaging/native closure and final scientific reports remain root's
gates. The local cp311 rebuilt extension is not five-version/wheel evidence.

The original Stage4 graph/view2.05414x and complete graph-family1.43729x B20
results still block final performance acceptance; no waiver or new performance
measurement is claimed here. Stage6 extraction, unrelated Stage7 ownership,
physical carrier/native path migration and Stage13 are not entered. The class
delta is +2 frozen snapshots−5 retired protocols, net−3 definitions; root owns
the actual budget/allowlist update and final evidence indexing.
