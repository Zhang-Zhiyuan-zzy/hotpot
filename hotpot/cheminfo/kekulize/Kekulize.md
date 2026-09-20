# Aromaticity Perception and Kekule Assignment

## 1. Scope

This subpackage owns Hotpot's active aromaticity-perception and
Kekule-assignment implementation. The name is retained for discoverability,
but its responsibility is broader than assigning alternating single and double
bonds.

The current function bodies are a code relocation: the former per-ring rules,
planarity threshold, and greedy mutation algorithm are preserved. The upstream
ring family intentionally changes from one NetworkX cycle basis to Relevant
Cycles, so molecule-level call count, order, and final output are not claimed
to remain identical for polycyclic graphs. Moving the implementation out of
`core.py` does not certify its chemical validity.

Core chemical objects remain thin facades:

```text
Molecule.determine_rings_aromatic()
    -> kekulize_molecule_rings(mol)

Ring.determine_aromatic(inplace=False)
    -> determine_ring_aromaticity(ring, inplace=False)

Ring.kekulize()
    -> kekulize_ring(ring)

JointRing.check_kekulize()
    -> check_joint_ring_kekulization(joint_ring)

Molecule.calc_atom_valence(assign_aromatic=...)
    -> perceive_ligand_ring_aromaticity(mol, force=...)
```

The RDKit conversion option `Chem.Kekulize`, Open Babel bond-order perception,
and `Molecule.kekulize_smiles` are external conversion concerns and are not
implemented by this package. The empty legacy `Molecule.assign_aromatic()`
placeholder contains no implementation to relocate and remains outside the
subpackage pending public-API cleanup.

## 2. Current package structure

```text
hotpot/cheminfo/kekulize/
├── __init__.py       Public exports
├── _protocols.py     Structural contracts; no runtime import of Core
├── settings.py       Legacy aromaticity constants
├── aromaticity.py    Existing per-ring aromaticity perception
├── assignment.py     Existing per-ring assignment and joint-ring check
├── workflow.py       Existing molecule-level orchestration
└── Kekulize.md       Contract, evidence, limitations, and redesign plan
```

| Module | Current responsibility |
|---|---|
| `_protocols.py` | Defines the minimum Atom/Bond/Ring/Molecule interfaces required by this package and prevents a Core import cycle |
| `settings.py` | Stores `AROMATIC_PLANARITY_RELATIVE_TOLERANCE = 0.03` |
| `aromaticity.py` | Executes the former `Ring.determine_aromatic()` body without changing its rules |
| `assignment.py` | Executes the former greedy `Ring.kekulize()` and `JointRing.check_kekulize()` bodies |
| `workflow.py` | Owns the molecule workflows: ligand-ring perception for valence setup, and hide/process/recover Kekule assignment |

Current public functions:

| Function | Responsibility |
|---|---|
| `determine_ring_aromaticity(ring, inplace=False)` | Apply the legacy per-ring aromaticity rules and optionally write atom flags |
| `perceive_ligand_ring_aromaticity(mol, force=None)` | Preserve the three-state perception gate used before atom-valence calculation |
| `kekulize_ring(ring)` | Apply the legacy greedy bond-order assignment to one ring |
| `kekulize_molecule_rings(mol)` | Hide metal--ligand bonds, process the molecule's Relevant Cycles, then restore the bonds |
| `check_joint_ring_kekulization(joint_ring)` | Apply the legacy atom-level validity check to a joined ring system |
| `kekulize_joint_ring(joint_ring)` | Explicitly retained unimplemented entry point; raises `NotImplementedError` |

## 3. Ring-family contract

The primary Core ring APIs now use Relevant Cycles:

| API | Ring family |
|---|---|
| `mol.rings` | Full-graph Relevant Cycles |
| `mol.ligand_rings` | Ligand-skeleton Relevant Cycles |
| `mol.rings_for_scope(...)` | Relevant Cycles for the requested scope |
| `mol.cycle_basis_rings` | Legacy NetworkX cycle basis |
| `mol.ligand_cycle_basis_rings` | Legacy ligand-skeleton cycle basis |
| `mol.cycle_basis_rings_for_scope(...)` | Legacy cycle basis for the requested scope |

Existing trained ML feature extractors explicitly use
`ligand_cycle_basis_rings`; this prevents the ring tensor count, order, and
shape from silently changing. This guarantee is limited to the ring tensor:
aromatic atom features can still change because perception intentionally
follows the new Relevant Cycle APIs during this transitional phase.

Relevant Cycles make the selected ring family unique for a fixed unweighted
graph and restore symmetry-equivalent rings.  They do not by themselves make
the legacy per-ring aromaticity algorithm correct. `mol.rings` and
`mol.ligand_rings` use a default limit of 10,000 results: they either return the
complete requested family or raise `RelevantCycleLimitExceeded`; they never
return a silently truncated family. `rings_for_scope(..., max_cycles=None)` is
the explicit unbounded form.

## 4. Preserved legacy behavior

The following behavior is frozen during the extraction:

1. A ring whose atoms are already all aromatic returns `True` immediately.
2. A ring containing a metal returns `False`.
3. The existing separate coordinate/no-coordinate electron rules are retained.
4. The strict planarity test remains
   `maximum_deviation / length_scale < 0.03`.
5. `inplace=True` writes one Boolean value to every atom in the current ring.
6. `kekulize_ring()` resets every current ring bond to order one, then greedily
   changes eligible bonds to order two in current bond order.
7. For a given ring sequence, the molecule workflow retains its
   hide/process/recover ordering and does not
   add fallback behavior.
8. `JointRing.kekulize()` remains explicitly unimplemented.

These are compatibility observations, not target behavior.

## 5. Test evidence after switching to Relevant Cycles

Focused integration tests establish the following:

- Core exposes six four-membered Relevant Cycles for cubane, while the legacy
  cycle basis contains five cycles and may contain a six-membered composite.
- Relevant Cycle output and numeric SMARTS results are invariant to bond
  insertion order in the cubane fixture.
- Existing single-ring and coordination-profile SMARTS tests remain valid.
- Fixed golden cases cover benzene, cyclobutane, furan, pyridine, and a
  metal-containing ring; the isolated functions and Core facade also produce
  identical results for the preserved benzene workflow.
- Existing ML ring extraction still emits the five-ring legacy cubane tensor,
  while the public Core API reports six Relevant Cycles.

Three known failures are retained as strict `xfail` tests:

1. Processing the same overlapping Relevant Cycles in reverse order can change
   final atom aromatic flags because a later non-aromatic ring overwrites shared
   atoms.
2. Clearing imported aromatic flags from azulene causes the legacy algorithm to
   reject both its five- and seven-membered Relevant Cycles, missing the global
   ten-electron aromatic system.
3. `Molecule.aromatic_joint_rings` rejects a valid pair of Relevant Cycles that
   shares a multi-edge path because `Ring.joint_with()` only accepts one shared
   edge.

Additional defects deliberately not repaired in this extraction include:

- `Ring.has_3d` compares an atom coordinate vector with an Atom object;
- pre-existing aromatic flags bypass perception;
- per-ring mutation is not transactional;
- greedy assignment is not a global valence-constrained matching;
- the molecule workflow does not recover hidden bonds if an exception interrupts
  processing.

At validation revision `4d847f4`, the repository's focused
inference/cheminformatics compatibility runner passes on both Python 3.9 and
3.14 with **710 passed, 4 skipped, and 3 expected strict xfails**. The strict
SMARTS gate passes **255 tests** on each interpreter. The three xfails above are
executable records of unresolved legacy behavior, not fallbacks or ignored
unexpected failures.

## 6. SMARTS `R/r` impact

SMARTS continues to obtain rings through the selected molecular view.  Both
`FULL_GRAPH` and `LIGAND_SKELETON` now use Relevant Cycles.

| Primitive | Effect of the migration |
|---|---|
| `R`, `R0`, bare `r`, `r0` | Ring/non-ring membership is unchanged for a valid undirected graph |
| `R<n>` | Now counts Relevant Cycles containing the atom; results become deterministic and symmetry-preserving |
| `r<n>` | Now means membership in at least one Relevant Cycle of size `n`; arbitrary basis omissions are removed, and the cubane composite-basis false positive is eliminated |

For cubane, the former basis could give asymmetric `R2/R3` results and make
some atoms match `r6`.  Relevant Cycles make all eight atoms match `R3` and
`r4`, while none match `R2` or `r6`; the committed differential test verifies
these six predicates against RDKit.

The existing Hotpot definition of `r<n>` is “membership in any selected ring of
size `n`”.  Some SMARTS implementations instead use the smallest ring size.
That standards question is independent of the Relevant Cycle migration and
must be decided separately.

## 7. Target architecture

The planned chemically revised package is:

```text
kekulize/
├── __init__.py
├── _protocols.py
├── settings.py
├── model.py          Immutable evidence and result records
├── ring_system.py    Relevant-ring relations and fused-system construction
├── conjugation.py    Atom/bond eligibility and conjugated-subgraph extraction
├── aromaticity.py    Aromaticity-model evaluation without source mutation
├── assignment.py     Global valence-constrained Kekule assignment
├── workflow.py       Explicit perception and assignment orchestration
└── Kekulize.md
```

The intended data flow is:

```text
Molecule topology
  -> Relevant Cycles
  -> ring relations and cyclic biconnected components
  -> conjugation-eligible subgraphs
  -> aromaticity evidence for the whole ring system
  -> one atomic commit of aromatic atom/bond flags
  -> global Kekule bond-order assignment
```

Geometry supplies planarity measurements only.  It must not decide whether a
ring system is aromatic.  Core supplies chemical objects and ring topology but
must not contain the perception or assignment implementation.

## 8. Refactoring plan

### Phase 1: freeze evidence

- Keep the current strict `xfail` cases and add fused, bridged, spiro, charged,
  heteroaromatic, antiaromatic, and non-planar fixtures.
- Compare aromatic atom/bond flags and Kekule feasibility with established
  toolkits, recording model differences instead of silently accepting them.
- Separate parser-provided aromatic flags from recomputed Hotpot results.

### Phase 2: construct ring systems

- Treat Relevant Cycles as the individual-ring family.
- Classify ring pairs by shared atoms and shared edge paths.
- Use cyclic biconnected components as an invariant system boundary so the
  workflow does not depend on enumerating every simple cycle.
- Keep fused, spiro, and bridged relationships distinct.

### Phase 3: identify conjugation

- Determine p-orbital eligibility from element, charge, valence, hydrogen,
  radical state, and bond representation.
- Normalize aromatic, Kekule, and delocalized input representations into the
  same evidence model.
- Do not infer conjugation solely from an existing alternating bond pattern.

### Phase 4: perceive aromaticity

- Evaluate the complete conjugated ring system rather than mutating one ring at
  a time.
- State the supported aromaticity model explicitly; the monocyclic Hückel rule
  must not be applied blindly to total electron counts of arbitrary polycycles.
- Produce an immutable result before changing source Atom or Bond objects.

### Phase 5: assign Kekule bond orders

- Formulate assignment as a global matching/constraint problem over the
  aromatic subgraph.
- Respect atom valence, charge, explicit hydrogens, exocyclic multiple bonds,
  and fused shared bonds.
- Distinguish “non-aromatic” from “aromatic but not kekulizable under the chosen
  representation”.
- Commit bond orders only after a complete valid assignment exists.

### Phase 6: integrate and retire legacy code

- Keep Core methods as thin facades.
- Replace the legacy modules only after differential and chemical tests pass.
- Remove strict `xfail` markers one defect at a time; never turn failures into
  success through broad exception handling or fallback to a cycle basis.

## 9. Completion criteria

The redesign is complete only when:

1. atom and bond order permutations do not change aromaticity or Kekule
   feasibility;
2. imported aromatic and explicit Kekule representations converge to the same
   result;
3. fused and bridged systems are evaluated globally;
4. assignment either returns a valence-valid complete solution or a typed
   failure without partial mutation;
5. SMARTS ring semantics remain separately specified and tested;
6. all supported Python versions import the package without a Core import cycle;
7. model feature extractors retain their explicitly declared ring-family
   contract.
