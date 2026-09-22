# Staged complex untangling refactor

## Verified current behavior

The current complex workflow has two sequential phases:

1. `_build_ligand_proxies()` hides every metal--ligand bond, builds and
   optimizes each ligand component, then restores every coordination bond at
   once.
2. `_OpenBabelOptimizer.optimize()` optimizes the fully connected complex and
   evaluates geometry after each epoch.

The second phase does **not** restart from identical coordinates: one Open
Babel molecule and optimizer state are initialized once, and every epoch
continues from the previous epoch.  Its structural limitation is different:
confirmed bond--ring piercing rejects an observed frame but does not trigger a
topology-changing repair.  Expensive full acceptance scans may therefore be
repeated while an ordinary force-field minimizer cannot escape the topology.

## Target workflow

```text
Stage 1: ligand construction
  OBBuilder once per independent candidate
    -> warm-up optimization
    -> inspect ligand-skeleton bond--ring relations
    -> while PIERCES and attempts < ligand_untangling_attempts (default 20):
         open one eligible ring edge
         perturb coordinates
         short optimization with the ring open
         restore the same edge
         inspect again
    -> retain the closed-topology frame with the lowest confirmed piercing count
    -> medium optimization and final candidate inspection

Stage 2.1: coordination-bond restoration
  hide all original metal--ligand bonds
    -> temporarily restore one pending metal--donor bond
    -> compare full-graph ring relations before and after the addition
    -> keep the bond only when it introduces no confirmed piercing
    -> short optimization after each successful restoration
    -> retry pending bonds
    -> on the first stalled round optimize; on later stalled rounds perturb first
    -> count only stalled rounds against the bounded retry budget
    -> after the budget, restore every still-pending original bond and warn

Stage 2.2: fully connected complex optimization
  apply the Stage-1 open-ring/perturb/restore loop to the complete complex
  (default 30 attempts)
    -> run the stateful final Open Babel relaxation until the first confirmed
       piercing or the requested optimization budget is exhausted
    -> immediately re-enter the repair loop after a piercing; continue with
       the remaining epoch budget after the topology has been restored
    -> retain the final frame, or the full trace when `save_movie=True`
```

## Semantic boundaries

- Geometry reports spatial facts only.  Forcefields owns the chemical policy.
- Covalent-ring repair uses `ring_scope="ligand_skeleton"`; coordination-bond
  admission compares pre-addition and post-addition `full_graph` reports.
  Both use Relevant Cycles up to size 16.  Excluded larger rings produce a
  warning but do not trigger repair.
- A coordination bond is first restored transactionally so ring perception
  sees every chelate cycle that the edge creates.  Only new relation keys are
  considered.  A narrow forcefield-only rule ignores a finding when the
  reported bond is the proposed coordination bond and the ring contains both
  endpoints; other bonds piercing that newly closed ring remain failures.
- `UNDETERMINED` is recorded but is not actively opened.  Only confirmed
  `PIERCES` relations trigger ring opening.
- Topology changes always require a fresh Open Babel force-field setup.
- Independent candidate count, independent build-attempt budget, ligand
  untangling budget, coordination restoration budget, and full-complex
  untangling budget are separate controls.
- Exhausting an untangling budget is a warning, not a topology-loss error.
  The ligand stage uses the closed-topology frame with the lowest confirmed
  piercing count; coordination restoration always restores the complete
  original metal--ligand topology before Stage 2.2.
- `coordination_restoration_attempts` limits stalled retry rounds, not the
  number of coordination bonds.  Every safely restored bond receives its own
  relaxation and does not consume the stalled retry budget.
- If the final requested epoch creates a piercing, one additional stabilization
  epoch may follow its repair.  `epochs` is therefore the normal relaxation
  budget; repair and post-repair stabilization work is reported in addition.

## Commit checkpoints

1. Add a bond-identity-preserving selective restore primitive and tests.
2. Add immutable workflow reports and shared ring-piercing repair helpers.
3. Replace ligand rebuild-on-piercing with bounded perturbative untangling.
4. Add incremental coordination-bond restoration with chelate-safe semantics.
5. Add full-complex untangling, trace preservation, and public parameter flow.
6. Run focused, integration, slow-case, lint, and compatibility tests.
